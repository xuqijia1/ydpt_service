#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
昇腾310P3 VDEC+VPC全硬件解码测试脚本
摄像头RTSP：rtsp://admin:tykj1357@192.168.13.115
修复：回调参数不匹配、内存泄漏、解码无输出问题
"""
import av
import acl
import numpy as np
import cv2
import sys
import os
import time
import traceback

# ===================== 全局常量 =====================
FMT_NV12 = 1
FMT_BGR  = 13
ACL_H2D = 1
ACL_D2H = 2
ENTYPE_H265_MAIN = 0
ENTYPE_H264_MAIN = 2
ENTYPE_H264_HIGH = 3

# 固定摄像头RTSP地址
RTSP_URL = "rtsp://admin:tykj1357@192.168.13.115:554/Streaming/Channels/101"
NPU_DEV_ID = int(os.environ.get("NPU_DEV_ID", 0))
TARGET_W = 640
TARGET_H = 640

# ===================== 工具函数 =====================
def align_up(val: int, alignment: int) -> int:
    return ((val + alignment - 1) // alignment) * alignment

def get_h265_nal_type(nal_data: bytes) -> int:
    if len(nal_data) < 5:
        return -1
    if nal_data[:4] == b"\x00\x00\x00\x01":
        return (nal_data[4] >> 1) & 0x3F
    elif nal_data[:3] == b"\x00\x00\x01":
        return (nal_data[3] >> 1) & 0x3F
    return -1

# ===================== 全局状态 =====================
vdec_run_flag = [True]
callback_frame_list = []
callback_count = [0]
global_context = None
global_stream = None

# ========== 修复：标准3参数VDEC回调函数 ==========
def vdec_decode_callback(stream_desc, pic_desc, user_data):
    try:
        callback_count[0] += 1
        # 销毁码流描述符
        if stream_desc is not None:
            acl.media.dvpp_destroy_stream_desc(stream_desc)
        if pic_desc is None:
            return

        ret_code = acl.media.dvpp_get_pic_desc_ret_code(pic_desc)
        dev_data = acl.media.dvpp_get_pic_desc_data(pic_desc)
        data_size = acl.media.dvpp_get_pic_desc_size(pic_desc)

        # user_data >=0 正常图像帧，负数为序列头空帧
        if ret_code == 0 and dev_data is not None and data_size > 0 and user_data >= 0:
            callback_frame_list.append({"dev_buf": dev_data, "buf_size": data_size})
            if callback_count[0] <= 10:
                print(f"[VDEC回调成功 #{callback_count[0]} 帧大小:{data_size}")
        # 销毁pic_desc，驱动自动回收内存，不手动free
        acl.media.dvpp_destroy_pic_desc(pic_desc)
    except Exception as e:
        print(f"[VDEC回调异常] {traceback.format_exc()}")

def report_process_thread(args):
    timeout_ms = args[0]
    acl.rt.set_context(global_context)
    print("[消息线程] 上下文绑定完成")
    loop_cnt = 0
    while vdec_run_flag[0]:
        acl.rt.process_report(timeout_ms)
        loop_cnt += 1
        if loop_cnt % 8000 == 0:
            print(f"[消息线程] 轮询{loop_cnt} 已解码:{callback_count[0]}")
    print("[消息线程] 退出循环")

# ===================== Step1 RTSP拉流 =====================
print("="*60)
print(f"拉流地址：{RTSP_URL}")
print("="*60)
rtsp_container = av.open(RTSP_URL, options={
    "rtsp_transport": "tcp",
    "stimeout": "5000000",
    "buffer_size": "1048576"
})
video_stream = rtsp_container.streams.video[0]
src_width = video_stream.codec_context.width
src_height = video_stream.codec_context.height
codec_type = video_stream.codec_context.name
is_h265 = codec_type in ("hevc", "h265")
encode_type = ENTYPE_H265_MAIN if is_h265 else ENTYPE_H264_MAIN
print(f"原始分辨率 {src_width}×{src_height} 编码:{codec_type}")

sps_pps_list = []
image_nal_list = []
max_read_packet = 80
for pkt in rtsp_container.demux(video_stream):
    nal_bytes = bytes(pkt)
    if len(nal_bytes) == 0:
        continue
    if is_h265:
        nal_type = get_h265_nal_type(nal_bytes)
        if nal_type in (32, 33, 34):
            sps_pps_list.append(nal_bytes)
            print(f"捕获SPS/PPS 长度{len(nal_bytes)}")
            continue
    image_nal_list.append(nal_bytes)
    if len(image_nal_list) >= max_read_packet:
        break
print(f"序列头:{len(sps_pps_list)} 图像NAL:{len(image_nal_list)}")
if len(image_nal_list) == 0:
    print("未读取视频码流，退出")
    sys.exit(1)

# ===================== Step2 ACL & VDEC初始化 =====================
print("\n" + "="*60)
print(f"绑定NPU设备{NPU_DEV_ID}")
print("="*60)
ret = acl.init()
if ret != 0:
    print(f"acl.init失败 ret={ret}")
    sys.exit(1)
ret = acl.rt.set_device(NPU_DEV_ID)
if ret != 0:
    print(f"set_device失败 ret={ret}")
    sys.exit(1)
global_context, ctx_ret = acl.rt.create_context(NPU_DEV_ID)
if ctx_ret != 0:
    print("创建context失败")
    sys.exit(1)
global_stream, stream_ret = acl.rt.create_stream()
if stream_ret != 0:
    print("创建stream失败")
    sys.exit(1)

# 启动消息线程，先subscribe再创建VDEC通道
thread_id, thread_ret = acl.util.start_thread(report_process_thread, [100])
acl.rt.subscribe_report(thread_id, global_stream)
time.sleep(0.3)

# 创建VDEC通道
vdec_chn_desc = acl.media.vdec_create_channel_desc()
acl.media.vdec_set_channel_desc_channel_id(vdec_chn_desc, 0)
acl.media.vdec_set_channel_desc_thread_id(vdec_chn_desc, thread_id)
acl.media.vdec_set_channel_desc_callback(vdec_chn_desc, vdec_decode_callback)
acl.media.vdec_set_channel_desc_entype(vdec_chn_desc, encode_type)
acl.media.vdec_set_channel_desc_out_pic_format(vdec_chn_desc, FMT_NV12)
vdec_create_ret = acl.media.vdec_create_channel(vdec_chn_desc)
if vdec_create_ret != 0:
    for try_type in [ENTYPE_H264_HIGH, ENTYPE_H264_MAIN, ENTYPE_H265_MAIN]:
        if try_type == encode_type:
            continue
        acl.media.vdec_set_channel_desc_entype(vdec_chn_desc, try_type)
        vdec_create_ret = acl.media.vdec_create_channel(vdec_chn_desc)
        if vdec_create_ret == 0:
            encode_type = try_type
            break
if vdec_create_ret != 0:
    print("VDEC通道创建失败")
    sys.exit(1)
frame_cfg = acl.media.vdec_create_frame_config()
align_w = align_up(src_width, 16)
align_h = align_up(src_height, 2)
nv12_frame_size = align_w * align_h * 3 // 2
max_nal_size = max([len(p) for p in image_nal_list + sps_pps_list])
# 全局复用NAL输入buffer，循环不重复malloc
host_nal_buf, _ = acl.media.dvpp_malloc(max_nal_size + 2048)

# ===================== Step3 发送码流 =====================
print("\n" + "="*60)
print("向VDEC发送码流")
print("="*60)
send_cnt = 0
# 1. 发送SPS/PPS序列头
for idx, hdr_data in enumerate(sps_pps_list):
    hdr_np = np.frombuffer(hdr_data, dtype=np.uint8)
    hdr_len = hdr_np.nbytes
    acl.rt.memcpy(int(host_nal_buf), hdr_len, int(hdr_np.ctypes.data), hdr_len, ACL_H2D)
    stream_desc = acl.media.dvpp_create_stream_desc()
    acl.media.dvpp_set_stream_desc_data(stream_desc, host_nal_buf)
    acl.media.dvpp_set_stream_desc_size(stream_desc, hdr_len)
    # pic_desc交给驱动管理，无需手动free
    dev_nv12_buf, _ = acl.media.dvpp_malloc(nv12_frame_size)
    pic_desc = acl.media.dvpp_create_pic_desc()
    acl.media.dvpp_set_pic_desc_data(pic_desc, dev_nv12_buf)
    acl.media.dvpp_set_pic_desc_size(pic_desc, nv12_frame_size)
    acl.media.dvpp_set_pic_desc_format(pic_desc, FMT_NV12)
    send_ret = acl.media.vdec_send_frame(vdec_chn_desc, stream_desc, pic_desc, frame_cfg, -(idx+1))
    print(f"序列头{idx} send_ret={send_ret}")
    # 不 destroy — VDEC 异步获取所有权，回调中负责 destroy
    time.sleep(0.05)

# 2. 发送图像NAL包
for idx, nal_data in enumerate(image_nal_list):
    nal_np = np.frombuffer(nal_data, dtype=np.uint8)
    nal_len = nal_np.nbytes
    acl.rt.memcpy(int(host_nal_buf), nal_len, int(nal_np.ctypes.data), nal_len, ACL_H2D)
    stream_desc = acl.media.dvpp_create_stream_desc()
    acl.media.dvpp_set_stream_desc_data(stream_desc, host_nal_buf)
    acl.media.dvpp_set_stream_desc_size(stream_desc, nal_len)
    dev_nv12_buf, _ = acl.media.dvpp_malloc(nv12_frame_size)
    pic_desc = acl.media.dvpp_create_pic_desc()
    acl.media.dvpp_set_pic_desc_data(pic_desc, dev_nv12_buf)
    acl.media.dvpp_set_pic_desc_size(pic_desc, nv12_frame_size)
    acl.media.dvpp_set_pic_desc_format(pic_desc, FMT_NV12)
    send_ret = acl.media.vdec_send_frame(vdec_chn_desc, stream_desc, pic_desc, frame_cfg, idx)
    if send_ret == 0:
        send_cnt += 1
    # 不 free dev_nv12_buf / pic_desc / stream_desc — VDEC 异步获取所有权，回调中负责 destroy
    if send_cnt >= 30:
        break
    time.sleep(0.04)
print(f"成功发送{send_cnt}帧图像，等待解码")

# 等待解码帧
wait_sec = 0
while wait_sec < 10 and len(callback_frame_list) < 2:
    time.sleep(1)
    wait_sec += 1
    print(f"等待{wait_sec}s | 回调计数:{callback_count[0]} 有效帧:{len(callback_frame_list)}")

# ===================== Step4 VPC硬件处理 =====================
if len(callback_frame_list) > 0:
    print("\n" + "="*60)
    print("VPC硬件缩放转色")
    print("="*60)
    first_frame = callback_frame_list[0]
    nv12_dev_buf = first_frame["dev_buf"]
    vpc_stream, _ = acl.rt.create_stream()
    vpc_chn_desc = acl.media.dvpp_create_channel_desc()
    acl.media.dvpp_create_channel(vpc_chn_desc)

    # VPC输入NV12描述
    vpc_in_desc = acl.media.dvpp_create_pic_desc()
    acl.media.dvpp_set_pic_desc_data(vpc_in_desc, nv12_dev_buf)
    acl.media.dvpp_set_pic_desc_format(vpc_in_desc, FMT_NV12)
    acl.media.dvpp_set_pic_desc_width(vpc_in_desc, src_width)
    acl.media.dvpp_set_pic_desc_height(vpc_in_desc, src_height)
    acl.media.dvpp_set_pic_desc_width_stride(vpc_in_desc, align_w)
    acl.media.dvpp_set_pic_desc_height_stride(vpc_in_desc, align_h)
    acl.media.dvpp_set_pic_desc_size(vpc_in_desc, nv12_frame_size)

    # 缩放输出
    target_align_w = align_up(TARGET_W, 16)
    target_align_h = align_up(TARGET_H, 2)
    target_nv12_size = target_align_w * target_align_h * 3 // 2
    dev_resize_nv12, _ = acl.media.dvpp_malloc(target_nv12_size)
    resize_desc = acl.media.dvpp_create_pic_desc()
    acl.media.dvpp_set_pic_desc_data(resize_desc, dev_resize_nv12)
    acl.media.dvpp_set_pic_desc_format(resize_desc, FMT_NV12)
    acl.media.dvpp_set_pic_desc_width(resize_desc, TARGET_W)
    acl.media.dvpp_set_pic_desc_height(resize_desc, TARGET_H)
    acl.media.dvpp_set_pic_desc_width_stride(resize_desc, target_align_w)
    acl.media.dvpp_set_pic_desc_height_stride(resize_desc, target_align_h)
    acl.media.dvpp_set_pic_desc_size(resize_desc, target_nv12_size)

    roi_cfg = acl.media.dvpp_create_roi_config(0, src_width-1, 0, src_height-1)
    resize_cfg = acl.media.dvpp_create_resize_config()
    acl.media.dvpp_set_resize_config_interpolation(resize_cfg, 0)
    crop_ret = acl.media.dvpp_vpc_crop_resize_async(vpc_chn_desc, vpc_in_desc, resize_desc, roi_cfg, resize_cfg, vpc_stream)
    acl.rt.synchronize_stream(vpc_stream)
    print(f"缩放结果:{crop_ret}")

    # NV12转BGR
    bgr_align_w = align_up(TARGET_W, 16) * 3
    bgr_align_h = align_up(TARGET_H, 2)
    bgr_total_size = bgr_align_w * bgr_align_h
    dev_bgr, _ = acl.media.dvpp_malloc(bgr_total_size)
    bgr_desc = acl.media.dvpp_create_pic_desc()
    acl.media.dvpp_set_pic_desc_data(bgr_desc, dev_bgr)
    acl.media.dvpp_set_pic_desc_format(bgr_desc, FMT_BGR)
    acl.media.dvpp_set_pic_desc_width(bgr_desc, TARGET_W)
    acl.media.dvpp_set_pic_desc_height(bgr_desc, TARGET_H)
    acl.media.dvpp_set_pic_desc_width_stride(bgr_desc, bgr_align_w)
    acl.media.dvpp_set_pic_desc_height_stride(bgr_desc, bgr_align_h)
    acl.media.dvpp_set_pic_desc_size(bgr_desc, bgr_total_size)
    color_ret = acl.media.dvpp_vpc_convert_color_async(vpc_chn_desc, resize_desc, bgr_desc, vpc_stream)
    acl.rt.synchronize_stream(vpc_stream)
    print(f"色域转换结果:{color_ret}")

    # 测试保存图片
    host_bgr = np.zeros(bgr_total_size, dtype=np.uint8)
    acl.rt.memcpy(int(host_bgr.ctypes.data), bgr_total_size, int(dev_bgr), bgr_total_size, ACL_D2H)
    img = host_bgr.reshape(bgr_align_h, bgr_align_w//3, 3)[:TARGET_H, :TARGET_W, :]
    save_path = f"/tmp/cam_dev{NPU_DEV_ID}.jpg"
    cv2.imwrite(save_path, img)
    print(f"图片保存:{save_path}")

    # VPC资源释放
    acl.media.dvpp_free(dev_resize_nv12)
    acl.media.dvpp_free(dev_bgr)
    acl.media.dvpp_destroy_pic_desc(vpc_in_desc)
    acl.media.dvpp_destroy_pic_desc(resize_desc)
    acl.media.dvpp_destroy_pic_desc(bgr_desc)
    acl.media.dvpp_destroy_roi_config(roi_cfg)
    acl.media.dvpp_destroy_resize_config(resize_cfg)
    acl.media.dvpp_destroy_channel(vpc_chn_desc)
    acl.rt.destroy_stream(vpc_stream)
else:
    print("VDEC未输出解码帧")

# ===================== 统一释放所有资源 =====================
print("\n释放硬件资源")
vdec_run_flag[0] = False
time.sleep(0.5)
# 停止线程
try:
    acl.util.stop_thread(thread_id)
except:
    pass
# 释放解码帧内存 — VDEC 驱动自动回收，不手动 free
# callback_frame_list 只保存指针，不需要 dvpp_free
# 销毁VDEC资源
acl.media.vdec_destroy_channel(vdec_chn_desc)
acl.media.vdec_destroy_frame_config(frame_cfg)
acl.media.dvpp_free(host_nal_buf)
# 运行时销毁
acl.rt.destroy_stream(global_stream)
acl.rt.destroy_context(global_context)
acl.rt.reset_device(NPU_DEV_ID)
acl.finalize()
rtsp_container.close()
print("测试脚本执行完毕")