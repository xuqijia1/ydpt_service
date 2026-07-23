#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
昇腾310P3 VDEC+VPC 全 NPU 链路测试

严格参照 SDK 示例 acl_vdec.py 的内存管理：
- 回调中只 destroy_pic_desc，不 dvpp_free(pic_data)
- pic_data 由主线程在用完后 dvpp_free
- input_buf 全局复用，回调中只 destroy_stream_desc 不 free data
"""
import av
import acl
import numpy as np
import cv2
import sys
import os
import time
import traceback

FMT_NV12 = 1
FMT_BGR  = 13
ACL_H2D = 1
ACL_D2H = 2
ENTYPE_H265_MAIN = 0
ENTYPE_H264_MAIN = 2
ENTYPE_H264_HIGH = 3

RTSP_URL = os.environ.get('RTSP_URL', 'rtsp://admin:tykj1357@192.168.13.115:554/Streaming/Channels/101')
NPU_DEV_ID = int(os.environ.get("NPU_DEV_ID", 0))
TARGET_W = 640
TARGET_H = 640

def align_up(val, alignment):
    return ((val + alignment - 1) // alignment) * alignment

def get_h265_nal_type(data):
    if len(data) < 5:
        return -1
    if data[:4] == b"\x00\x00\x00\x01":
        return (data[4] >> 1) & 0x3F
    elif data[:3] == b"\x00\x00\x01":
        return (data[3] >> 1) & 0x3F
    return -1

# ==================== 全局状态 ====================
vdec_run = True
cb_count = [0]
cb_frames = []  # [{"buffer": ptr, "size": int}, ...]
global_ctx = None
global_stream = None

# 回调延迟销毁队列：回调中只取数据+存 desc，主线程统一 destroy
cb_desc_queue = []  # [(stream_desc, pic_desc), ...]

# VDEC 回调 — 取数据，延迟销毁 desc
def vdec_callback(stream_desc, pic_desc, user_data):
    cb_count[0] += 1
    try:
        # 延迟销毁：存到队列，主线程统一 destroy（避免回调线程中 double free）
        cb_desc_queue.append((stream_desc, pic_desc))

        # 输出 pic_desc：提取数据
        if pic_desc is not None:
            ret_code = acl.media.dvpp_get_pic_desc_ret_code(pic_desc)
            pic_data = acl.media.dvpp_get_pic_desc_data(pic_desc)
            pic_size = acl.media.dvpp_get_pic_desc_size(pic_desc)

            if cb_count[0] <= 10:
                print(f"  [回调] #{cb_count[0]}: ret_code={ret_code}, size={pic_size}, user_data={user_data}")

            if ret_code == 0 and pic_data is not None and pic_size > 0 and user_data is not None and user_data >= 0:
                # 有效图像帧 — 保存 buffer 指针（主线程最后 dvpp_free）
                cb_frames.append({"buffer": pic_data, "size": pic_size})
            else:
                # 序列头空帧 / 解码失败 — 释放 device buffer
                if pic_data is not None:
                    acl.media.dvpp_free(pic_data)
    except Exception as e:
        print(f"  [回调异常] {e}")

def cb_thread_func(args):
    timeout = args[0] if args else 100
    acl.rt.set_context(global_ctx)
    print("  [cb_thread] started")
    loop = 0
    while vdec_run:
        acl.rt.process_report(timeout)
        loop += 1
        if loop % 5000 == 0:
            print(f"  [cb_thread] loop={loop}, cb={cb_count[0]}")
    print(f"  [cb_thread] exit")

# ==================== Step 1: PyAV ====================
print("=" * 60)
print("Step 1: PyAV 拉 RTSP")
print("=" * 60)
container = av.open(RTSP_URL, options={'rtsp_transport': 'tcp', 'stimeout': '5000000'})
stream = container.streams.video[0]
SRC_W = stream.codec_context.width
SRC_H = stream.codec_context.height
codec_name = stream.codec_context.name
is_h265 = codec_name in ('hevc', 'h265')
en_type = ENTYPE_H265_MAIN if is_h265 else ENTYPE_H264_MAIN
print(f"  {SRC_W}x{SRC_H}, codec={codec_name}, en_type={en_type}")

nal_list = []
seq_headers = []
for packet in container.demux([stream]):
    nal = bytes(packet)
    if len(nal) == 0:
        continue
    if is_h265:
        nt = get_h265_nal_type(nal)
        if nt in (32, 33, 34):
            seq_headers.append(nal)
            print(f"  序列头: NAL type={nt}, size={len(nal)}")
            continue
    nal_list.append(nal)
    if len(nal_list) >= 50:
        break
print(f"  序列头: {len(seq_headers)}, 图像帧: {len(nal_list)}")

# ==================== Step 2: ACL + VDEC ====================
print("\n" + "=" * 60)
print("Step 2: ACL init + VDEC")
print("=" * 60)

acl.init()
acl.rt.set_device(NPU_DEV_ID)
global_ctx, ret = acl.rt.create_context(NPU_DEV_ID)
print(f"  context: ret={ret}")

global_stream, ret = acl.rt.create_stream()
print(f"  stream: ret={ret}")

# 启动回调线程
cb_tid, ret = acl.util.start_thread(cb_thread_func, [100])
print(f"  start_thread: ret={ret}")

# subscribe_report
ret = acl.rt.subscribe_report(cb_tid, global_stream)
print(f"  subscribe_report: ret={ret}")
time.sleep(0.3)

# 创建 VDEC 通道
ch_desc = acl.media.vdec_create_channel_desc()
acl.media.vdec_set_channel_desc_channel_id(ch_desc, 0)
acl.media.vdec_set_channel_desc_thread_id(ch_desc, cb_tid)
acl.media.vdec_set_channel_desc_callback(ch_desc, vdec_callback)
acl.media.vdec_set_channel_desc_entype(ch_desc, en_type)
acl.media.vdec_set_channel_desc_out_pic_format(ch_desc, FMT_NV12)
out_mode = acl.media.vdec_get_channel_desc_out_mode(ch_desc)
acl.media.vdec_set_channel_desc_out_mode(ch_desc, out_mode)

ret = acl.media.vdec_create_channel(ch_desc)
print(f"  vdec_create_channel: ret={ret}")
if ret != 0:
    for alt in [ENTYPE_H264_HIGH, ENTYPE_H264_MAIN]:
        if alt == en_type:
            continue
        acl.media.vdec_set_channel_desc_entype(ch_desc, alt)
        ret = acl.media.vdec_create_channel(ch_desc)
        print(f"  retry en_type={alt}: ret={ret}")
        if ret == 0:
            break
    if ret != 0:
        print("FAIL: vdec_create_channel")
        sys.exit(1)

frame_cfg = acl.media.vdec_create_frame_config()
vdec_w = align_up(SRC_W, 16)
vdec_h = align_up(SRC_H, 2)
vdec_size = vdec_w * vdec_h * 3 // 2

max_nal = max(len(p) for p in nal_list) if nal_list else vdec_size
if seq_headers:
    max_nal = max(max_nal, max(len(h) for h in seq_headers))
input_buf, ret = acl.media.dvpp_malloc(max_nal + 2048)
print(f"  input_buf: ret={ret}")

# ==================== Step 3: 发送 NAL ====================
print("\n" + "=" * 60)
print("Step 3: 发送 NAL")
print("=" * 60)

sent = 0

def flush_desc_queue():
    """在主线程中统一销毁回调存入的 stream_desc/pic_desc"""
    while cb_desc_queue:
        sd, pd = cb_desc_queue.pop(0)
        if sd is not None:
            acl.media.dvpp_destroy_stream_desc(sd)
        if pd is not None:
            acl.media.dvpp_destroy_pic_desc(pd)

# 发送序列头
for hi, hdr in enumerate(seq_headers):
    hdr_np = np.frombuffer(hdr, dtype=np.uint8)
    hdr_len = hdr_np.nbytes
    acl.rt.memcpy(int(input_buf), hdr_len, int(hdr_np.ctypes.data), hdr_len, ACL_H2D)
    sd = acl.media.dvpp_create_stream_desc()
    acl.media.dvpp_set_stream_desc_data(sd, input_buf)
    acl.media.dvpp_set_stream_desc_size(sd, hdr_len)
    dev_out, _ = acl.media.dvpp_malloc(vdec_size)
    pd = acl.media.dvpp_create_pic_desc()
    acl.media.dvpp_set_pic_desc_data(pd, dev_out)
    acl.media.dvpp_set_pic_desc_size(pd, vdec_size)
    acl.media.dvpp_set_pic_desc_format(pd, FMT_NV12)
    ret = acl.media.vdec_send_frame(ch_desc, sd, pd, frame_cfg, -(hi + 1))
    print(f"  序列头 #{hi}: size={hdr_len}, ret={ret}")
    if ret != 0:
        # send_frame 失败，VDEC 未获取所有权，需手动释放
        acl.media.dvpp_destroy_stream_desc(sd)
        acl.media.dvpp_destroy_pic_desc(pd)
        acl.media.dvpp_free(dev_out)
    time.sleep(0.1)

# 发送图像帧
for i, nal_raw in enumerate(nal_list):
    nal_np = np.frombuffer(nal_raw, dtype=np.uint8)
    nal_len = nal_np.nbytes
    acl.rt.memcpy(int(input_buf), nal_len, int(nal_np.ctypes.data), nal_len, ACL_H2D)
    sd = acl.media.dvpp_create_stream_desc()
    acl.media.dvpp_set_stream_desc_data(sd, input_buf)
    acl.media.dvpp_set_stream_desc_size(sd, nal_len)
    dev_out, _ = acl.media.dvpp_malloc(vdec_size)
    pd = acl.media.dvpp_create_pic_desc()
    acl.media.dvpp_set_pic_desc_data(pd, dev_out)
    acl.media.dvpp_set_pic_desc_size(pd, vdec_size)
    acl.media.dvpp_set_pic_desc_format(pd, FMT_NV12)
    ret = acl.media.vdec_send_frame(ch_desc, sd, pd, frame_cfg, i)
    if ret == 0:
        sent += 1
    else:
        print(f"  send_frame failed: ret={ret}, pkt={i}")
        acl.media.dvpp_destroy_stream_desc(sd)
        acl.media.dvpp_destroy_pic_desc(pd)
        acl.media.dvpp_free(dev_out)
    if i == 0:
        print(f"  首帧: size={nal_len}, ret={ret}")
    if sent >= 30:
        break
    time.sleep(0.04)

print(f"  sent={sent}, 等待回调...")

# 主线程统一销毁回调存入的 desc（避免回调线程中 double free）
flush_desc_queue()

# 等待
for wait in range(10):
    time.sleep(1)
    print(f"  {wait+1}s: cb={cb_count[0]}, frames={len(cb_frames)}")
    if len(cb_frames) >= 2:
        time.sleep(2)
        break

print(f"\n结果: cb={cb_count[0]}, frames={len(cb_frames)}")

if len(cb_frames) > 0:
    f0 = cb_frames[0]
    # 验证 VDEC 输出
    verify = np.zeros(min(1024, f0['size']), dtype=np.uint8)
    acl.rt.memcpy(int(verify.ctypes.data), len(verify), int(f0['buffer']), len(verify), ACL_D2H)
    print(f"  VDEC 输出: size={f0['size']}, nonzero={np.count_nonzero(verify)}")
    print("  VDEC SUCCESS!")

    # ==================== Step 4: VPC ====================
    print("\n" + "=" * 60)
    print("Step 4: VPC crop_resize + convert_color")
    print("=" * 60)

    vpc_stream, _ = acl.rt.create_stream()
    vpc_ch = acl.media.dvpp_create_channel_desc()
    ret = acl.media.dvpp_create_channel(vpc_ch)
    print(f"  VPC channel: ret={ret}")

    vpc_in = acl.media.dvpp_create_pic_desc()
    acl.media.dvpp_set_pic_desc_data(vpc_in, f0['buffer'])
    acl.media.dvpp_set_pic_desc_format(vpc_in, FMT_NV12)
    acl.media.dvpp_set_pic_desc_width(vpc_in, SRC_W)
    acl.media.dvpp_set_pic_desc_height(vpc_in, SRC_H)
    acl.media.dvpp_set_pic_desc_width_stride(vpc_in, vdec_w)
    acl.media.dvpp_set_pic_desc_height_stride(vpc_in, vdec_h)
    acl.media.dvpp_set_pic_desc_size(vpc_in, f0['size'])

    DST_W, DST_H = TARGET_W, TARGET_H
    rsz_w = align_up(DST_W, 16)
    rsz_h = align_up(DST_H, 2)
    rsz_size = rsz_w * rsz_h * 3 // 2

    dev_rsz, _ = acl.media.dvpp_malloc(rsz_size)
    acl.rt.memset(int(dev_rsz), rsz_size, 0, rsz_size)
    rsz_desc = acl.media.dvpp_create_pic_desc()
    acl.media.dvpp_set_pic_desc_data(rsz_desc, dev_rsz)
    acl.media.dvpp_set_pic_desc_format(rsz_desc, FMT_NV12)
    acl.media.dvpp_set_pic_desc_width(rsz_desc, DST_W)
    acl.media.dvpp_set_pic_desc_height(rsz_desc, DST_H)
    acl.media.dvpp_set_pic_desc_width_stride(rsz_desc, rsz_w)
    acl.media.dvpp_set_pic_desc_height_stride(rsz_desc, rsz_h)
    acl.media.dvpp_set_pic_desc_size(rsz_desc, rsz_size)

    roi = acl.media.dvpp_create_roi_config(0, SRC_W - 1, 0, SRC_H - 1)
    rcfg = acl.media.dvpp_create_resize_config()
    acl.media.dvpp_set_resize_config_interpolation(rcfg, 0)

    ret = acl.media.dvpp_vpc_crop_resize_async(vpc_ch, vpc_in, rsz_desc, roi, rcfg, vpc_stream)
    acl.rt.synchronize_stream(vpc_stream)
    print(f"  crop_resize: ret={ret}")

    bgr_w = align_up(DST_W, 16) * 3
    bgr_h = align_up(DST_H, 2)
    bgr_size = bgr_w * bgr_h
    dev_bgr, _ = acl.media.dvpp_malloc(bgr_size)
    acl.rt.memset(int(dev_bgr), bgr_size, 0, bgr_size)
    bgr_desc = acl.media.dvpp_create_pic_desc()
    acl.media.dvpp_set_pic_desc_data(bgr_desc, dev_bgr)
    acl.media.dvpp_set_pic_desc_format(bgr_desc, FMT_BGR)
    acl.media.dvpp_set_pic_desc_width(bgr_desc, DST_W)
    acl.media.dvpp_set_pic_desc_height(bgr_desc, DST_H)
    acl.media.dvpp_set_pic_desc_width_stride(bgr_desc, bgr_w)
    acl.media.dvpp_set_pic_desc_height_stride(bgr_desc, bgr_h)
    acl.media.dvpp_set_pic_desc_size(bgr_desc, bgr_size)

    ret = acl.media.dvpp_vpc_convert_color_async(vpc_ch, rsz_desc, bgr_desc, vpc_stream)
    acl.rt.synchronize_stream(vpc_stream)
    print(f"  convert_color: ret={ret}")

    bgr_buf = np.zeros(bgr_size, dtype=np.uint8)
    acl.rt.memcpy(int(bgr_buf.ctypes.data), bgr_size, int(dev_bgr), bgr_size, ACL_D2H)
    bgr_nz = np.count_nonzero(bgr_buf)
    print(f"  BGR nonzero: {bgr_nz}/{bgr_size}")

    if bgr_nz > 0:
        bgr_frame = bgr_buf[:DST_H * bgr_w].reshape(DST_H, bgr_w // 3, 3)[:, :DST_W, :].copy()
        print(f"  BGR: mean={bgr_frame.mean():.1f}, shape={bgr_frame.shape}")
        cv2.imwrite('/tmp/test_npu_pipeline.jpg', bgr_frame)
        print("  保存 /tmp/test_npu_pipeline.jpg")
        print("  全 NPU 链路 SUCCESS!")

    # 清理 VPC
    acl.media.dvpp_free(dev_rsz)
    acl.media.dvpp_free(dev_bgr)
    acl.media.dvpp_destroy_pic_desc(vpc_in)
    acl.media.dvpp_destroy_pic_desc(rsz_desc)
    acl.media.dvpp_destroy_pic_desc(bgr_desc)
    acl.media.dvpp_destroy_resize_config(rcfg)
    acl.media.dvpp_destroy_roi_config(roi)
    acl.media.dvpp_destroy_channel(vpc_ch)
    acl.rt.destroy_stream(vpc_stream)

    # 释放 VDEC 帧 — SDK 示例：主线程最后 dvpp_free
    for f in cb_frames:
        acl.media.dvpp_free(f['buffer'])
else:
    print("  FAIL: VDEC 无回调")

# ==================== 清理 ====================
print("\n" + "=" * 60)
print("清理")
print("=" * 60)

vdec_run = False
time.sleep(0.5)
flush_desc_queue()
acl.media.vdec_destroy_channel(ch_desc)
acl.media.vdec_destroy_frame_config(frame_cfg)
acl.util.stop_thread(cb_tid)
acl.media.dvpp_free(input_buf)
acl.rt.destroy_stream(global_stream)
acl.rt.reset_device(NPU_DEV_ID)
acl.finalize()
container.close()
print("Done")
