#!/usr/bin/env python3
# hi_mpi V2 昇腾310P3 VDEC+VPC硬件解码测试
# RTSP: rtsp://admin:tykj1357@192.168.13.115:554/Streaming/Channels/101
import av
import os
import time
import traceback

# 环境变量
RTSP_URL = "rtsp://admin:tykj1357@192.168.13.115:554/Streaming/Channels/101"
NPU_DEV = int(os.environ.get("NPU_DEV_ID", 0))
DST_W, DST_H = 640, 640

# 加载昇腾hi_mpi、acl
import acl
from ctypes import *
# 加载媒体V2 so，容器自动source set_env.sh后可直接使用
mpi_lib = CDLL("/usr/local/Ascend/runtime/lib64/libacl_dvpp_mpi.so")

# 全局常量
H265_MAIN = 0
PIX_FMT_NV12 = 1
PIX_FMT_BGR888 = 13
VB_BLOCK_NUM = 16

# 全局句柄
epoll_fd = -1
vdec_chn = -1
vb_pool = -1
running = [True]
frame_cache = []

def align(val, align_num):
    return ((val + align_num - 1) // align_num) * align_num

# 1. 系统全局初始化
def mpi_sys_init(dev_id):
    ret = mpi_lib.hi_mpi_sys_full_init()
    assert ret == 0, f"sys init fail ret={ret}"
    ret = mpi_lib.hi_mpi_sys_bind(dev_id)
    assert ret == 0, f"bind dev{dev_id} fail"
    print("hi_mpi系统初始化完成")

# 2. 创建VB内存池
def create_vb_pool(w, h):
    w_align = align(w, 16)
    h_align = align(h, 2)
    buf_size = w_align * h_align * 3 // 2
    class VbPoolCfg(Structure):
        _fields_ = [("u32BlkSize", c_uint), ("u32BlkCnt", c_uint), ("u32Mode", c_uint)]
    cfg = VbPoolCfg()
    cfg.u32BlkSize = buf_size
    cfg.u32BlkCnt = VB_BLOCK_NUM
    cfg.u32Mode = 0
    pool_handle = c_int()
    ret = mpi_lib.hi_mpi_vb_create_pool(byref(cfg), byref(pool_handle))
    assert ret == 0
    global vb_pool
    vb_pool = pool_handle.value
    print(f"VB池创建成功，单块大小:{buf_size}")
    return buf_size

# 3. epoll事件线程
def epoll_loop(args):
    global epoll_fd, running
    while running[0]:
        events = (c_void_p * 10)()
        num = mpi_lib.hi_mpi_sys_wait_epoll(epoll_fd, events, 10, 100)
        if num <= 0:
            continue
        for i in range(num):
            chn_type = c_int()
            chn_id = c_int()
            ret = mpi_lib.hi_mpi_sys_get_epoll_info(events[i], byref(chn_type), byref(chn_id))
            if ret != 0:
                continue
            # VDEC解码帧回调读取
            if chn_type == 1:
                frame_ptr = c_void_p()
                frame_info = c_void_p()
                ret = mpi_lib.hi_vdec_get_frame(chn_id, byref(frame_ptr), byref(frame_info), 50)
                if ret == 0:
                    frame_cache.append((frame_ptr, frame_info))
                    print(f"VDEC获取到解码帧，累计{len(frame_cache)}帧")
    print("epoll线程退出")

# 4. 创建VDEC通道
def create_vdec_chn(src_w, src_h, codec_type):
    global epoll_fd, vdec_chn
    epoll_fd = mpi_lib.hi_mpi_sys_create_epoll()
    class VdecChnAttr(Structure):
        _fields_ = [
            ("enType", c_uint),
            ("u32PicWidth", c_uint),
            ("u32PicHeight", c_uint),
            ("enOutputFormat", c_uint),
            ("u32BufNum", c_uint),
            ("u32QueueDepth", c_uint)
        ]
    vdec_attr = VdecChnAttr()
    vdec_attr.enType = codec_type
    vdec_attr.u32PicWidth = src_w
    vdec_attr.u32PicHeight = src_h
    vdec_attr.enOutputFormat = PIX_FMT_NV12
    vdec_attr.u32BufNum = VB_BLOCK_NUM
    vdec_attr.u32QueueDepth = 4
    chn_handle = c_int()
    ret = mpi_lib.hi_vdec_create_chn(0, byref(vdec_attr), byref(chn_handle))
    assert ret == 0
    vdec_chn = chn_handle.value
    # 注册epoll事件
    mpi_lib.hi_mpi_sys_bind_epoll(epoll_fd, 1, vdec_chn)
    print(f"VDEC通道{vdec_chn}创建完成")

# 5. 发送H265 NAL码流至VDEC
def send_vdec_stream(nal_bytes, is_sps):
    data_buf = c_char_p(nal_bytes)
    data_len = len(nal_bytes)
    ret = mpi_lib.hi_vdec_send_stream(vdec_chn, data_buf, data_len, 0 if is_sps else 1)
    return ret

# 6. VPC硬件缩放+色域转换
def vpc_process(frame_ptr, src_w, src_h, dst_w, dst_h):
    w_align = align(src_w, 16)
    h_align = align(src_h, 2)
    dst_w_align = align(dst_w, 16)
    dst_h_align = align(dst_h, 2)
    # 获取VB输出块
    blk = c_void_p()
    ret = mpi_lib.hi_mpi_vb_get_block(vb_pool, 0, byref(blk))
    if ret != 0:
        return None
    # VPC通道创建
    vpc_chn = c_int()
    ret = mpi_vpc_create_chn(0)
    # 硬件缩放转换流水线
    ret = mpi_lib.hi_mpi_vpc_crop_resize_convert_color(vpc_chn, frame_ptr, blk,
                                                       0,0,src_w,src_h,0,0,dst_w,dst_h,
                                                       PIX_FMT_BGR888, 0)
    if ret == 0:
        # 拷贝BGR图像回Host用于测试保存
        bgr_size = dst_w_align * dst_h_align * 3
        host_buf = (c_ubyte * bgr_size)()
        mpi_lib.hi_mpi_vb_handle_to_addr(blk, byref(c_void_p(host_buf)))
        import numpy as np
        import cv2
        arr = np.frombuffer(host_buf, dtype=np.uint8).reshape(dst_h_align, dst_w_align*3)
        img = arr[:dst_h, :dst_w*3].reshape(dst_h, dst_w, 3)
        cv2.imwrite(f"/tmp/dev{NPU_DEV}_v2.jpg", img)
        print("VPC处理完成，图片已保存")
    mpi_lib.hi_mpi_vb_release_block(blk)
    return ret

# 主流程
if __name__ == "__main__":
    # Step1 RTSP拉流，仅提取NAL小包（纯CPU网络逻辑，无图像运算）
    print(f"拉流地址: {RTSP_URL}")
    container = av.open(RTSP, options={
        "rtsp_transport": "tcp",
        "stimeout": "5000000",
        "buffer_size": "1048576"
    })
    video_stream = container.streams.video[0]
    src_w = video.codec_context.width
    src_h = video.codec_context.height
    codec = video.codec_context.name
    is_h265 = codec in ("hevc", "h265")
    print(f"原始分辨率 {src_w}×{src_h} 编码:{codec}")

    sps_list = []
    nal_list = []
    for pkt in container.demux(video_stream):
        data = bytes(pkt)
        if len(data) == 0:
            continue
        # 分离H265 SPS/PPS
        if is_h265:
            if (data[:4] == b"\x00\x00\x01" and (data[3] >> 1) & 0x3F in (32,33,34)) or \
               (data[:3] == b"\x00\x00\x00\x01" and (data[4] >> 1) & 0x3F in (32,33,34)):
                sps_list.append(data)
                continue
        nal_list.append(data)
        if len(nal_list) >= 60:
            break
    print(f"捕获序列头:{len(sps_list)} 图像包:{len(nal_list)}")

    # Step2 hi_mpi V2 初始化
    acl.init()
    acl.rt.set_device(NPU_DEV)
    mpi_sys_init(NPU_DEV)
    create_vb_pool(src_w, src_h)
    create_vdec_chn(src_w, src_h, H265_MAIN if is_h265 else 2)

    # 启动epoll异步线程
    import threading
    epoll_t = threading.Thread(target=epoll_loop, args=([],))
    epoll_t.start()

    # Step3 先发SPS序列头
    for sps in sps_list:
        ret = send_vdec_stream(sps, True)
        print(f"发送SPS ret={ret}")
        time.sleep(0.05)
    # 发送图像NAL包
    for idx, nal in enumerate(nal_list):
        ret = send_vdec_stream(nal, False)
        if ret == 0 and idx % 10 == 0:
            print(f"发送图像包{idx} ok")
        time.sleep(0.03)

    # 等待解码帧
    time.sleep(5)
    # 取第一帧做VPC硬件处理
    if len(frame_cache) > 0:
        frame_data, frame_info = frame_cache[0]
        vpc_process(frame_data, src_w, src_h, DST_W, DST_H)
    else:
        print("VDEC未输出解码帧，检查容器privileged/大页配置")

    # 资源释放
    running[0] = False
    epoll_t.join()
    # 释放解码帧VB块
    for f_ptr, f_info in frame_cache:
        mpi_lib.hi_vdec_release_frame(vdec_chn, f_ptr)
    mpi_lib.hi_vdec_destroy_chn(vdec_chn)
    mpi_lib.hi_mpi_vb_destroy_pool(vb_pool)
    mpi_lib.hi_mpi_sys_exit()
    acl.rt.reset_device(NPU_DEV)
    acl.finalize()
    container.close()
    print("V2媒体脚本执行完成")