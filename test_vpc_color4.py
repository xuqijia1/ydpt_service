#!/usr/bin/env python3
"""VPC convert_color 测试 — 用 VDEC 真实 NV12 数据

JPEG decode 在此环境不工作（synchronize_stream ret=507018），
但 VDEC（FFmpeg 硬解码）正常输出 NV12。

测试流程：
1. 用 VDEC 读一帧 NV12（host bytes）
2. memcpy H2D 到 device
3. VPC convert_color NV12→BGR
4. memcpy D2H 读回 BGR
5. 验证颜色正确性
"""
import acl
import numpy as np
import cv2
import sys
import os

FMT_NV12 = 1
FMT_BGR  = 13
FMT_RGB  = 12
ACL_H2D = 1
ACL_D2H = 2

def align_up(val, alignment):
    return ((val + alignment - 1) // alignment) * alignment

# ==================== 配置 ====================
RTSP_URL = os.environ.get('RTSP_URL', 'rtsp://admin:tykj1357@192.168.1.142:554/Streaming/Channels/101')
SRC_W, SRC_H = 1920, 1080

# ==================== Step 1: VDEC 读一帧 NV12 ====================
print("=" * 60)
print("Step 1: VDEC 读一帧 NV12")
print("=" * 60)

from dvpp_decoder import create_dvpp_decoder

decoder = create_dvpp_decoder(RTSP_URL, device_id=0, en_type="H264")
if decoder is None:
    print("FAIL: VDEC 启动失败")
    sys.exit(1)

frame = decoder.read_frame()
if frame is None or frame.get('format') != 'nv12':
    print("FAIL: 读取 NV12 帧失败")
    sys.exit(1)

nv12_bytes = frame['data']
nv12_np = np.frombuffer(nv12_bytes, dtype=np.uint8)
print(f"  NV12: {len(nv12_bytes)} bytes, nonzero={np.count_nonzero(nv12_np)}")

# CPU 转换做对照
cpu_bgr = cv2.cvtColor(nv12_np.reshape(SRC_H * 3 // 2, SRC_W), cv2.COLOR_YUV2BGR_NV12)
print(f"  CPU BGR: mean={cpu_bgr.mean():.1f}, shape={cpu_bgr.shape}")

# 只读一帧就释放 VDEC
decoder.release()

# ==================== Step 2: 初始化 ACL + VPC ====================
print("\n" + "=" * 60)
print("Step 2: 初始化 ACL + VPC")
print("=" * 60)

acl.init()
acl.rt.set_device(0)
ctx = acl.rt.create_context(0)

stream, ret = acl.rt.create_stream()
print(f"  create_stream: ret={ret}")

vpc_desc = acl.media.dvpp_create_channel_desc()
ret = acl.media.dvpp_create_channel(vpc_desc)
print(f"  create_channel: ret={ret}")
if ret != 0:
    print("FAIL: VPC 通道创建失败")
    sys.exit(1)

# ==================== Step 3: NV12 H2D ====================
print("\n" + "=" * 60)
print("Step 3: NV12 数据上传到 device")
print("=" * 60)

# NV12 stride
nv12_w_stride = align_up(SRC_W, 16)  # 1920
nv12_h_stride = align_up(SRC_H, 2)   # 1080
nv12_size = nv12_w_stride * nv12_h_stride * 3 // 2  # 3110400

# 如果 VDEC 输出的 bytes 不含 stride padding，需要构造 stride 对齐的 buffer
raw_nv12_size = SRC_W * SRC_H * 3 // 2  # 3110400
print(f"  raw_nv12_size={raw_nv12_size}, stride_nv12_size={nv12_size}")

# 构造 stride 对齐的 NV12 buffer
if nv12_size == raw_nv12_size:
    # 无需 padding
    nv12_aligned = nv12_np
else:
    # Y plane: 逐行复制，每行 nv12_w_stride
    nv12_aligned = np.zeros(nv12_size, dtype=np.uint8)
    y_plane = nv12_np[:SRC_H * SRC_W].reshape(SRC_H, SRC_W)
    uv_plane = nv12_np[SRC_H * SRC_W:].reshape(SRC_H // 2, SRC_W)
    for row in range(SRC_H):
        nv12_aligned[row * nv12_w_stride : row * nv12_w_stride + SRC_W] = y_plane[row]
    uv_offset = nv12_h_stride * nv12_w_stride
    for row in range(SRC_H // 2):
        nv12_aligned[uv_offset + row * nv12_w_stride : uv_offset + row * nv12_w_stride + SRC_W] = uv_plane[row]

dev_nv12, ret = acl.media.dvpp_malloc(nv12_size)
print(f"  dvpp_malloc: ret={ret}")

# 用 int(ctypes.data) 做 memcpy（已验证可用）
ret = acl.rt.memcpy(int(dev_nv12), nv12_size, int(nv12_aligned.ctypes.data), nv12_size, ACL_H2D)
print(f"  memcpy H2D: ret={ret}")

# 验证写入
verify_nv12 = np.zeros(nv12_size, dtype=np.uint8)
acl.rt.memcpy(int(verify_nv12.ctypes.data), nv12_size, int(dev_nv12), nv12_size, ACL_D2H)
print(f"  verify nonzero: {np.count_nonzero(verify_nv12)}/{nv12_size}")

# 创建 NV12 pic_desc
nv12_desc = acl.media.dvpp_create_pic_desc()
acl.media.dvpp_set_pic_desc_data(nv12_desc, dev_nv12)
acl.media.dvpp_set_pic_desc_format(nv12_desc, FMT_NV12)
acl.media.dvpp_set_pic_desc_width(nv12_desc, SRC_W)
acl.media.dvpp_set_pic_desc_height(nv12_desc, SRC_H)
acl.media.dvpp_set_pic_desc_width_stride(nv12_desc, nv12_w_stride)
acl.media.dvpp_set_pic_desc_height_stride(nv12_desc, nv12_h_stride)
acl.media.dvpp_set_pic_desc_size(nv12_desc, nv12_size)

# ==================== Step 4: VPC convert_color NV12→BGR ====================
print("\n" + "=" * 60)
print("Step 4: VPC convert_color NV12→BGR")
print("=" * 60)

bgr_w_stride = align_up(SRC_W, 16) * 3  # 5760
bgr_h_stride = align_up(SRC_H, 2)       # 1080
bgr_size = bgr_w_stride * bgr_h_stride   # 6220800

dev_bgr, ret = acl.media.dvpp_malloc(bgr_size)
print(f"  dvpp_malloc BGR: ret={ret}")
acl.rt.memset(int(dev_bgr), bgr_size, 0, bgr_size)

bgr_desc = acl.media.dvpp_create_pic_desc()
acl.media.dvpp_set_pic_desc_data(bgr_desc, dev_bgr)
acl.media.dvpp_set_pic_desc_format(bgr_desc, FMT_BGR)
acl.media.dvpp_set_pic_desc_width(bgr_desc, SRC_W)
acl.media.dvpp_set_pic_desc_height(bgr_desc, SRC_H)
acl.media.dvpp_set_pic_desc_width_stride(bgr_desc, bgr_w_stride)
acl.media.dvpp_set_pic_desc_height_stride(bgr_desc, bgr_h_stride)
acl.media.dvpp_set_pic_desc_size(bgr_desc, bgr_size)

print(f"  INPUT:  NV12 {SRC_W}x{SRC_H} stride=({nv12_w_stride},{nv12_h_stride})")
print(f"  OUTPUT: BGR  {SRC_W}x{SRC_H} stride=({bgr_w_stride},{bgr_h_stride})")

# convert_color_async(channel, INPUT, OUTPUT, stream)
ret = acl.media.dvpp_vpc_convert_color_async(vpc_desc, nv12_desc, bgr_desc, stream)
print(f"  convert_color_async: ret={ret}")

sync_ret = acl.rt.synchronize_stream(stream)
print(f"  synchronize_stream: ret={sync_ret}")

try:
    rc = acl.media.dvpp_get_pic_desc_ret_code(bgr_desc)
    print(f"  ret_code: {rc}")
except Exception as e:
    print(f"  get_ret_code: {e}")

# 读回 BGR
bgr_buf = np.zeros(bgr_size, dtype=np.uint8)
ret = acl.rt.memcpy(int(bgr_buf.ctypes.data), bgr_size, int(dev_bgr), bgr_size, ACL_D2H)
print(f"  memcpy D2H: ret={ret}")

bgr_nonzero = np.count_nonzero(bgr_buf)
print(f"  BGR nonzero: {bgr_nonzero}/{bgr_size}")

if bgr_nonzero > 0:
    # 提取 BGR
    bgr_frame = np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)
    for row in range(SRC_H):
        offset = row * bgr_w_stride
        bgr_frame[row] = bgr_buf[offset:offset + SRC_W * 3].reshape(SRC_W, 3)

    print(f"  NPU BGR: mean={bgr_frame.mean():.1f}, min={bgr_frame.min()}, max={bgr_frame.max()}")
    print(f"  CPU BGR: mean={cpu_bgr.mean():.1f}")

    # 对比 NPU vs CPU
    diff = np.abs(bgr_frame.astype(int) - cpu_bgr.astype(int))
    print(f"  diff: mean={diff.mean():.2f}, max={diff.max()}")

    cv2.imwrite('/tmp/test_vpc_npu_bgr.jpg', bgr_frame)
    cv2.imwrite('/tmp/test_vpc_cpu_bgr.jpg', cpu_bgr)
    print(f"  保存 /tmp/test_vpc_npu_bgr.jpg 和 /tmp/test_vpc_cpu_bgr.jpg")
    print(f"  SUCCESS: VPC convert_color NV12→BGR!")
else:
    print(f"  FAIL: BGR 输出全零")

    # 尝试 stream=0
    print(f"\n  --- Retry with stream=0 ---")
    acl.rt.memset(int(dev_bgr), bgr_size, 0, bgr_size)
    ret = acl.media.dvpp_vpc_convert_color_async(vpc_desc, nv12_desc, bgr_desc, 0)
    print(f"  convert_color_async(stream=0): ret={ret}")
    acl.rt.synchronize_stream(0)

    try:
        rc = acl.media.dvpp_get_pic_desc_ret_code(bgr_desc)
        print(f"  ret_code: {rc}")
    except:
        pass

    bgr_buf2 = np.zeros(bgr_size, dtype=np.uint8)
    acl.rt.memcpy(int(bgr_buf2.ctypes.data), bgr_size, int(dev_bgr), bgr_size, ACL_D2H)
    nz2 = np.count_nonzero(bgr_buf2)
    print(f"  BGR nonzero: {nz2}/{bgr_size}")

# ==================== Step 5: VPC resize NV12→NV12 (对照) ====================
print("\n" + "=" * 60)
print("Step 5: VPC resize NV12→NV12 640x640 (对照)")
print("=" * 60)

RSZ_W, RSZ_H = 640, 640
rsz_w_stride = align_up(RSZ_W, 16)
rsz_h_stride = align_up(RSZ_H, 2)
rsz_size = rsz_w_stride * rsz_h_stride * 3 // 2

dev_rsz, _ = acl.media.dvpp_malloc(rsz_size)
acl.rt.memset(int(dev_rsz), rsz_size, 0, rsz_size)

rsz_desc = acl.media.dvpp_create_pic_desc()
acl.media.dvpp_set_pic_desc_data(rsz_desc, dev_rsz)
acl.media.dvpp_set_pic_desc_format(rsz_desc, FMT_NV12)
acl.media.dvpp_set_pic_desc_width(rsz_desc, RSZ_W)
acl.media.dvpp_set_pic_desc_height(rsz_desc, RSZ_H)
acl.media.dvpp_set_pic_desc_width_stride(rsz_desc, rsz_w_stride)
acl.media.dvpp_set_pic_desc_height_stride(rsz_desc, rsz_h_stride)
acl.media.dvpp_set_pic_desc_size(rsz_desc, rsz_size)

resize_cfg = acl.media.dvpp_create_resize_config()
ret = acl.media.dvpp_vpc_resize_async(vpc_desc, nv12_desc, rsz_desc, resize_cfg, stream)
print(f"  resize_async: ret={ret}")
sync_ret = acl.rt.synchronize_stream(stream)
print(f"  synchronize_stream: ret={sync_ret}")

rsz_buf = np.zeros(rsz_size, dtype=np.uint8)
acl.rt.memcpy(int(rsz_buf.ctypes.data), rsz_size, int(dev_rsz), rsz_size, ACL_D2H)
rsz_nonzero = np.count_nonzero(rsz_buf)
print(f"  resize nonzero: {rsz_nonzero}/{rsz_size}")

# ==================== Step 6: VPC crop_resize NV12→BGR 640x640 ====================
print("\n" + "=" * 60)
print("Step 6: VPC crop_resize NV12→BGR 640x640")
print("=" * 60)

RSZ2_W, RSZ2_H = 640, 640
rsz2_w_stride = align_up(RSZ2_W, 16) * 3  # BGR
rsz2_h_stride = align_up(RSZ2_H, 2)
rsz2_size = rsz2_w_stride * rsz2_h_stride

dev_rsz2, _ = acl.media.dvpp_malloc(rsz2_size)
acl.rt.memset(int(dev_rsz2), rsz2_size, 0, rsz2_size)

rsz2_desc = acl.media.dvpp_create_pic_desc()
acl.media.dvpp_set_pic_desc_data(rsz2_desc, dev_rsz2)
acl.media.dvpp_set_pic_desc_format(rsz2_desc, FMT_BGR)
acl.media.dvpp_set_pic_desc_width(rsz2_desc, RSZ2_W)
acl.media.dvpp_set_pic_desc_height(rsz2_desc, RSZ2_H)
acl.media.dvpp_set_pic_desc_width_stride(rsz2_desc, rsz2_w_stride)
acl.media.dvpp_set_pic_desc_height_stride(rsz2_desc, rsz2_h_stride)
acl.media.dvpp_set_pic_desc_size(rsz2_desc, rsz2_size)

roi = acl.media.dvpp_create_roi_config(0, SRC_W - 1, 0, SRC_H - 1)
rcfg = acl.media.dvpp_create_resize_config()
acl.media.dvpp_set_resize_config_interpolation(rcfg, 0)

ret = acl.media.dvpp_vpc_crop_resize_async(vpc_desc, nv12_desc, rsz2_desc, roi, rcfg, stream)
print(f"  crop_resize_async: ret={ret}")
sync_ret = acl.rt.synchronize_stream(stream)
print(f"  synchronize_stream: ret={sync_ret}")

try:
    cr_rc = acl.media.dvpp_get_pic_desc_ret_code(rsz2_desc)
    print(f"  ret_code: {cr_rc}")
except:
    pass

rsz2_buf = np.zeros(rsz2_size, dtype=np.uint8)
acl.rt.memcpy(int(rsz2_buf.ctypes.data), rsz2_size, int(dev_rsz2), rsz2_size, ACL_D2H)
rsz2_nonzero = np.count_nonzero(rsz2_buf)
print(f"  BGR 640x640 nonzero: {rsz2_nonzero}/{rsz2_size}")

if rsz2_nonzero > 0:
    rsz2_frame = np.zeros((RSZ2_H, RSZ2_W, 3), dtype=np.uint8)
    for row in range(RSZ2_H):
        offset = row * rsz2_w_stride
        rsz2_frame[row] = rsz2_buf[offset:offset + RSZ2_W * 3].reshape(RSZ2_W, 3)
    print(f"  BGR 640x640: mean={rsz2_frame.mean():.1f}")
    cv2.imwrite('/tmp/test_vpc_crop_resize_bgr.jpg', rsz2_frame)
    print(f"  SUCCESS: crop_resize NV12→BGR 640x640!")

# ==================== 清理 ====================
print("\n" + "=" * 60)
print("清理")
print("=" * 60)

acl.media.dvpp_free(dev_nv12)
acl.media.dvpp_free(dev_bgr)
acl.media.dvpp_free(dev_rsz)
acl.media.dvpp_free(dev_rsz2)
acl.media.dvpp_destroy_pic_desc(nv12_desc)
acl.media.dvpp_destroy_pic_desc(bgr_desc)
acl.media.dvpp_destroy_pic_desc(rsz_desc)
acl.media.dvpp_destroy_pic_desc(rsz2_desc)
acl.media.dvpp_destroy_resize_config(resize_cfg)
acl.media.dvpp_destroy_resize_config(rcfg)
acl.media.dvpp_destroy_roi_config(roi)
acl.media.dvpp_destroy_channel(vpc_desc)
acl.rt.destroy_stream(stream)
acl.rt.reset_device(0)
acl.finalize()
print("\nDone!")
