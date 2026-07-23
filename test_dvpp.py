#!/usr/bin/env python3
"""Ascend-FFmpeg 硬解码测试 - 逐步排查"""
import sys
import os
import time
import subprocess
import numpy as np

RTSP_URL = "rtsp://admin:tykj1357@192.168.1.142/ch1/mian/av_stream"
OUTPUT_DIR = "./test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 查找 ffmpeg
FFMPEG = "/usr/local/ascend_ffmpeg/bin/ffmpeg"
FFPROBE = "/usr/local/ascend_ffmpeg/bin/ffprobe"
if not os.path.isfile(FFMPEG):
    FFMPEG = "ffmpeg"
if not os.path.isfile(FFPROBE):
    FFPROBE = "ffprobe"

# 设置环境
env = os.environ.copy()
lib_dir = "/usr/local/ascend_ffmpeg/lib"
if os.path.isdir(lib_dir):
    env['LD_LIBRARY_PATH'] = lib_dir + ':' + env.get('LD_LIBRARY_PATH', '')

print("=" * 60)
print("Ascend-FFmpeg 硬解码管道测试")
print("=" * 60)

# Step 1: 检查 ffmpeg
print(f"\n[1] ffmpeg 路径: {FFMPEG}")
result = subprocess.run([FFMPEG, '-version'], capture_output=True, text=True, timeout=5, env=env)
print(f"  版本: {result.stdout.split(chr(10))[0]}")

# Step 2: 测试 ffmpeg 命令行（不输出到管道，先看能否正常解码）
print(f"\n[2] 测试 ffmpeg 命令行硬解码（输出到文件）...")
test_cmd = [
    FFMPEG,
    '-rtsp_transport', 'tcp', '-stimeout', '5000000',
    '-hwaccel', 'ascend', '-c:v', 'h264_ascend',
    '-device_id', '0', '-channel_id', '0',
    '-i', RTSP_URL,
    '-f', 'rawvideo', '-pix_fmt', 'nv12',
    '-t', '1',  # 只解码1秒
    '-loglevel', 'verbose',  # 详细日志看问题
    '-y', os.path.join(OUTPUT_DIR, 'test_raw.nv12')
]
print(f"  命令: {' '.join(test_cmd)}")
proc = subprocess.Popen(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
_, stderr = proc.communicate(timeout=30)
stderr_text = stderr.decode('utf-8', errors='replace')
print(f"  返回码: {proc.returncode}")
if stderr_text:
    # 只打印最后20行
    lines = stderr_text.strip().split('\n')
    for line in lines[-20:]:
        print(f"  | {line}")

if proc.returncode == 0:
    fsize = os.path.getsize(os.path.join(OUTPUT_DIR, 'test_raw.nv12'))
    print(f"  ✓ 成功! 输出文件大小: {fsize} bytes")
else:
    print(f"  ✗ 失败!")

# Step 3: 测试管道方式 - bgr24
print(f"\n[3] 测试 bgr24 管道硬解码...")
frame_size = 1920 * 1080 * 3
cmd = [
    FFMPEG,
    '-rtsp_transport', 'tcp', '-stimeout', '5000000',
    '-hwaccel', 'ascend', '-c:v', 'h264_ascend',
    '-device_id', '0', '-channel_id', '0',
    '-i', RTSP_URL,
    '-f', 'rawvideo', '-pix_fmt', 'bgr24',
    '-loglevel', 'verbose',
    'pipe:1'
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
# 等一下看进程是否存活
time.sleep(2)
retcode = proc.poll()
if retcode is not None:
    # 进程已退出，读取 stderr
    stderr_text = proc.stderr.read().decode('utf-8', errors='replace')
    print(f"  ✗ ffmpeg 退出 (ret={retcode})")
    lines = stderr_text.strip().split('\n')
    for line in lines[-20:]:
        print(f"  | {line}")
else:
    print(f"  进程存活，尝试读取帧...")
    try:
        raw = proc.stdout.read(frame_size)
        if len(raw) == frame_size:
            import cv2
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(1080, 1920, 3).copy()
            path = os.path.join(OUTPUT_DIR, 'bgr24_pipe_frame_1.jpg')
            cv2.imwrite(path, frame)
            print(f"  ✓ bgr24 管道成功! mean={frame.mean():.1f}, saved={path}")
        else:
            print(f"  ✗ 读取不完整: got {len(raw)}/{frame_size} bytes")
            # 读 stderr 看原因
            time.sleep(1)
            stderr_text = ""
            try:
                proc.stderr.close()
                proc.kill()
                proc.wait(timeout=3)
            except:
                pass
    except Exception as e:
        print(f"  ✗ 读取异常: {e}")

    # 清理进程
    try:
        proc.stdout.close()
        proc.stderr.close()
        proc.kill()
        proc.wait(timeout=3)
    except:
        pass

# Step 4: 测试管道方式 - nv12
print(f"\n[4] 测试 nv12 管道硬解码...")
nv12_size = 1920 * 1080 * 3 // 2
cmd = [
    FFMPEG,
    '-rtsp_transport', 'tcp', '-stimeout', '5000000',
    '-hwaccel', 'ascend', '-c:v', 'h264_ascend',
    '-device_id', '0', '-channel_id', '0',
    '-i', RTSP_URL,
    '-f', 'rawvideo', '-pix_fmt', 'nv12',
    '-loglevel', 'verbose',
    'pipe:1'
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
time.sleep(2)
retcode = proc.poll()
if retcode is not None:
    stderr_text = proc.stderr.read().decode('utf-8', errors='replace')
    print(f"  ✗ ffmpeg 退出 (ret={retcode})")
    lines = stderr_text.strip().split('\n')
    for line in lines[-20:]:
        print(f"  | {line}")
else:
    print(f"  进程存活，尝试读取帧...")
    try:
        raw = proc.stdout.read(nv12_size)
        if len(raw) == nv12_size:
            import cv2
            nv12 = np.frombuffer(raw, dtype=np.uint8).reshape(1080 * 3 // 2, 1920)
            y_plane = nv12[:1080, :]
            uv_plane = nv12[1080:1080 + 540, :]
            nv12_clean = np.concatenate([y_plane, uv_plane], axis=0)
            bgr = cv2.cvtColor(nv12_clean, cv2.COLOR_YUV2BGR_NV12)
            path = os.path.join(OUTPUT_DIR, 'nv12_pipe_frame_1.jpg')
            cv2.imwrite(path, bgr)
            print(f"  ✓ nv12 管道成功! mean={bgr.mean():.1f}, saved={path}")
        else:
            print(f"  ✗ 读取不完整: got {len(raw)}/{nv12_size} bytes")
    except Exception as e:
        print(f"  ✗ 读取异常: {e}")

    try:
        proc.stdout.close()
        proc.stderr.close()
        proc.kill()
        proc.wait(timeout=3)
    except:
        pass

# Step 5: CPU 软解对比
print(f"\n[5] CPU 软解码对比...")
cmd = [
    FFMPEG,
    '-rtsp_transport', 'tcp', '-stimeout', '5000000',
    '-i', RTSP_URL,
    '-f', 'rawvideo', '-pix_fmt', 'bgr24',
    '-loglevel', 'error', 'pipe:1'
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
decoded = 0
start = time.time()
import cv2
while decoded < 30:
    raw = proc.stdout.read(frame_size)
    if len(raw) != frame_size:
        break
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(1080, 1920, 3)
    decoded += 1
    if decoded == 1:
        path = os.path.join(OUTPUT_DIR, 'cpu_soft_frame_1.jpg')
        cv2.imwrite(path, frame)
        print(f"  软解帧#1: mean={frame.mean():.1f}, saved={path}")
elapsed = time.time() - start
proc.stdout.close()
proc.stderr.close()
proc.kill()
proc.wait(timeout=3)
print(f"  CPU 软解: {decoded} 帧, {decoded/elapsed:.1f} fps")

print(f"\n{'=' * 60}")
print(f"测试完成，输出: {OUTPUT_DIR}/")
