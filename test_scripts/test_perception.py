"""
测试脚本：SAM3 + DINOv3 关键点提议 → CoTracker3 在线跟踪
流程:
  1. 从 D405 拍第一帧 → SAM3 分割 → DINOv3 提议关键点（复用 test_dinov3）
  2. 启动 D405 持续拍摄
  3. CoTracker3 Online 逐窗口跟踪关键点
  4. 实时可视化 + 保存视频
"""

import sys
sys.path.insert(0, "/home/ypf/sam3-main")
sys.path.insert(0, "/home/ypf/co-tracker")

import numpy as np
import torch
import cv2
import pyrealsense2 as rs
import time
import os

from cotracker.predictor import CoTrackerOnlinePredictor

# 复用 test_dinov3 中的函数
from test_dinov3 import (
    capture_rgbd,
    depth_to_pointcloud,
    sam3_segment,
    DINOv3KeypointProposer,
    draw_keypoints,
    CAMERA_SERIAL,
    CAMERA_RESOLUTION,
    MODEL_PATH,
    DEVICE,
)

# ============================================
# 配置
# ============================================
COTRACKER_CHECKPOINT = "/home/ypf/co-tracker/checkpoints/scaled_online.pth"
SAVE_DIR = "/home/ypf/ReKep/test_output/cotracker"
TRACK_DURATION = 10  # 跟踪持续秒数
FPS = 30

os.makedirs(SAVE_DIR, exist_ok=True)


def create_d405_stream(serial, resolution=(640, 480), fps=30):
    """创建 D405 持续流"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, fps)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    # 等待曝光稳定
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, align, profile


def get_frame(pipeline, align):
    """从流中获取最新一帧 RGB (H, W, 3) uint8，丢弃缓冲"""
    # 连续读取，清空缓冲区中的旧帧
    frame_set = None
    for _ in range(5):
        try:
            frame_set = pipeline.wait_for_frames(timeout_ms=0)
        except RuntimeError:
            break
    # 如果上面没读到，阻塞等一帧
    if frame_set is None:
        frame_set = pipeline.wait_for_frames()

    aligned = align.process(frame_set)
    color_frame = aligned.get_color_frame()
    if not color_frame:
        return None
    bgr = np.asanyarray(color_frame.get_data())
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def draw_tracks_on_frame(frame, tracks_xy, visibility, colors):
    """在帧上画跟踪点和编号"""
    img = frame.copy()
    for i in range(tracks_xy.shape[0]):
        x, y = int(tracks_xy[i, 0]), int(tracks_xy[i, 1])
        vis = visibility[i]
        color = colors[i % len(colors)]
        if vis:
            cv2.circle(img, (x, y), 6, color, -1)
            cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
            cv2.putText(img, str(i), (x + 8, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, str(i), (x + 8, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.circle(img, (x, y), 6, color, 1)  # 空心 = 不可见
    return img


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="bowl,pen",
                        help="SAM3 物体 prompt（逗号分隔）")
    parser.add_argument("--duration", type=float, default=TRACK_DURATION,
                        help="跟踪持续秒数")
    parser.add_argument("--confidence", type=float, default=0.3)
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",")]

    # ============================
    # 步骤 1：拍初始帧 + 提议关键点
    # ============================
    print("=" * 60)
    print("步骤 1：拍摄初始帧 + SAM3 分割 + DINOv3 关键点提议")
    print("=" * 60)

    rgb_init, depth_init, intrinsics = capture_rgbd(CAMERA_SERIAL, resolution=CAMERA_RESOLUTION)
    points = depth_to_pointcloud(depth_init, intrinsics)
    seg_mask, prompt_labels = sam3_segment(rgb_init, prompts, confidence=args.confidence)

    proposer = DINOv3KeypointProposer(MODEL_PATH, device=DEVICE)
    keypoints_3d, keypoints_2d, projected_img = proposer.propose_keypoints(rgb_init, points, seg_mask)

    if len(keypoints_3d) == 0:
        print("未检测到关键点，退出")
        return

    print(f"提议了 {len(keypoints_3d)} 个关键点")
    for i, (kp3d, kp2d) in enumerate(zip(keypoints_3d, keypoints_2d)):
        print(f"  关键点 {i}: pixel=({kp2d[1]}, {kp2d[0]}), "
              f"3D=({kp3d[0]:.4f}, {kp3d[1]:.4f}, {kp3d[2]:.4f})")

    cv2.imwrite(os.path.join(SAVE_DIR, "init_keypoints.png"),
                cv2.cvtColor(projected_img, cv2.COLOR_RGB2BGR))

    # 释放 DINOv3 和 SAM3 显存
    del proposer
    torch.cuda.empty_cache()

    # ============================
    # 步骤 2：初始化 CoTracker3 Online
    # ============================
    print("\n" + "=" * 60)
    print("步骤 2：初始化 CoTracker3 Online")
    print("=" * 60)

    model = CoTrackerOnlinePredictor(checkpoint=COTRACKER_CHECKPOINT)
    model = model.to(DEVICE)

    # 构建 queries: (B, N, 3) → (frame_idx, x, y)
    # keypoints_2d 是 (row, col) 格式，CoTracker 要 (x, y) 即 (col, row)
    N = len(keypoints_2d)
    queries = torch.zeros(1, N, 3, device=DEVICE)
    queries[0, :, 0] = 0  # 都在第 0 帧
    queries[0, :, 1] = torch.tensor([kp[1] for kp in keypoints_2d], dtype=torch.float32)  # x = col
    queries[0, :, 2] = torch.tensor([kp[0] for kp in keypoints_2d], dtype=torch.float32)  # y = row

    print(f"CoTracker3 step size: {model.step} 帧")
    print(f"查询点: {N} 个")

    # ============================
    # 步骤 3：启动 D405 流 + 在线跟踪
    # ============================
    print("\n" + "=" * 60)
    print(f"步骤 3：开始在线跟踪 ({args.duration}s)")
    print("=" * 60)

    pipeline, align, profile = create_d405_stream(
        CAMERA_SERIAL, resolution=CAMERA_RESOLUTION, fps=FPS
    )

    # 颜色列表
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 255, 0), (255, 128, 0), (0, 128, 255),
        (255, 128, 128), (128, 255, 128), (128, 128, 255),
    ]

    frame_buffer = []
    video_frames = []  # 用于保存视频
    t_start = time.time()
    frame_count = 0
    is_first_step = True
    last_tracks = None  # 最近一次跟踪结果
    last_vis = None

    print("按 'q' 提前退出")

    try:
        while time.time() - t_start < args.duration:
            rgb = get_frame(pipeline, align)
            if rgb is None:
                continue

            frame_buffer.append(rgb)
            frame_count += 1

            # CoTracker 需要每 model.step 帧处理一次
            if is_first_step and len(frame_buffer) >= model.step * 2:
                chunk = np.stack(frame_buffer[-model.step * 2:])
                video_chunk = torch.tensor(chunk, device=DEVICE).float().permute(0, 3, 1, 2)[None]
                model(video_chunk, is_first_step=True, queries=queries)
                is_first_step = False
                print(f"  CoTracker 初始化完成 (frame {frame_count})")

            elif not is_first_step and frame_count % model.step == 0:
                chunk = np.stack(frame_buffer[-model.step * 2:])
                video_chunk = torch.tensor(chunk, device=DEVICE).float().permute(0, 3, 1, 2)[None]
                pred_tracks, pred_visibility = model(video_chunk)

                if pred_tracks is not None:
                    last_tracks = pred_tracks[0, -1].cpu().numpy()
                    last_vis = pred_visibility[0, -1].cpu().numpy()

                    visible_count = int(last_vis.sum())
                    print(f"  frame {frame_count}: {visible_count}/{N} 点可见", end="")
                    for i in range(N):
                        if last_vis[i]:
                            print(f"  [{i}]=({last_tracks[i,0]:.0f},{last_tracks[i,1]:.0f})", end="")
                    print()

                    # 只在 CoTracker 更新时显示和保存
                    vis_frame = draw_tracks_on_frame(rgb, last_tracks, last_vis, colors)
                    video_frames.append(vis_frame)

                    display = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                    elapsed = time.time() - t_start
                    cv2.putText(display, f"t={elapsed:.1f}s  frame={frame_count}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("CoTracker3 Tracking", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # ============================
    # 步骤 4：保存结果
    # ============================
    print(f"\n跟踪结束: 共 {frame_count} 帧, {len(video_frames)} 个带跟踪的帧")

    if len(video_frames) > 0:
        import imageio
        video_path = os.path.join(SAVE_DIR, "tracking_result.mp4")
        writer = imageio.get_writer(video_path, fps=FPS) 
        for frame in video_frames:
            writer.append_data(frame)
        writer.close()
        print(f"视频已保存: {video_path}")

        # 保存最后一帧
        cv2.imwrite(os.path.join(SAVE_DIR, "last_tracked_frame.png"),
                    cv2.cvtColor(video_frames[-1], cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()