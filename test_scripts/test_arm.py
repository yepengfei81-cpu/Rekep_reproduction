"""
Verify keypoints by moving the robot arm to each one sequentially.
Each move requires user confirmation (Enter) for safety.

Usage:
    python verify_keypoints.py --query_dir vlm_query/2026-04-09_17-02-12_pick_up_the_bowl
"""
import sys
sys.path.insert(0, "/home/ypf/qiuzhiarm_LLM")
import time
import json
import argparse
import numpy as np
from play_sdk import PlayRealRobot

HOME_JOINT = [-0.0612, -1.0550, 0.8829, -1.2343, 1.0351, 0.7597]
ARM_PORT = 50050
HOVER_HEIGHT = 0.12  # 悬停高度 (m)，先到达关键点上方再下降


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_dir', type=str, required=True,
                        help='VLM query output directory containing metadata.json')
    parser.add_argument('--hover', type=float, default=HOVER_HEIGHT,
                        help='hover height above each keypoint (m)')
    args = parser.parse_args()

    # ===== 加载关键点 =====
    metadata_path = f"{args.query_dir}/metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    keypoints = np.array(metadata['init_keypoint_positions'])
    print(f"Loaded {len(keypoints)} keypoints from {metadata_path}")
    for i, kp in enumerate(keypoints):
        print(f"  kp{i}: [{kp[0]:.4f}, {kp[1]:.4f}, {kp[2]:.4f}]")

    # ===== 连接机器人 =====
    print("\nConnecting to robot...")
    robot = PlayRealRobot(port=ARM_PORT, enable_cameras=False)

    # ===== 回 Home，并读取 Home 姿态作为运动朝向 =====
    print("Moving to HOME_JOINT...")
    robot.set_joint_positions(HOME_JOINT, blocking=True)
    time.sleep(1.0)

    pose = robot.get_end_pose()
    if pose is None:
        print("ERROR: Cannot read EE pose")
        robot.shutdown()
        return
    home_pos, home_quat = pose
    print(f"Home pos : {[f'{v:.4f}' for v in home_pos]}")
    print(f"Home quat: {[f'{v:.4f}' for v in home_quat]}")

    # ===== 依次访问每个关键点 =====
    for i, kp in enumerate(keypoints):
        print(f"\n{'='*50}")
        print(f"Keypoint {i}: [{kp[0]:.4f}, {kp[1]:.4f}, {kp[2]:.4f}]")
        print(f"{'='*50}")

        # --- 悬停 ---
        hover_z = float(kp[2] + args.hover)
        hover_pos = [float(kp[0]), float(kp[1]), hover_z]
        input(f"  [Enter] 移动到 kp{i} 上方悬停 [{hover_pos[0]:.4f}, {hover_pos[1]:.4f}, {hover_pos[2]:.4f}]")
        robot.set_end_pose(hover_pos, [float(q) for q in home_quat], blocking=True)
        time.sleep(0.5)

        # --- 下降到关键点 ---
        target_pos = [float(kp[0]), float(kp[1]), float(kp[2])]
        input(f"  [Enter] 下降到 kp{i}      [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
        robot.set_gripper(position=1.0)  # 打开夹爪
        time.sleep(0.3)
        robot.set_end_pose(target_pos, [float(q) for q in home_quat], blocking=True)
        time.sleep(0.5)

        # --- 读取实际位姿 ---
        actual = robot.get_end_pose()
        if actual:
            apos, _ = actual
            err = np.linalg.norm(np.array(apos) - kp)
            print(f"  Actual : [{apos[0]:.4f}, {apos[1]:.4f}, {apos[2]:.4f}]")
            print(f"  Target : [{kp[0]:.4f}, {kp[1]:.4f}, {kp[2]:.4f}]")
            print(f"  Error  : {err*1000:.1f} mm")

        # --- 抬升回悬停 ---
        input(f"  [Enter] 抬升回悬停高度")
        robot.set_end_pose(hover_pos, [float(q) for q in home_quat], blocking=True)
        time.sleep(0.5)

    # ===== 回 Home =====
    input("\n>>> [Enter] 返回 Home <<<")
    robot.set_joint_positions(HOME_JOINT, blocking=True)
    time.sleep(1.0)
    print("Done! All keypoints verified.")
    robot.shutdown()


if __name__ == "__main__":
    main()