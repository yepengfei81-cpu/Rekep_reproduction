"""
Real-world environment for ReKep (Phase 1: real arm execution with closed-loop control).

Matches the interface of environment.ReKepOGEnv so that real_main.py can swap
seamlessly between simulation and real.
"""
import sys
sys.path.insert(0, "/home/ypf/sam3-main")
sys.path.insert(0, "/home/ypf/co-tracker")
sys.path.insert(0, "/home/ypf/qiuzhiarm_LLM")
import threading
import time
import torch
import numpy as np
import json
import os
import cv2
import pyrealsense2 as rs
import transform_utils as T
from utils import (
    bcolors,
    get_clock_time,
    angle_between_quats,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
)
from play_sdk import PlayRealRobot


class RealEnvironment:
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.interpolate_pos_step_size = self.config['interpolate_pos_step_size']
        self.interpolate_rot_step_size = self.config['interpolate_rot_step_size']
        self.step_counter = 0
        self.video_cache = []
        self._grasping = False
        self._abort_event = threading.Event()
        # CoTracker3
        self._tracker = None
        self._tracker_initialized = False
        self._frame_buffer = []
        self._last_tracks_2d = None
        # visualization & tracking thread
        self._vis_enabled = False
        self._kp_lock = threading.Lock()
        self._vis_thread = None
        # gripper / timing config
        self._gripper_open_width = self.config.get('gripper_open_width', 1.0)
        self._gripper_close_width = self.config.get('gripper_close_width', 0.0)
        self._gripper_grasp_threshold = self.config.get('gripper_grasp_threshold', 0.02)
        self._gripper_grasp_min = self.config.get('gripper_grasp_min', 0.01)
        self._gripper_wait_sec = self.config.get('gripper_wait_sec', 0.5)
        self._move_wait_sec = self.config.get('move_wait_sec', 0.3)

        # ============================
        # = load calibration
        # ============================
        calib_dir = self.config.get('calib_dir', '/home/ypf/qiuzhiarm_LLM/calibration/play')
        head_extrinsics_file = os.path.join(calib_dir, 'head_camera_extrinsics.json')
        with open(head_extrinsics_file) as f:
            head_data = json.load(f)
        R_mat = np.array(head_data['rotation_matrix'])
        t_vec = np.array(head_data['translation'])
        self.T_head2base = np.eye(4)
        self.T_head2base[:3, :3] = R_mat
        self.T_head2base[:3, 3] = t_vec
        self.head_intrinsics = head_data.get('head_camera_intrinsics', {})

        # ============================
        # = camera setup
        # ============================
        self.camera_serial = self.config.get('camera_serial', '230422271972')
        self.camera_resolution = (
            self.head_intrinsics.get('width', 640),
            self.head_intrinsics.get('height', 480),
        )
        self.depth_raw_to_mm = self.config.get('d405_depth_raw_to_mm', 0.1)
        self._pipeline = None
        self._align = None
        self._init_camera()

        # ============================
        # = robot setup
        # ============================
        arm_port = self.config.get('arm_port', 50050)
        self.robot = PlayRealRobot(port=arm_port, enable_cameras=False)
        print(f'{bcolors.OKGREEN}[RealEnv] Robot connected (port={arm_port}){bcolors.ENDC}')

        home_joint = self.config.get('home_joint', [-0.0612, -1.0550, 0.8829, -1.2343, 1.0351, 0.7597])
        self.reset_joint_pos = np.array(home_joint)
        self.world2robot_homo = np.eye(4)  # base frame = world frame

        # Move to home
        self.robot.set_joint_positions(list(home_joint), blocking=True)
        time.sleep(0.5)
        self.robot.set_gripper(position=self._gripper_open_width)
        time.sleep(self._gripper_wait_sec)

        self._home_ee_pose = self._read_ee_pose()
        print(f'{bcolors.OKGREEN}[RealEnv] Home EE: pos={self._home_ee_pose[:3].round(4)}, '
              f'quat={self._home_ee_pose[3:].round(4)}{bcolors.ENDC}')

        # keypoints
        self.keypoints = None
        self._keypoint2object = {}
        self._grasped_kp_indices = set()
        self._last_visibility = None

    # ======================================
    # = robot state helpers
    # ======================================
    def _read_ee_pose(self):
        """Read current EE pose from robot as np.array [x,y,z,qx,qy,qz,qw]."""
        pose = self.robot.get_end_pose()
        if pose is None:
            print(f'{bcolors.WARNING}[RealEnv] WARNING: get_end_pose returned None{bcolors.ENDC}')
            return self._home_ee_pose.copy()
        position, orientation = pose
        return np.array(list(position) + list(orientation))

    # ======================================
    # = camera
    # ======================================
    def _init_camera(self):
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.camera_serial)
        w, h = self.camera_resolution
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
        self._align = rs.align(rs.stream.color)
        profile = self._pipeline.start(cfg)
        # warmup
        for _ in range(30):
            self._pipeline.wait_for_frames()
        # get intrinsics from profile if not loaded from file
        if not self.head_intrinsics:
            color_stream = profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            self.head_intrinsics = {
                'fx': intr.fx, 'fy': intr.fy,
                'cx': intr.ppx, 'cy': intr.ppy,
                'width': intr.width, 'height': intr.height,
            }
        print(f'{bcolors.OKGREEN}[RealEnv] Camera initialized (serial={self.camera_serial}){bcolors.ENDC}')

    def _capture(self):
        """Capture aligned RGB + depth from D405."""
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        bgr = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
        return rgb, depth_raw

    def init_tracker(self, keypoints_2d):
        from cotracker.predictor import CoTrackerOnlinePredictor
        CHECKPOINT = "/home/ypf/co-tracker/checkpoints/scaled_online.pth"
        self._tracker = CoTrackerOnlinePredictor(checkpoint=CHECKPOINT).to('cuda')

        N = len(keypoints_2d)
        self._queries = torch.zeros(1, N, 3, device='cuda')
        self._queries[0, :, 0] = 0
        self._queries[0, :, 1] = torch.tensor([kp[1] for kp in keypoints_2d], dtype=torch.float32)
        self._queries[0, :, 2] = torch.tensor([kp[0] for kp in keypoints_2d], dtype=torch.float32)

        self._frame_buffer = []
        self._tracker_step = self._tracker.step
        self._tracker_initialized = False
        self._last_tracks_2d = keypoints_2d.copy()
        self._track_frame_count = 0  # <-- 新增
        print(f'[RealEnv] CoTracker3 initialized, tracking {N} keypoints, step={self._tracker_step}')

    def enable_tracking_visualization(self):
        """Start background thread for continuous capture + tracking + visualization."""
        if self._vis_thread is not None:
            return
        self._vis_enabled = True
        self._vis_thread = threading.Thread(target=self._tracking_vis_loop, daemon=True)
        self._vis_thread.start()
        print(f'{bcolors.OKGREEN}[RealEnv] Live tracking visualization started{bcolors.ENDC}')

    def _tracking_vis_loop(self):
        """Background: continuous capture → CoTracker feed → 3D update → cv2.imshow."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0), (0, 128, 255),
            (255, 128, 128), (128, 255, 128), (128, 128, 255),
        ]
        cv2.namedWindow("ReKep Tracking", cv2.WINDOW_NORMAL)

        while self._vis_enabled and not self._abort_event.is_set():
            # --- capture ---
            try:
                rgb, depth_raw = self._capture()
            except Exception:
                time.sleep(0.01)
                continue

            # --- CoTracker feed ---
            if self._tracker is not None and self._last_tracks_2d is not None:
                self._frame_buffer.append(rgb)
                self._track_frame_count += 1
                step = self._tracker_step

                if not self._tracker_initialized and self._track_frame_count >= step * 2:
                    chunk = np.stack(self._frame_buffer[-step * 2:])
                    video_chunk = torch.tensor(chunk, device='cuda').float().permute(0, 3, 1, 2)[None]
                    self._tracker(video_chunk, is_first_step=True, queries=self._queries)
                    self._tracker_initialized = True
                    print(f'{bcolors.OKGREEN}[RealEnv] CoTracker tracking started '
                          f'(frame {self._track_frame_count}){bcolors.ENDC}')
                elif self._tracker_initialized and self._track_frame_count % step == 0:
                    chunk = np.stack(self._frame_buffer[-step * 2:])
                    video_chunk = torch.tensor(chunk, device='cuda').float().permute(0, 3, 1, 2)[None]
                    pred_tracks, pred_visibility = self._tracker(video_chunk)
                    if pred_tracks is not None:
                        tracks_2d = pred_tracks[0, -1].cpu().numpy()  # (N, 2) [x, y]
                        vis = pred_visibility[0, -1].cpu().numpy()    # (N,)
                        self._last_visibility = vis
                        new_tracks = np.stack([tracks_2d[:, 1], tracks_2d[:, 0]], axis=-1)  # [row, col]
                        for i in range(len(new_tracks)):
                            if i in self._grasped_kp_indices:
                                continue
                            if vis[i] > 0.5:
                                self._last_tracks_2d[i] = new_tracks[i]

                # trim buffer to save memory
                if len(self._frame_buffer) > step * 4:
                    self._frame_buffer = self._frame_buffer[-step * 2:]

                # update 3D positions
                points_base = self._depth_to_points_base(depth_raw)
                ee_pos = self._read_ee_pose()[:3]
                tracked_3d = []
                for i, row_col in enumerate(self._last_tracks_2d):
                    if i in self._grasped_kp_indices:
                        tracked_3d.append(ee_pos.copy())
                    else:
                        r = int(np.clip(row_col[0], 0, depth_raw.shape[0] - 1))
                        c = int(np.clip(row_col[1], 0, depth_raw.shape[1] - 1))
                        tracked_3d.append(points_base[r, c])
                with self._kp_lock:
                    self.keypoints = np.array(tracked_3d)

            # --- visualization overlay ---
            img = rgb.copy()
            if self._last_tracks_2d is not None:
                with self._kp_lock:
                    kp_snap = self.keypoints.copy() if self.keypoints is not None else None
                for i, row_col in enumerate(self._last_tracks_2d):
                    r, c = int(row_col[0]), int(row_col[1])
                    color = colors[i % len(colors)]
                    if i in self._grasped_kp_indices:
                        cv2.rectangle(img, (c - 8, r - 8), (c + 8, r + 8), (255, 255, 255), 2)
                        label = f"{i}[EE]"
                    else:
                        cv2.circle(img, (c, r), 6, color, -1)
                        cv2.circle(img, (c, r), 6, (255, 255, 255), 1)
                        if self._last_visibility is not None and i < len(self._last_visibility) and self._last_visibility[i] <= 0.5:
                            cv2.circle(img, (c, r), 10, (128, 128, 128), 1)
                            label = f"{i}[frozen]"
                        else:
                            label = str(i)
                    if kp_snap is not None and i < len(kp_snap):
                        kp = kp_snap[i]
                        label += f" ({kp[0]:.3f},{kp[1]:.3f},{kp[2]:.3f})"
                    cv2.putText(img, label, (c + 8, r - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                    cv2.putText(img, label, (c + 8, r - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(display, f"frame={self._track_frame_count}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("ReKep Tracking", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') or key == ord('R'):
                self._abort_event.set()
                print(f"\n{bcolors.WARNING}>>> 'R' pressed — aborting... <<<{bcolors.ENDC}")

        cv2.destroyWindow("ReKep Tracking")

    def get_keypoint_positions(self):
        assert self.keypoints is not None, "Keypoints have not been registered yet."

        # Background thread handles capture + tracking → just read latest
        if self._vis_enabled:
            with self._kp_lock:
                return self.keypoints.copy()

        # Fallback: inline capture + tracking (no visualization mode)
        if self._tracker is None:
            return self.keypoints.copy()

        rgb, depth_raw = self._capture()
        self._frame_buffer.append(rgb)
        buf_len = len(self._frame_buffer)
        step = self._tracker_step

        if not self._tracker_initialized and buf_len >= step * 2:
            chunk = np.stack(self._frame_buffer[-step * 2:])
            video_chunk = torch.tensor(chunk, device='cuda').float().permute(0, 3, 1, 2)[None]
            self._tracker(video_chunk, is_first_step=True, queries=self._queries)
            self._tracker_initialized = True
        elif self._tracker_initialized and buf_len % step == 0:
            chunk = np.stack(self._frame_buffer[-step * 2:])
            video_chunk = torch.tensor(chunk, device='cuda').float().permute(0, 3, 1, 2)[None]
            pred_tracks, pred_visibility = self._tracker(video_chunk)
            if pred_tracks is not None:
                tracks_2d = pred_tracks[0, -1].cpu().numpy()
                vis = pred_visibility[0, -1].cpu().numpy()
                self._last_visibility = vis
                new_tracks = np.stack([tracks_2d[:, 1], tracks_2d[:, 0]], axis=-1)
                for i in range(len(new_tracks)):
                    if i in self._grasped_kp_indices:
                        continue
                    if vis[i] > 0.5:
                        self._last_tracks_2d[i] = new_tracks[i]

        points_base = self._depth_to_points_base(depth_raw)
        ee_pos = self._read_ee_pose()[:3]
        tracked_3d = []
        for i, row_col in enumerate(self._last_tracks_2d):
            if i in self._grasped_kp_indices:
                tracked_3d.append(ee_pos.copy())
            else:
                r = int(np.clip(row_col[0], 0, depth_raw.shape[0] - 1))
                c = int(np.clip(row_col[1], 0, depth_raw.shape[1] - 1))
                tracked_3d.append(points_base[r, c])

        self.keypoints = np.array(tracked_3d)
        return self.keypoints.copy()

    def _depth_to_points_base(self, depth_raw):
        """Convert depth image to 3D point cloud in robot base frame."""
        fx = self.head_intrinsics['fx']
        fy = self.head_intrinsics['fy']
        cx = self.head_intrinsics['cx']
        cy = self.head_intrinsics['cy']
        h, w = depth_raw.shape
        depth_m = depth_raw.astype(np.float32) * self.depth_raw_to_mm / 1000.0
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)
        z = depth_m
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_cam = np.stack([x, y, z], axis=-1)  # (H, W, 3)
        pts_flat = points_cam.reshape(-1, 3)
        pts_homo = np.hstack([pts_flat, np.ones((pts_flat.shape[0], 1))])
        pts_base = (self.T_head2base @ pts_homo.T).T[:, :3]
        points_base = pts_base.reshape(h, w, 3)
        return points_base

    # ======================================
    # = exposed functions (matching ReKepOGEnv)
    # ======================================
    def get_cam_obs(self):
        """Returns dict {cam_id: {'rgb', 'points', 'seg'}}."""
        rgb, depth_raw = self._capture()
        points = self._depth_to_points_base(depth_raw)
        seg = np.ones(rgb.shape[:2], dtype=np.int32)
        self.last_cam_obs = {
            0: {
                'rgb': rgb,
                'points': points,
                'seg': seg,
                'depth': depth_raw,
            }
        }
        return self.last_cam_obs

    def get_sdf_voxels(self, resolution, **kwargs):
        """Return all-positive SDF (no obstacles)."""
        shape = np.ceil((self.bounds_max - self.bounds_min) / resolution).astype(int)
        sdf_voxels = np.ones(shape) * 1.0
        return sdf_voxels

    def register_keypoints(self, keypoints):
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        self.keypoints = keypoints.copy()
        self._keypoint2object = {}
        for idx in range(len(keypoints)):
            self._keypoint2object[idx] = f"object_{idx}"
        print(f'{bcolors.OKGREEN}[RealEnv] Registered {len(keypoints)} keypoints{bcolors.ENDC}')

    def set_grasped_keypoints(self, indices):
        """Mark keypoints as grasped — their 3D will be bound to EE position."""
        self._grasped_kp_indices = set(indices)
        if indices:
            print(f'{bcolors.OKGREEN}[RealEnv] Grasped keypoints: {indices} → bound to EE{bcolors.ENDC}')

    def clear_grasped_keypoints(self):
        """Clear grasped keypoints — resume CoTracker tracking for all."""
        self._grasped_kp_indices.clear()
        print(f'{bcolors.OKGREEN}[RealEnv] Grasped keypoints cleared → all tracked by CoTracker{bcolors.ENDC}')

    def get_object_by_keypoint(self, keypoint_idx):
        assert self._keypoint2object is not None, "Keypoints have not been registered yet."
        return self._keypoint2object[keypoint_idx]

    def get_collision_points(self, noise=True):
        return None

    def get_ee_pose(self):
        return self._read_ee_pose()

    def get_ee_pos(self):
        return self._read_ee_pose()[:3]

    def get_ee_quat(self):
        return self._read_ee_pose()[3:]

    def get_arm_joint_postions(self):
        joint_q = self.robot.get_joint_q()
        if joint_q is None:
            return self.reset_joint_pos.copy()
        return np.array(joint_q)

    def is_grasping(self, candidate_obj=None):
        if candidate_obj is not None and self._grasped_kp_indices:
            try:
                kp_idx = int(candidate_obj.split("_")[1])
                return kp_idx in self._grasped_kp_indices
            except (IndexError, ValueError):
                pass
        gripper_width = self.robot.get_gripper_state()
        if gripper_width is None:
            return self._grasping
        return self._gripper_grasp_min < gripper_width < self._gripper_grasp_threshold

    def get_gripper_null_action(self):
        return 0.0

    def get_gripper_open_action(self):
        return -1.0

    def get_gripper_close_action(self):
        return 1.0

    def open_gripper(self):
        self.robot.set_gripper(position=self._gripper_open_width)
        time.sleep(self._gripper_wait_sec)
        self._grasping = False
        print(f'{bcolors.WARNING}[RealEnv | {get_clock_time()}] open_gripper(){bcolors.ENDC}')

    def close_gripper(self):
        self.robot.set_gripper(position=self._gripper_close_width)
        time.sleep(self._gripper_wait_sec)
        gripper_width = self.robot.get_gripper_state()
        if gripper_width is not None:
            self._grasping = self._gripper_grasp_min < gripper_width < self._gripper_grasp_threshold
        else:
            self._grasping = True
        print(f'{bcolors.WARNING}[RealEnv | {get_clock_time()}] close_gripper() -> '
            f'width={gripper_width}, grasping={self._grasping}{bcolors.ENDC}')

    def execute_action(self, action, precise=True):
        action = np.array(action).copy()
        assert action.shape == (8,), f"Expected action shape (8,), got {action.shape}"
        target_pose = action[:7]
        gripper_action = action[7]

        # bounds check
        if np.any(target_pose[:3] < self.bounds_min) or np.any(target_pose[:3] > self.bounds_max):
            print(f'{bcolors.WARNING}[RealEnv] Target OOB, clipping{bcolors.ENDC}')
            target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

        # interpolation
        current_pose = self.get_ee_pose()
        pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
        rot_diff = angle_between_quats(current_pose[3:7], target_pose[3:7])
        pos_is_close = pos_diff < self.interpolate_pos_step_size
        rot_is_close = rot_diff < self.interpolate_rot_step_size

        if pos_is_close and rot_is_close:
            pose_seq = np.array([target_pose])
        else:
            num_steps = get_linear_interpolation_steps(
                current_pose, target_pose,
                self.interpolate_pos_step_size, self.interpolate_rot_step_size,
            )
            pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)

        # execute waypoints
        for i, pose in enumerate(pose_seq):
            if self._abort_event.is_set():
                print(f'{bcolors.WARNING}[RealEnv] Abort detected, stopping motion{bcolors.ENDC}')
                break
            pos = [float(pose[0]), float(pose[1]), float(pose[2])]
            quat = [float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])]
            success = self.robot.set_end_pose(pos, quat, blocking=True)
            if not success:
                print(f'{bcolors.WARNING}[RealEnv] set_end_pose failed at waypoint {i}{bcolors.ENDC}')
                break

        # settle
        time.sleep(self._move_wait_sec)

        # gripper action
        if gripper_action == self.get_gripper_open_action():
            self.open_gripper()
        elif gripper_action == self.get_gripper_close_action():
            self.close_gripper()

        # compute error
        actual_pose = self.get_ee_pose()
        pos_error = np.linalg.norm(actual_pose[:3] - target_pose[:3])
        rot_error = angle_between_quats(actual_pose[3:], target_pose[3:])

        self.verbose and print(
            f'{bcolors.OKCYAN}[RealEnv | {get_clock_time()}] execute_action: '
            f'pos_err={pos_error * 1000:.1f}mm rot_err={np.rad2deg(rot_error):.1f}°{bcolors.ENDC}')

        self.step_counter += 1
        return pos_error, rot_error

    def reset(self):
        self.robot.set_joint_positions(list(self.reset_joint_pos), blocking=True)
        time.sleep(0.5)
        self.robot.set_gripper(position=self._gripper_open_width)
        time.sleep(self._gripper_wait_sec)
        self._grasping = False
        self.video_cache = []
        self.step_counter = 0
        print(f'{bcolors.HEADER}[RealEnv] Reset done.{bcolors.ENDC}')

    def sleep(self, seconds):
        time.sleep(seconds)

    def save_video(self, save_path=None):
        print(f'{bcolors.WARNING}[RealEnv] save_video: no-op{bcolors.ENDC}')
        return "no_video"

    def is_abort_requested(self):
        return self._abort_event.is_set()

    def request_abort(self):
        self._abort_event.set()

    def emergency_home(self):
        """Open gripper and return to home — called on abort."""
        print(f'{bcolors.WARNING}[RealEnv] Emergency stop: opening gripper...{bcolors.ENDC}')
        try:
            self.robot.set_gripper(position=self._gripper_open_width)
            time.sleep(self._gripper_wait_sec)
        except Exception as e:
            print(f'{bcolors.WARNING}[RealEnv] Gripper open failed: {e}{bcolors.ENDC}')
        self._grasping = False
        print(f'{bcolors.WARNING}[RealEnv] Returning to home joint...{bcolors.ENDC}')
        try:
            self.robot.set_joint_positions(list(self.reset_joint_pos), blocking=True)
            time.sleep(0.5)
        except Exception as e:
            print(f'{bcolors.WARNING}[RealEnv] Return home failed: {e}{bcolors.ENDC}')
        print(f'{bcolors.OKGREEN}[RealEnv] Emergency home complete.{bcolors.ENDC}')

    def shutdown(self):
        self._vis_enabled = False
        if self._pipeline:
            self._pipeline.stop()
            print(f'{bcolors.OKGREEN}[RealEnv] Camera stopped{bcolors.ENDC}')
        self.robot.shutdown()
        print(f'{bcolors.OKGREEN}[RealEnv] Robot shutdown{bcolors.ENDC}')