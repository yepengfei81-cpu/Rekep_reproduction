"""
ReKep main loop for real robot (Phase 0: perception + solver, no arm execution).
Usage:
    python real_main.py --task pen --use_cached_query
"""
import sys
import threading
import select
import torch
import numpy as np
import json
import os
import argparse
from real_environment import RealEnvironment
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from real_ik_solver import RealIKSolver
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
from visualizer import Visualizer
import transform_utils as T
import cv2
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

def _keyboard_abort_listener(abort_event):
    """Background thread: press 'R' to abort task and return home."""
    import tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not abort_event.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == 'r':
                    abort_event.set()
                    print(f"\n{bcolors.WARNING}>>> 'R' pressed — aborting task, returning home... <<<{bcolors.ENDC}")
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

class RealMain:
    def __init__(self, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # initialize keypoint proposer and constraint generator
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
        # initialize environment
        env_config = global_config['env']
        env_config.update(global_config.get('real_env', {}))  # merge real_env overrides
        self.env = RealEnvironment(env_config, verbose=False)
        # setup dummy IK solver
        ik_solver = RealIKSolver(
            reset_joint_pos=self.env.reset_joint_pos,
            world2robot_homo=self.env.world2robot_homo,
        )
        # initialize solvers
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, self.env.reset_joint_pos)
        self.path_solver = PathSolver(global_config['path_solver'], ik_solver, self.env.reset_joint_pos)
        # initialize visualizer
        if self.visualize:
            self.visualizer = Visualizer(global_config['visualizer'], self.env)

    def perform_task(self, instruction, prompts, rekep_program_dir=None):
        # 手眼相机精校用要抓取的物体作为 prompt（prompts[0] 约定为抓取物）
        self._handeye_prompt = prompts[0] if prompts else None
        self.env.reset()
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[0]['rgb']
        points = cam_obs[0]['points']

        if rekep_program_dir is None:
            # ===== SAM3 =====
            from test_scripts.test_dinov3 import sam3_segment
            seg_mask, prompt_labels = sam3_segment(rgb, prompts, confidence=0.5)
            torch.cuda.empty_cache()
            
            # ===== DINOv3 =====
            keypoints, keypoints_2d, projected_img = self.keypoint_proposer.get_keypoints(
                rgb, points, seg_mask
            )
            del self.keypoint_proposer
            torch.cuda.empty_cache()

            # ===== Visualize =====
            self._visualize_perception(rgb, seg_mask, prompt_labels, projected_img)

            # ===== VLM =====
            metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
            rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
            
            # ===== CoTracker3 =====
            self.env.init_tracker(keypoints_2d)  # keypoints_2d: (N, 2) [row, col]

        self._execute(rekep_program_dir)

    def perform_multi_task(self, instruction, prompts, num_blocks, rekep_program_dir=None):
        """循环抓取多个积木并放入碗中。"""
        # 手眼相机精校用要抓取的物体作为 prompt（prompts[0] 约定为抓取物）
        self._handeye_prompt = prompts[0] if prompts else None
        from test_scripts.test_dinov3 import sam3_segment
        global_config = get_config(config_path="./configs/config.yaml")
        # 释放初始化时加载的 keypoint_proposer（后面每轮重新创建）
        if hasattr(self, 'keypoint_proposer'):
            del self.keypoint_proposer
            torch.cuda.empty_cache()

        cached_dir = rekep_program_dir

        for block_idx in range(num_blocks):
            print(f"\n{bcolors.OKGREEN}[Main] ===== 积木 {block_idx+1}/{num_blocks} ====={bcolors.ENDC}")

            # 停止上一轮的追踪线程，回到 home
            self.env.stop_tracking_visualization()
            self.env.reset()
            if self.env._handeye_segmenter is not None:
                del self.env._handeye_segmenter
                self.env._handeye_segmenter = None
                torch.cuda.empty_cache()

            if self.env.is_abort_requested():
                break

            cam_obs = self.env.get_cam_obs()
            rgb = cam_obs[0]['rgb']
            points = cam_obs[0]['points']

            # ===== SAM3 =====
            kp_proposer = KeypointProposer(global_config['keypoint_proposer'])
            seg_mask, prompt_labels = sam3_segment(rgb, prompts, confidence=0.5)
            torch.cuda.empty_cache()

            # 检测是否还有积木
            if seg_mask is None or np.sum(seg_mask > 0) < 100:
                print(f"{bcolors.WARNING}[Main] 未检测到积木，已完成 {block_idx} 个，停止。{bcolors.ENDC}")
                del kp_proposer
                break

            # ===== DINOv3 =====
            keypoints, keypoints_2d, projected_img = kp_proposer.get_keypoints(rgb, points, seg_mask)
            del kp_proposer
            torch.cuda.empty_cache()

            # ===== Visualize =====
            self._visualize_perception(rgb, seg_mask, prompt_labels, projected_img, block_idx=block_idx+1)

            metadata = {'init_keypoint_positions': keypoints.tolist(), 'num_keypoints': len(keypoints)}

            if cached_dir is None:
                # 第一次：调用 VLM 生成约束
                cached_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
            else:
                # 后续：复用约束文件，只更新 keypoint 坐标
                meta_path = os.path.join(cached_dir, 'metadata.json')
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if meta['num_keypoints'] == len(keypoints):
                    meta['init_keypoint_positions'] = keypoints.tolist()
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=2)
                else:
                    print(f"{bcolors.WARNING}[Main] keypoint 数量变化，重新生成约束...{bcolors.ENDC}")
                    cached_dir = self.constraint_generator.generate(projected_img, instruction, metadata)

            # ===== CoTracker3 =====
            self.env.init_tracker(keypoints_2d)

            self._execute(cached_dir)

        print(f"{bcolors.OKGREEN}[Main] 多积木任务结束。{bcolors.ENDC}")

    def _visualize_perception(self, rgb, seg_mask, prompt_labels, projected_img, block_idx=None):
        """将 SAM3 分割 + DINOv3 关键点拼接保存到文件。"""
        from test_scripts.test_dinov3 import visualize_sam3_masks
        seg_vis = visualize_sam3_masks(rgb, seg_mask, prompt_labels)
        seg_bgr = cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR)
        kp_bgr  = cv2.cvtColor(projected_img, cv2.COLOR_RGB2BGR)
        h1, w1 = seg_bgr.shape[:2]
        h2, w2 = kp_bgr.shape[:2]
        if h1 != h2:
            kp_bgr = cv2.resize(kp_bgr, (int(w2 * h1 / h2), h1))
        combined = np.hstack([seg_bgr, kp_bgr])
        suffix = f"_block{block_idx}" if block_idx is not None else ""
        save_path = os.path.join("vlm_query", f"perception_vis{suffix}.jpg")
        cv2.imwrite(save_path, combined)
        print(f"{bcolors.OKGREEN}[Vis] 左:SAM3分割  右:DINOv3关键点 → 已保存: {save_path}{bcolors.ENDC}")

    def _execute(self, rekep_program_dir):
        # load metadata
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        # register keypoints to be tracked
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        self.env.enable_tracking_visualization()
        # load constraints
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self.constraint_fns[stage] = stage_dict

        # bookkeeping
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True

        # main loop
        self.last_sim_step_counter = -np.inf
        self._update_stage(1)
        max_iterations = 50  # safety limit for Phase 0
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            # --- abort check ---
            if self.env.is_abort_requested():
                print(f"{bcolors.WARNING}[Main] Abort requested, emergency home...{bcolors.ENDC}")
                self.env.emergency_home()
                return
            # 修改 _execute 主循环中的关键点获取部分：
            scene_keypoints = self.env.get_keypoint_positions()

            # 对静态关键点（非被抓），用初始位置覆盖 CoTracker 漂移值
            init_kps = np.array(self.program_info['init_keypoint_positions'])
            for i in range(len(scene_keypoints)):
                if not self.keypoint_movable_mask[i + 1]:  # +1 因为 mask[0] 是 EE
                    scene_keypoints[i] = init_kps[i]

            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints], axis=0)
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_postions()
            exclude_pos = None
            if self.is_release_stage:
                release_kp_idx = self.program_info['release_keypoints'][self.stage - 1]
                if release_kp_idx != -1:
                    exclude_pos = scene_keypoints[release_kp_idx]
            self.sdf_voxels = self.env.get_sdf_voxels(
                self.config['sdf_voxel_size'],
                exclude_target_pos=exclude_pos,
                exclude_target_radius=0.08,
            )
            self.collision_points = self.env.get_collision_points()

            # ====================================
            # = decide whether to backtrack
            # ====================================
            backtrack = False
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]['path']
                for constraints in path_constraints:
                    violation = constraints(self.keypoints[0], self.keypoints[1:])
                    if violation > self.config['constraint_tolerance']:
                        backtrack = True
                        break
            if backtrack:
                for new_stage in range(self.stage - 1, 0, -1):
                    path_constraints = self.constraint_fns[new_stage]['path']
                    if len(path_constraints) == 0:
                        break
                    all_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints(self.keypoints[0], self.keypoints[1:])
                        if violation > self.config['constraint_tolerance']:
                            all_satisfied = False
                            break
                    if all_satisfied:
                        break
                print(f"{bcolors.HEADER}[stage={self.stage}] backtrack to stage {new_stage}{bcolors.ENDC}")
                self._update_stage(new_stage)
            else:
                # ====================================
                # = get optimized plan
                # ====================================
                if self.last_sim_step_counter == self.env.step_counter:
                    print(f"{bcolors.WARNING}sim did not step forward{bcolors.ENDC}")
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
                next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
                self.first_iter = False
                self.action_queue = next_path.tolist()
                self.last_sim_step_counter = self.env.step_counter

                # ====================================
                # = execute
                # ====================================
                count = 0
                while len(self.action_queue) > 0 and count < self.config['action_steps_per_iter']:
                    if self.env.is_abort_requested():
                        break
                    next_action = self.action_queue.pop(0)
                    precise = len(self.action_queue) == 0
                    self.env.execute_action(next_action, precise=precise)
                    count += 1
                if len(self.action_queue) == 0:
                    if self.is_grasp_stage:
                        if not self._execute_grasp_action():
                            self.env.open_gripper()
                            return False 
                    elif self.is_release_stage:
                        self._execute_release_action()
                    # if completed
                    if self.stage == self.program_info['num_stages']:
                        self.env.sleep(1.0)
                        print(f"{bcolors.OKGREEN}Task completed!{bcolors.ENDC}")
                        self.env.open_gripper()
                        self.env.reset()  # return to home
                        return
                    self._update_stage(self.stage + 1)

        print(f"{bcolors.WARNING}Reached max iterations ({max_iterations}), stopping.{bcolors.ENDC}")
        self.env.open_gripper()
        self.env.reset()  # return to home

    def _get_next_subgoal(self, from_scratch):
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        subgoal_pose, debug_dict = self.subgoal_solver.solve(
            self.curr_ee_pose,
            self.keypoints,
            self.keypoint_movable_mask,
            subgoal_constraints,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.is_grasp_stage,
            self.curr_joint_pos,
            from_scratch=from_scratch,
        )
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        if self.is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        # ← 新增：金字塔放置阶段覆盖 yaw
        if self.is_release_stage and getattr(self, '_pyramid_release_yaw', None) is not None:
            from scipy.spatial.transform import Rotation
            r_base = Rotation.from_euler('y', np.pi / 2)          # 夹爪X轴朝下
            r_yaw  = Rotation.from_euler('z', self._pyramid_release_yaw)  # 绕Z旋转yaw
            subgoal_pose[3:7] = (r_yaw * r_base).as_quat()        # 和采集时完全一致
        # ← 新增结束
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose)
        return subgoal_pose

    def _get_next_path(self, next_subgoal, from_scratch):
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(
            self.curr_ee_pose,
            next_subgoal,
            self.keypoints,
            self.keypoint_movable_mask,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.curr_joint_pos,
            from_scratch=from_scratch,
        )
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        if self.visualize:
            self.visualizer.visualize_path(processed_path)
        return processed_path

    def _process_path(self, path):
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(
            full_control_points[0], full_control_points[-1],
            self.config['interpolate_pos_step_size'],
            self.config['interpolate_rot_step_size'],
        )
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        if self.is_grasp_stage and self.is_release_stage:
            print(f"{bcolors.WARNING}[Warning] Stage {stage} 同时有 grasp 和 release keypoint（VLM生成错误），保留 release，忽略 grasp{bcolors.ENDC}")
            self.is_grasp_stage = False
        if self.is_grasp_stage:
            self.env.open_gripper()
        self.action_queue = []
        self._update_keypoint_movable_mask()
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        for i in range(1, len(self.keypoint_movable_mask)):
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        pregrasp_pose = self.env.get_ee_pose()
        from scipy.spatial.transform import Rotation
        euler = Rotation.from_quat(pregrasp_pose[3:7]).as_euler('zyx', degrees=True)
        print(f"{bcolors.OKBLUE}[DEBUG Grasp] raw quat={pregrasp_pose[3:7].round(4)}, "
            f"ZYX euler={euler.round(1)}{bcolors.ENDC}")
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([self.config['grasp_depth'], 0, 0])

        # ===== 手眼相机精校 XY =====
        prompt = getattr(self, '_handeye_prompt', None)
        if prompt and hasattr(self.env, 'handeye_refine_position'):
            grasp_kp_idx = self.program_info['grasp_keypoints'][self.stage - 1]
            ref_z = grasp_pose[2]
            from scipy.spatial.transform import Rotation as _R
            ref_yaw = _R.from_quat(pregrasp_pose[3:7]).as_euler('zyx')[0]
            refined = self.env.handeye_refine_position(prompt, ref_z, ref_yaw)
            if refined is not None:
                print(f"{bcolors.OKGREEN}[Grasp] Hand-eye XY校正: "
                    f"({grasp_pose[0]:.4f},{grasp_pose[1]:.4f}) → "
                    f"({refined[0]:.4f},{refined[1]:.4f}){bcolors.ENDC}")
                grasp_pose[0] = refined[0]
                grasp_pose[1] = refined[1]
                grasp_pose[2] = refined[2]
                # 绕进近轴旋转，使夹爪开合方向对准砖块长轴
                R_curr = _R.from_quat(grasp_pose[3:7])
                R_mat = R_curr.as_matrix()
                approach_axis = R_mat[:, 0]   # EE X轴（进近轴）
                opening_axis  = R_mat[:, 1]   # EE Y轴（开合轴，若不对改成[:, 2]）
                brick_dir = np.array([-np.sin(refined[3]), np.cos(refined[3]), 0.0])
                if np.dot(opening_axis[:2], brick_dir[:2]) < 0:
                    brick_dir = -brick_dir
                def proj_perp(v, axis):
                    return v - np.dot(v, axis) * axis
                cur_proj = proj_perp(opening_axis, approach_axis)
                tgt_proj = proj_perp(brick_dir,    approach_axis)
                if np.linalg.norm(cur_proj) > 0.01 and np.linalg.norm(tgt_proj) > 0.01:
                    cur_proj /= np.linalg.norm(cur_proj)
                    tgt_proj /= np.linalg.norm(tgt_proj)
                    cos_a = np.clip(np.dot(cur_proj, tgt_proj), -1, 1)
                    sin_a = np.dot(np.cross(cur_proj, tgt_proj), approach_axis)
                    delta = np.arctan2(sin_a, cos_a)
                    print(f"{bcolors.OKGREEN}[Grasp] Hand-eye 进近轴旋转校正: "
                        f"Δ={np.rad2deg(delta):.1f}°{bcolors.ENDC}")
                    R_delta = _R.from_rotvec(delta * approach_axis)
                    grasp_pose[3:7] = (R_delta * R_curr).as_quat()

        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])
        self.env.execute_action(grasp_action, precise=True)
        grasp_kp_idx = self.program_info['grasp_keypoints'][self.stage - 1]
        if grasp_kp_idx != -1:
            self.env.set_grasped_keypoints([grasp_kp_idx])
        
        # ← 新增：检测是否真的抓到了
        success = self.env.is_grasping()
        if not success:
            print(f"{bcolors.WARNING}[Grasp] 夹爪未检测到物体（width过小），抓取失败。{bcolors.ENDC}")
        return success

    def _execute_release_action(self):
        if getattr(self, '_pyramid_release_yaw', None) is not None:
            from scipy.spatial.transform import Rotation, Slerp
            curr_pose = self.env.get_ee_pose()

            # 目标姿态：和 teach_targets 采集时完全一致（夹爪X轴朝下 + 指定yaw）
            r_base   = Rotation.from_euler('y', np.pi / 2)
            r_yaw    = Rotation.from_euler('z', self._pyramid_release_yaw)
            r_target = r_yaw * r_base
            target_quat = r_target.as_quat()

            # 计算当前与目标的角度差，决定插值步数
            r_curr    = Rotation.from_quat(curr_pose[3:7])
            delta_rad = (r_curr.inv() * r_target).magnitude()
            delta_deg = np.rad2deg(delta_rad)
            print(f"{bcolors.OKBLUE}[Release] 姿态差: {delta_deg:.1f}°, 目标YAW: "
                f"{np.rad2deg(self._pyramid_release_yaw):.1f}°{bcolors.ENDC}")

            n_steps  = max(2, int(np.ceil(delta_rad / 0.26)))  # 每步 ≤15°
            key_rots = Rotation.from_quat([curr_pose[3:7], target_quat])
            slerp    = Slerp([0, 1], key_rots)

            target_pose = curr_pose.copy()
            for i in range(1, n_steps + 1):
                if self.env.is_abort_requested():
                    break
                target_pose[3:7] = slerp(i / n_steps).as_quat()
                action = np.concatenate([target_pose, [self.env.get_gripper_null_action()]])
                self.env.execute_action(action, precise=False)

        self.env.open_gripper()
        self.env.clear_grasped_keypoints()

    def _sam3_segment_shared(self, segmenter, rgb, prompts):
        """使用已加载的 SAM3 实例分割头部相机图像，不重新加载模型。"""
        H, W = rgb.shape[:2]
        seg_mask = np.zeros((H, W), dtype=np.int32)
        prompt_labels = {0: "background"}
        current_id = 1

        for prompt in prompts:
            print(f"  [SAM3] '{prompt}' ...", end=" ")
            masks = segmenter.segment_rgb(rgb, prompt)
            if masks is None or len(masks) == 0:
                print("未检测到")
                continue
            print(f"{len(masks)} 个")
            for mk in masks:
                mask = mk.squeeze()
                if mask.shape != (H, W):
                    mask = cv2.resize(mask.astype(np.float32), (W, H),
                                    interpolation=cv2.INTER_NEAREST)
                binary = mask > 0.5
                new_region = binary & (seg_mask == 0)
                if int(new_region.sum()) < 100:
                    continue
                seg_mask[new_region] = current_id
                prompt_labels[current_id] = f"{prompt}_{current_id}"
                current_id += 1

        print(f"SAM3 分割完成: {current_id - 1} 个物体")
        return seg_mask, prompt_labels

    def perform_pyramid_task(self, prompts, targets_json_path='/home/ypf/qiuzhiarm_LLM/config/play_targets_brick.json'):
        """按 JSON 中 order 顺序将积木堆叠成金字塔，VLM 负责生成约束。"""
        import json as _json
        from perception.sam3_segmenter import create_segmenter
        global_config = get_config(config_path="./configs/config.yaml")

        # 加载目标位置，按 order 排序
        with open(targets_json_path, 'r') as f:
            targets_data = _json.load(f)
        targets = sorted(targets_data['targets'], key=lambda t: t['order'])  # 6个，order=1..6

        # 释放当前 keypoint_proposer，后续每轮重建
        if hasattr(self, 'keypoint_proposer'):
            del self.keypoint_proposer
            torch.cuda.empty_cache()

        # ===== 整个任务只加载一次 SAM3，头部相机和手眼相机共用 =====
        print("[Pyramid] 加载 SAM3 模型（整个任务共用一次）...")
        shared_segmenter = create_segmenter(
            "/home/ypf/sam3-main/checkpoint/sam3.pt", confidence=0.5
        )
        self.env._handeye_segmenter = shared_segmenter  # 手眼相机直接使用同一实例
        torch.cuda.empty_cache()

        actually_placed = []  
        try:
            for tgt in targets:
                order = tgt['order']
                tgt_pos = np.array([tgt['x'], tgt['y'], tgt['z']])
                print(f"\n{bcolors.OKGREEN}[Pyramid] ===== 放置第 {order}/6 块 → 目标 {tgt_pos} ====={bcolors.ENDC}")

                self.env.stop_tracking_visualization()
                self.env.reset()
                # 只释放 CoTracker，不释放 SAM3（共用实例）
                if self.env._tracker is not None:
                    del self.env._tracker
                    self.env._tracker = None
                    self.env._tracker_initialized = False
                torch.cuda.empty_cache()
                if self.env.is_abort_requested():
                    break

                cam_obs = self.env.get_cam_obs()
                rgb = cam_obs[0]['rgb']
                points = cam_obs[0]['points']

                # ===== SAM3（使用共享实例，不重新加载模型）=====
                kp_proposer = KeypointProposer(global_config['keypoint_proposer'])
                seg_mask, prompt_labels = self._sam3_segment_shared(shared_segmenter, rgb, prompts)
                torch.cuda.empty_cache()
                if seg_mask is None or np.sum(seg_mask > 0) < 100:
                    print(f"{bcolors.WARNING}[Pyramid] 未检测到积木，停止。{bcolors.ENDC}")
                    del kp_proposer
                    break

                # ===== DINOv3 =====
                brick_keypoints, brick_keypoints_2d, projected_img = kp_proposer.get_keypoints(rgb, points, seg_mask)
                del kp_proposer
                torch.cuda.empty_cache()
                K = len(brick_keypoints)  # 积木关键点数量

                # ===== 追加 6 个目标位置作为虚拟关键点 =====
                target_positions = np.array([[t['x'], t['y'], t['z']] for t in targets])  # shape (6, 3)
                all_keypoints = np.concatenate([brick_keypoints, target_positions], axis=0)  # (K+6, 3)

                # ===== 找最近的积木关键点作为 src，排除已放置位置 =====
                EXCLUSION_RADIUS = 0.12  # 6cm，已放置的砖块排除半径

                valid_mask = np.ones(len(brick_keypoints), dtype=bool)
                for px, py in actually_placed:
                    too_close = np.linalg.norm(brick_keypoints[:, :2] - np.array([px, py]), axis=1) < EXCLUSION_RADIUS
                    valid_mask &= ~too_close

                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) == 0:
                    print(f"{bcolors.WARNING}[Pyramid] 警告：所有关键点都在已放置区域内，使用全部关键点。{bcolors.ENDC}")
                    valid_indices = np.arange(len(brick_keypoints))

                dists = np.linalg.norm(brick_keypoints[valid_indices, :2] - tgt_pos[:2], axis=1)
                src_idx = int(valid_indices[np.argmin(dists)])
                tgt_idx = K + (order - 1)  # 对应当前目标的虚拟关键点

                # ===== 可视化 =====
                self._visualize_perception(rgb, seg_mask, prompt_labels, projected_img, block_idx=order)

                # ===== 构造 instruction，明确告知 VLM 关键点语义 =====
                instruction = (
                    f"Pick up the brick at keypoint {src_idx} and place it precisely at keypoint {tgt_idx}. "
                    f"Keypoints 0 to {K-1} are bricks on the table (colored by object). "
                    f"Keypoints {K} to {K+5} are fixed pre-measured target positions (you cannot grasp them). "
                    f"The task requires 2 stages: stage 1 grasp keypoint {src_idx}, "
                    f"stage 2 place it at keypoint {tgt_idx}."
                )

                # ===== VLM 生成约束 =====
                metadata = {
                    'init_keypoint_positions': all_keypoints.tolist(),
                    'num_keypoints': len(all_keypoints),
                }
                rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)

                # ===== CoTracker 只初始化积木关键点（目标点无需追踪）=====
                self.env.init_tracker(brick_keypoints_2d)

                # ===== 执行 =====
                self._handeye_prompt = prompts[0] if prompts else "brick"
                self._pyramid_release_yaw = tgt['yaw']
                MAX_GRASP_RETRIES = 3
                for attempt in range(MAX_GRASP_RETRIES):
                    result = self._execute(rekep_program_dir)
                    if result is not False:
                        # 记录目标位置（EE已回home，用JSON目标坐标）
                        actually_placed.append((tgt_pos[0], tgt_pos[1]))
                        print(f"[Pyramid] 已放置位置记录: ({tgt_pos[0]:.3f}, {tgt_pos[1]:.3f})")
                        break
                    print(f"{bcolors.WARNING}[Pyramid] 抓取失败，第 {attempt+1}/{MAX_GRASP_RETRIES} 次重试...{bcolors.ENDC}")
                    # 清理追踪线程和tracker状态
                    self.env.stop_tracking_visualization()
                    self.env.open_gripper()
                    self.env.reset()
                    if self.env._tracker is not None:
                        del self.env._tracker
                        self.env._tracker = None
                        self.env._tracker_initialized = False
                    torch.cuda.empty_cache()
                    # 重新初始化tracker（复用同一批关键点2D坐标）
                    self.env.init_tracker(brick_keypoints_2d)
                else:
                    print(f"{bcolors.WARNING}[Pyramid] 连续 {MAX_GRASP_RETRIES} 次抓取失败，跳过此块。{bcolors.ENDC}")

        finally:
            # 任务结束后统一释放 SAM3
            print("[Pyramid] 释放共享 SAM3 模型...")
            self.env._handeye_segmenter = None
            del shared_segmenter
            torch.cuda.empty_cache()

        print(f"{bcolors.OKGREEN}[Pyramid] 金字塔堆叠任务结束。{bcolors.ENDC}")

if __name__ == "__main__":
    import tty, termios

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen')
    parser.add_argument('--use_cached_query', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--prompts', type=str, default='bowl', help='SAM3 prompt')
    parser.add_argument('--instruction', type=str, default=None,
                        help='Task instruction for VLM (default: "Pick up the {task}")')
    parser.add_argument('--num_blocks', type=int, default=1, help='要抓取的积木数量')
    parser.add_argument('--pyramid', action='store_true', help='执行积木金字塔堆叠任务')
    parser.add_argument('--targets_json', type=str,
                        default='/home/ypf/qiuzhiarm_LLM/config/play_targets_brick.json',
                        help='金字塔目标位置 JSON 文件路径')    
    args = parser.parse_args()

    task = RealMain(visualize=args.visualize)
    rekep_program_dir = os.path.join('vlm_query', args.task) if args.use_cached_query else None
    prompts = [p.strip() for p in args.prompts.split(",")]
    instruction = args.instruction if args.instruction else f"Pick up the {args.task}"

    fd = sys.stdin.fileno()
    old_terminal_settings = termios.tcgetattr(fd)

    abort_event = task.env._abort_event
    kb_thread = threading.Thread(target=_keyboard_abort_listener, args=(abort_event,), daemon=True)
    kb_thread.start()
    print(f"{bcolors.OKGREEN}[Info] Press 'R' at any time to abort and return home.{bcolors.ENDC}")

    try:
        if args.pyramid:
            task.perform_pyramid_task(
                prompts=prompts,
                targets_json_path=args.targets_json,
            )
        elif args.num_blocks > 1:
            task.perform_multi_task(
                instruction,
                prompts=prompts,
                num_blocks=args.num_blocks,
                rekep_program_dir=rekep_program_dir,
            )
        else:
            task.perform_task(
                instruction,
                prompts=prompts,
                rekep_program_dir=rekep_program_dir,
            )
    except KeyboardInterrupt:
        print(f"\n{bcolors.WARNING}[Interrupt] Ctrl+C — emergency home...{bcolors.ENDC}")
        task.env.emergency_home()
    finally:
        abort_event.set()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_terminal_settings)
        task.env.shutdown()