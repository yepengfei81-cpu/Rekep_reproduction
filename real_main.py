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
        self.env.reset()
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[0]['rgb']
        points = cam_obs[0]['points']

        if rekep_program_dir is None:
            # ===== SAM3 =====
            from test_scripts.test_dinov3 import sam3_segment
            seg_mask, prompt_labels = sam3_segment(rgb, prompts, confidence=0.3)
            torch.cuda.empty_cache()
            
            # ===== DINOv3 =====
            keypoints, keypoints_2d, projected_img = self.keypoint_proposer.get_keypoints(
                rgb, points, seg_mask
            )
            del self.keypoint_proposer
            torch.cuda.empty_cache()
            
            # ===== VLM =====
            metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
            rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
            
            # ===== CoTracker3 =====
            self.env.init_tracker(keypoints_2d)  # keypoints_2d: (N, 2) [row, col]

        self._execute(rekep_program_dir)

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
            scene_keypoints = self.env.get_keypoint_positions()
            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints], axis=0)
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_postions()
            self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
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
                        self._execute_grasp_action()
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
        assert self.is_grasp_stage + self.is_release_stage <= 1
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
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([self.config['grasp_depth'], 0, 0])
        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])
        self.env.execute_action(grasp_action, precise=True)
        grasp_kp_idx = self.program_info['grasp_keypoints'][self.stage - 1]
        if grasp_kp_idx != -1:
            self.env.set_grasped_keypoints([grasp_kp_idx])

    def _execute_release_action(self):
        self.env.open_gripper()
        self.env.clear_grasped_keypoints()


if __name__ == "__main__":
    import tty, termios

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen')
    parser.add_argument('--use_cached_query', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--prompts', type=str, default='bowl', help='SAM3 prompt')
    parser.add_argument('--instruction', type=str, default=None,
                        help='Task instruction for VLM (default: "Pick up the {task}")')
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