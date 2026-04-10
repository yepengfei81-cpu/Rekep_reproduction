"""
Dummy IK solver for Phase 0 (no real IK checking).
Returns success for all queries so the optimizer treats every pose as reachable.
"""
import numpy as np


class _DummyIKResult:
    __slots__ = ('success', 'num_descents', 'position_error', 'cspace_position')

    def __init__(self, num_joints=6):
        self.success = True
        self.num_descents = 0
        self.position_error = 0.0
        self.cspace_position = np.zeros(num_joints)


class RealIKSolver:
    def __init__(self, reset_joint_pos, world2robot_homo=None):
        self.reset_joint_pos = reset_joint_pos
        self.world2robot_homo = world2robot_homo if world2robot_homo is not None else np.eye(4)

    def solve(self, target_pose_homo, **kwargs):
        return _DummyIKResult(num_joints=len(self.reset_joint_pos))