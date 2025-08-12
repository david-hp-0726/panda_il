# helpers.py
# Exact copies of your pure helpers (no logic changes).

import os
import math
from typing import List, Tuple

import numpy as np
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


WS_AABB = dict(x=(0.2, 0.7), y=(-0.4, 0.4), z=(0.0, 0.6))
BASE_LINK = "panda_link0"

# -----------------------------
# Helpers (as-is)
# -----------------------------
def point_in_candidate_box(p, cx, cy, cz, sx, sy, sz, m):
    hx, hy, hz = sx * 0.5 + m, sy * 0.5 + m, sz * 0.5 + m
    return (abs(p[0] - cx) <= hx and abs(p[1] - cy) <= hy and abs(p[2] - cz) <= hz)

def rnd(a, b):
    return a + np.random.rand() * (b - a)


def quat_to_np(q):
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float64)


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_multiply(q1, q2):
    # (x, y, z, w)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dtype=np.float64)


def quat_error(q_curr, q_goal):
    # returns quaternion q_err such that q_goal = q_err * q_curr
    # ensure shortest path (positive scalar part)
    q_curr = np.array(q_curr, dtype=np.float64)
    q_goal = np.array(q_goal, dtype=np.float64)
    q_err = quat_multiply(q_goal, quat_conjugate(q_curr))
    if q_err[3] < 0:
        q_err = -q_err
    return q_err


def pose_distance(p1, p2) -> float:
    dx = p1.pose.position.x - p2.pose.position.x
    dy = p1.pose.position.y - p2.pose.position.y
    dz = p1.pose.position.z - p2.pose.position.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def msg_robot_state_from_q(joint_names: List[str], q: np.ndarray) -> RobotState:
    rstate = RobotState()
    rstate.joint_state = JointState()
    rstate.joint_state.name = joint_names
    rstate.joint_state.position = q.tolist()
    rstate.is_diff = True
    return rstate


def resample_trajectory(times: List[float], Q: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    times: list of cumulative seconds, len=N
    Q: [N, 7] joint positions at those times
    Returns:
      Qu: [T+1, 7] positions at uniform grid (0..t_end)
      tgrid: [T+1]
    """
    t_end = times[-1]
    if t_end < dt * 2:
        # too short; pad by duplicating last to allow at least one action
        t_end = max(t_end, dt * 2)
    T = int(np.ceil(t_end / dt))
    tgrid = np.linspace(0.0, t_end, T + 1)
    Q_interp = []
    for j in range(Q.shape[1]):
        Q_interp.append(np.interp(tgrid, times, Q[:, j]))
    Q_interp = np.stack(Q_interp, axis=1)
    return Q_interp, tgrid


def finite_diff(Q: np.ndarray, dt: float) -> np.ndarray:
    dq = (Q[1:] - Q[:-1]) / dt
    return dq

def finite_diff_uniform(Qu: np.ndarray, dt: float) -> np.ndarray:
    # Centered difference interior, forward/backward edges
    dq = np.zeros_like(Qu)
    dq[1:-1] = (Qu[2:] - Qu[:-2]) / (2.0*dt)
    dq[0]    = (Qu[1]  - Qu[0])   / dt
    dq[-1]   = (Qu[-1] - Qu[-2])  / dt
    return dq

def sample_pose_in_aabb() -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = BASE_LINK
    pose.pose.position.x = rnd(*WS_AABB["x"])
    pose.pose.position.y = rnd(*WS_AABB["y"])
    pose.pose.position.z = rnd(*WS_AABB["z"])
    # orientation: identity (EE pointing down for Panda default)
    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 1.0
    return pose

