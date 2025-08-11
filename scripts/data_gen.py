#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_gen.py — Panda + MoveIt expert rollout generator (Behavior Cloning)
Author: you :)
Run:
  rosrun il_panda_bc data_gen.py --episodes 500 --dt 0.05

Outputs:
  ~/il_panda_bc/datasets/<timestamp>/
    - rollouts.npz  (obs, acts, dones, episode_starts)
    - stats.json    (feature mean/std and config)
"""
import os
import sys
import json
import time
import math
import argparse
import datetime as dt
from typing import List, Tuple

import numpy as np
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionFK, GetPositionFKRequest
from sensor_msgs.msg import JointState

# -----------------------------
# Config defaults
# -----------------------------
DEFAULT_DT = 0.05  # s
DEFAULT_EPISODES = 300
PLANNING_TIME = 2.0  # s
MAX_GOAL_RETRIES = 8
MIN_EE_DIST = 0.01   # m
BASE_LINK = "panda_link0"
EEF_LINK = "panda_link8"
ARM_GROUP = "panda_arm"

# Goal & obstacle sampling workspace (meters)
WS_AABB = dict(x=(0.2, 0.7), y=(-0.4, 0.4), z=(0.0, 0.6))
BOX_EDGE_RANGE = (0.04, 0.12)

# -----------------------------
# Helpers
# -----------------------------
def rnd(a, b): return a + np.random.rand() * (b - a)

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

def quat_to_np(q):
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

def quat_multiply(q1, q2):
    # (x, y, z, w)
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
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
    return math.sqrt(dx*dx + dy*dy + dz*dz)

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

# -----------------------------
# Core generator class
# -----------------------------
class PandaILDataGen:
    def __init__(self, args):
        np.random.seed(args.seed)
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("panda_il_data_gen", anonymous=True)

        self.robot = RobotCommander()
        self.group = MoveGroupCommander(ARM_GROUP)
        self.scene = PlanningSceneInterface(synchronous=True)

        self.group.set_planning_time(PLANNING_TIME)
        self.group.set_max_velocity_scaling_factor(0.6)
        self.group.set_max_acceleration_scaling_factor(0.6)

        self.joint_names = self.group.get_active_joints()  # 7 Panda joints
        if len(self.joint_names) < 7:
            rospy.logwarn("Unexpected number of joints in group; continuing anyway.")

        rospy.wait_for_service('/compute_ik')
        rospy.wait_for_service('/compute_fk')
        self.ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        self.fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)

        # Buffers
        self.obs_buf = []
        self.act_buf = []
        self.done_buf = []
        self.ep_start_buf = []

        # Obstacles cache per episode (centers+sizes)
        self.box_params = None

        # Output paths
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_dir = os.path.expanduser("~/ws_il/src/panda_il/datasets/" + ts)
        ensure_dir(base_dir)
        self.out_npz = os.path.join(base_dir, "rollouts.npz")
        self.out_stats = os.path.join(base_dir, "stats.json")
        self.cfg = dict(
            dt=args.dt,
            episodes=args.episodes,
            ws_aabb=WS_AABB,
            box_edge_range=BOX_EDGE_RANGE,
            action_space="joint_delta",
            obs_spec=dict(q=7, qdot=7, ee_err=7, obstacles=18),
            joint_names=self.joint_names,
            base_link=BASE_LINK, eef_link=EEF_LINK
        )
        rospy.loginfo(f"Saving to: {base_dir}")

    # -------- Obstacles --------
    def randomize_three_boxes(self):
        # Remove old
        for i in range(3):
            self.scene.remove_world_object(f"box_{i}")
        rospy.sleep(0.2)

        params = []
        for i in range(3):
            name = f"box_{i}"
            pose = PoseStamped()
            pose.header.frame_id = BASE_LINK
            pose.pose.position.x = rnd(*WS_AABB["x"])
            pose.pose.position.y = rnd(*WS_AABB["y"])
            pose.pose.position.z = rnd(*WS_AABB["z"])
            pose.pose.orientation.w = 1.0
            size = (
                rnd(*BOX_EDGE_RANGE),
                rnd(*BOX_EDGE_RANGE),
                rnd(*BOX_EDGE_RANGE),
            )
            self.scene.add_box(name, pose, size)
            params.extend([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
                           size[0], size[1], size[2]])
        rospy.sleep(0.5)  # ensure update
        self.box_params = np.array(params, dtype=np.float64)  # [18]

    # -------- IK / FK --------
    def compute_ik(self, target_pose: PoseStamped) -> Tuple[bool, np.ndarray]:
        req = GetPositionIKRequest()
        req.ik_request.group_name = ARM_GROUP
        req.ik_request.pose_stamped = target_pose
        req.ik_request.ik_link_name = EEF_LINK
        req.ik_request.timeout = rospy.Duration(0.5)
        # Seed with current state
        req.ik_request.robot_state = self.robot.get_current_state()
        try:
            res = self.ik_srv(req)
            if res.error_code.val > 0:
                q = np.array(res.solution.joint_state.position, dtype=np.float64)
                # Map to active joints order
                name_to_pos = dict(zip(res.solution.joint_state.name, q))
                q_out = np.array([name_to_pos[nm] for nm in self.joint_names], dtype=np.float64)
                return True, q_out
            else:
                return False, None
        except rospy.ServiceException as e:
            rospy.logwarn(f"IK service exception: {e}")
            return False, None

    def fk_for_q(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (pos[3], quat[4]) of EEF for given joint vector q.
        """
        rstate = msg_robot_state_from_q(self.joint_names, q)
        req = GetPositionFKRequest()
        req.fk_link_names = [EEF_LINK]
        req.robot_state = rstate
        req.header.frame_id = BASE_LINK
        try:
            res = self.fk_srv(req)
            if not res.pose_stamped:
                raise RuntimeError("FK returned no poses")
            p = res.pose_stamped[0].pose.position
            o = res.pose_stamped[0].pose.orientation
            pos = np.array([p.x, p.y, p.z], dtype=np.float64)
            quat = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)
            # normalize quat just in case
            quat = quat / np.linalg.norm(quat)
            return pos, quat
        except Exception as e:
            rospy.logwarn(f"FK service exception: {e}")
            # fallback zeros
            return np.zeros(3), np.array([0,0,0,1], dtype=np.float64)

    # -------- Planning --------
    def plan_from_to(self, q_start: np.ndarray, q_goal: np.ndarray):
        # Set start state
        start_state = msg_robot_state_from_q(self.joint_names, q_start)
        self.group.set_start_state(start_state)
        self.group.set_joint_value_target(q_goal.tolist())
        success, plan, _, _ = self.group.plan()
        if not success or len(plan.joint_trajectory.points) < 2:
            return None
        return plan

    # -------- Episode --------
    def run_episode(self, dt_step: float) -> bool:
        """
        Returns True if a rollout was added, else False.
        """
        # 1) Randomize obstacles
        self.randomize_three_boxes()

        # 2) Sample a valid start & goal via IK on random poses
        # Try a few times for start
        for _ in range(MAX_GOAL_RETRIES):
            start_pose = sample_pose_in_aabb()
            ok, q_start = self.compute_ik(start_pose)
            if ok: break
        else:
            return False

        # Goal must be sufficiently separated
        for _ in range(MAX_GOAL_RETRIES):
            goal_pose = sample_pose_in_aabb()
            if pose_distance(start_pose, goal_pose) < MIN_EE_DIST:
                continue
            ok, q_goal = self.compute_ik(goal_pose)
            if ok: break
        else:
            return False

        # 3) Plan expert trajectory
        plan = self.plan_from_to(q_start, q_goal)
        if plan is None:
            return 
        
        traj_dur = plan.joint_trajectory.points[-1].time_from_start.to_sec()
        rospy.sleep(traj_dur + 1)

        ### Sanity check
        jt = plan.joint_trajectory
        last_q = np.array(jt.points[-1].positions, dtype=np.float64)

        # FK of final planned joint state
        final_pos, final_quat = self.fk_for_q(last_q)

        # FK of the intended goal joint state (your "canonical" target)
        goal_pos, goal_quat = self.fk_for_q(q_goal)

        # Position error (meters)
        pos_err = np.linalg.norm(final_pos - goal_pos)

        # Orientation error (radians): 2*acos(|dot(q_final, q_goal)|)
        dq = abs(np.dot(final_quat, goal_quat))
        dq = max(min(dq, 1.0), -1.0)
        ang_err = 2.0 * math.acos(dq)

        rospy.logwarn(f"EE position error: {pos_err*1000:.2f} mm, orientation error: {ang_err*180/math.pi:.2f} deg")
        ###

        # Extract joint trajectory
        jt = plan.joint_trajectory
        times = [pt.time_from_start.to_sec() for pt in jt.points]
        Q = np.array([pt.positions for pt in jt.points], dtype=np.float64)  # [N, 7]

        # 4) Resample to uniform dt, compute actions Δq
        Qu, tgrid = resample_trajectory(times, Q, dt_step)  # [T+1,7]
        if Qu.shape[0] < 2:
            return False
        A = Qu[1:] - Qu[:-1]        # [T,7]
        Qaligned = Qu[:-1]          # [T,7]

        # 5) Build observations
        # q, qdot
        qdot = finite_diff(Qu, dt_step)  # [T,7]

        # FK for each q_t to compute EE pose (pos, quat)
        ee_pos = np.zeros((Qaligned.shape[0], 3), dtype=np.float64)
        ee_quat = np.zeros((Qaligned.shape[0], 4), dtype=np.float64)
        for i in range(Qaligned.shape[0]):
            p, q = self.fk_for_q(Qaligned[i])
            ee_pos[i] = p
            ee_quat[i] = q

        # Goal pose arrays (constant across the episode)
        gpos, gquat = self.fk_for_q(q_goal)  # use FK of goal_q to get canonical goal pose
        gpos = gpos.reshape(1,3)
        gquat = gquat.reshape(1,4)

        ee_pos_err = (gpos - ee_pos)             # [T,3]
        # quaternion error: q_err = q_goal * q_curr^{-1}
        q_err = np.zeros_like(ee_quat)
        for i in range(ee_quat.shape[0]):
            q_err[i] = quat_error(ee_quat[i], gquat[0])

        # Obstacles features (repeat constants): [18] -> [T,18]
        obb = np.repeat(self.box_params.reshape(1, -1), Qaligned.shape[0], axis=0)

        # Observation = [q, qdot, ee_pos_err(3) + q_err(4), obstacles(18)] = 39 dims
        obs = np.concatenate([Qaligned, qdot, ee_pos_err, q_err, obb], axis=1)

        # dones & episode_starts
        T = obs.shape[0]
        dones = np.zeros((T,), dtype=np.bool_)
        dones[-1] = True
        ep_starts = np.zeros((T,), dtype=np.bool_)
        ep_starts[0] = True

        # 6) Append to buffers
        self.obs_buf.append(obs)
        self.act_buf.append(A)
        self.done_buf.append(dones)
        self.ep_start_buf.append(ep_starts)
        return True

    # -------- Save --------
    def save_dataset(self):
        if len(self.obs_buf) == 0:
            rospy.logwarn("No data to save.")
            return
        obs = np.concatenate(self.obs_buf, axis=0)
        acts = np.concatenate(self.act_buf, axis=0)
        dones = np.concatenate(self.done_buf, axis=0)
        ep_starts = np.concatenate(self.ep_start_buf, axis=0)

        # Stats for normalization
        mean = obs.mean(axis=0).tolist()
        std = obs.std(axis=0)
        std[std < 1e-6] = 1e-6
        std = std.tolist()

        np.savez_compressed(self.out_npz,
                            obs=obs.astype(np.float32),
                            acts=acts.astype(np.float32),
                            dones=dones,
                            episode_starts=ep_starts)
        meta = dict(
            shape=dict(obs=list(obs.shape), acts=list(acts.shape)),
            mean=mean, std=std,
            config=self.cfg
        )
        with open(self.out_stats, "w") as f:
            json.dump(meta, f, indent=2)
        rospy.loginfo(f"Saved dataset to {self.out_npz} and stats to {self.out_stats}")

# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                    help="Number of episodes (plans) to generate")
    ap.add_argument("--dt", type=float, default=DEFAULT_DT,
                    help="Uniform sampling step in seconds")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    return ap.parse_args(rospy.myargv(argv=sys.argv)[1:])

def main():
    args = parse_args()
    gen = PandaILDataGen(args)

    successes = 0
    attempts = 0
    target = args.episodes

    rospy.loginfo(f"Starting data generation: target episodes = {target}, dt={args.dt}")
    while not rospy.is_shutdown() and successes < target:
        attempts += 1
        ok = gen.run_episode(args.dt)
        if ok:
            successes += 1
            if successes % 1 == 0:
                rospy.loginfo(f"Collected {successes}/{target} episodes...")
        else:
            # soft retry; adjust workspace/obstacles if too many fails
            pass

        time.sleep(2)

    gen.save_dataset()
    rospy.loginfo("Done.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
