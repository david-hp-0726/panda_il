#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, sys, time, math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn

import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from moveit_commander import RobotCommander, MoveGroupCommander, PlanningSceneInterface
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionFK, GetPositionFKRequest, GetStateValidity, GetStateValidityRequest
from moveit_msgs.msg import RobotState

# ---- use your helpers instead of re-defining ----
from helpers import (
    msg_robot_state_from_q,       # (joint_names, q) -> RobotState
    sample_pose_in_aabb,          # returns PoseStamped in BASE_LINK
    quat_error,                   # q_err = q_goal * conj(q_curr), [x,y,z,w], unit, w>=0
    point_in_candidate_box,       # (p, cx,cy,cz,sx,sy,sz, margin) -> bool
)

# -------- constants (match your gen script) --------
ARM_GROUP  = "panda_arm"
EEF_LINK   = "panda_link8"
BASE_LINK  = "panda_link0"
WS_AABB    = dict(x=(0.2, 0.7), y=(-0.4, 0.4), z=(0.0, 0.6))
# BOX_EDGE_RANGE = (0.04, 0.12)
BOX_EDGE_RANGE = (0.001, 0.001)
PLANNING_TIME = 2.0
MAX_GOAL_RETRIES = 16
MIN_EE_DIST = 0.01     # m
BOX_MARGIN = 0.02      # m
POS_TOL = 0.01         # m
ANG_TOL = 0.087        # rad (~5°)
DT_STEP = 0.05
MAX_STEPS = 200

# -------- small utils --------
def rnd(a, b): return np.random.uniform(a, b)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class BCController:
    def __init__(self, policy_dir: Path):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("panda_bc_rollout", anonymous=True)

        self.robot = RobotCommander()
        self.group = MoveGroupCommander(ARM_GROUP)
        self.scene = PlanningSceneInterface(synchronous=True)
        self.group.set_planning_time(PLANNING_TIME)
        self.group.set_max_velocity_scaling_factor(0.6)
        self.group.set_max_acceleration_scaling_factor(0.6)
        self.joint_names = self.group.get_active_joints()

        rospy.wait_for_service("/compute_ik")
        rospy.wait_for_service("/compute_fk")
        rospy.wait_for_service("/check_state_validity")
        self.ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)
        self.fk_srv = rospy.ServiceProxy("/compute_fk", GetPositionFK)
        self.state_valid_srv = rospy.ServiceProxy("/check_state_validity", GetStateValidity)

        # load policy + scalers
        with open(policy_dir / "scalers.json","r") as f:
            sc = json.load(f)
        self.obs_mean = torch.tensor(sc["obs_mean"], dtype=torch.float32)
        self.obs_std  = torch.tensor(sc["obs_std"],  dtype=torch.float32)
        self.a_scale  = torch.tensor(sc["act_scale"], dtype=torch.float32)
        in_dim  = int(sc.get("obs_dim", len(self.obs_mean)))
        out_dim = int(sc.get("act_dim", len(self.a_scale)))
        hidden  = tuple(sc.get("hidden", [256,256]))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(in_dim, out_dim, hidden=hidden).to(self.device)
        sd = torch.load(policy_dir / "bc_policy_best.pt", map_location=self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        self.box_params = None

        self.safe_q_default = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785] 

    # ---- IK/FK/validity ----
    def compute_ik(self, goal_pose: PoseStamped, seed: Optional[np.ndarray]) -> Tuple[bool, Optional[np.ndarray]]:
        req = GetPositionIKRequest()
        req.ik_request.group_name = ARM_GROUP
        req.ik_request.pose_stamped = goal_pose
        req.ik_request.ik_link_name = EEF_LINK
        req.ik_request.timeout = rospy.Duration(0.7)
        req.ik_request.robot_state = (msg_robot_state_from_q(self.joint_names, seed)
                                      if seed is not None and len(seed)==len(self.joint_names)
                                      else self.robot.get_current_state())
        res = self.ik_srv(req)
        if res.error_code.val <= 0 or not res.solution.joint_state.name:
            return False, None
        name_to_pos = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        q = np.array([name_to_pos[n] for n in self.joint_names], dtype=np.float64)
        return True, q

    def fk_for_q(self, q: np.ndarray):
        req = GetPositionFKRequest()
        req.header.frame_id = BASE_LINK
        req.fk_link_names = [EEF_LINK]
        req.robot_state = msg_robot_state_from_q(self.joint_names, q)
        res = self.fk_srv(req)
        ps = res.pose_stamped[0].pose
        p = np.array([ps.position.x, ps.position.y, ps.position.z], dtype=np.float64)
        qxyzw = np.array([ps.orientation.x, ps.orientation.y, ps.orientation.z, ps.orientation.w], dtype=np.float64)
        qxyzw /= (np.linalg.norm(qxyzw) + 1e-12)
        return p, qxyzw

    def is_state_valid(self, q: np.ndarray) -> bool:
        req = GetStateValidityRequest()
        req.group_name = ARM_GROUP
        req.robot_state = msg_robot_state_from_q(self.joint_names, q)
        return bool(self.state_valid_srv(req).valid)

    # ---- boxes ----
    def remove_boxes(self):
        for i in range(3): self.scene.remove_world_object(f"box_{i}")
        rospy.sleep(0.2)

    def randomize_three_boxes_avoiding(self, forbid_pts: List[np.ndarray], margin=BOX_MARGIN, max_tries=200):
        self.remove_boxes()
        params = []
        for i in range(3):
            ok = False
            name = f"box_{i}"
            for _ in range(max_tries):
                cx = rnd(*WS_AABB["x"]); cy = rnd(*WS_AABB["y"]); cz = rnd(*WS_AABB["z"])
                sx = rnd(*BOX_EDGE_RANGE); sy = rnd(*BOX_EDGE_RANGE); sz = rnd(*BOX_EDGE_RANGE)
                if any(point_in_candidate_box(p, cx,cy,cz,sx,sy,sz, margin) for p in forbid_pts):
                    continue
                ps = PoseStamped(); ps.header.frame_id = BASE_LINK
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = cx, cy, cz
                ps.pose.orientation.w = 1.0
                self.scene.add_box(name, ps, (sx,sy,sz))
                params.extend([cx,cy,cz,sx,sy,sz]); ok = True; break
            if not ok: return False
        rospy.sleep(0.4)
        self.box_params = np.array(params, dtype=np.float64)  # [18]
        return True

    # ---- policy ----
    @torch.no_grad()
    def policy_step(self, obs_row: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs_row, dtype=torch.float32, device=self.device)
        x = (x - self.obs_mean.to(self.device)) / self.obs_std.to(self.device)   # normalize obs
        a_scaled = self.model(x.unsqueeze(0)).squeeze(0)                          # [-1,1]
        dq = (a_scaled * self.a_scale.to(self.device)).cpu().numpy()              # unscale Δq
        return dq
    
    def reset_panda(self):
        g = self.group
        g.stop(); g.clear_pose_targets(); g.set_start_state_to_current_state()
        # 1) Try a named target if available
        for name in ("ready", "home"):
            try:
                if name in g.get_named_targets():
                    g.set_named_target(name)
                    if g.go(wait=True): 
                        g.stop(); g.clear_pose_targets(); return True
            except Exception:
                pass
        # 2) Fallback to a known-good joint vector
        try:
            g.set_joint_value_target(self.safe_q_default)
            ok = g.go(wait=True)
            g.stop(); g.clear_pose_targets()
            return bool(ok)
        except Exception:
            return False

    # ---- rollout once ----
    def rollout_once(self) -> bool:
        # reset to a valid pose
        if not self.reset_panda():
            rospy.logwarn("Reset failed")
            return False

        # start
        q = np.array(self.group.get_current_joint_values(), dtype=np.float64)
        if not self.is_state_valid(q):
            rospy.logwarn("Invalid start"); return False
        start_p, _ = self.fk_for_q(q)

        # goal via IK (seeded with q)
        for _ in range(MAX_GOAL_RETRIES):
            goal_pose = sample_pose_in_aabb()
            gp = np.array([goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z])
            if np.linalg.norm(gp - start_p) < MIN_EE_DIST: continue
            ok, q_goal = self.compute_ik(goal_pose, seed=q)
            if ok: break
        else:
            rospy.logwarn("IK failed"); return False
        gpos, gquat = self.fk_for_q(q_goal)

        # Sanity check
        rospy.loginfo(f"start_pos: {start_p}, goal_pos: {gpos}")

        # boxes after goal; avoid endpoints
        for _ in range(5):
            if self.randomize_three_boxes_avoiding([start_p, gpos]): break
        else:
            rospy.logwarn("Box placement failed"); return False

        # incremental plan→execute loop
        prev_q = q.copy()
        for step in range(MAX_STEPS):
            qdot = (q - prev_q) / DT_STEP
            prev_q = q.copy()

            ee_pos, ee_quat = self.fk_for_q(q)
            ee_pos_err = (gpos - ee_pos)             # [3]
            q_err = quat_error(gquat, ee_quat)       # [4]
            obb = self.box_params                    # [18]

            # [ q (7), qdot (7), (gpos - ee_pos) (3), quat_error(gquat, ee_quat) (4), boxes (18) ] = [39]
            obs_row = np.concatenate([q, qdot, ee_pos_err, q_err, obb], axis=0)  
            dq = self.policy_step(obs_row)
            q_target = q + dq

            if not self.is_state_valid(q_target):
                q_target = q + 0.5*dq
                if not self.is_state_valid(q_target):
                    rospy.logwarn("Δq invalid"); return False

            self.group.set_start_state_to_current_state()
            self.group.set_joint_value_target(q_target.tolist())
            plan = self.group.plan()
            traj = plan[1] if isinstance(plan, tuple) else plan
            if traj is None or len(traj.joint_trajectory.points) < 1:
                rospy.logwarn("Planner failed"); return False
            if not self.group.execute(traj, wait=True):
                rospy.logwarn("Execution failed"); return False

            q = np.array(self.group.get_current_joint_values(), dtype=np.float64)

            # success check
            p_now, q_now = self.fk_for_q(q)
            pos_err = np.linalg.norm(p_now - gpos)
            dqdot = abs(np.dot(q_now, gquat)); dqdot = max(min(dqdot,1.0), -1.0)
            ang_err = 2.0 * math.acos(dqdot)
            if pos_err <= POS_TOL and ang_err <= ANG_TOL:
                rospy.loginfo(f"[SUCCESS] steps={step+1} pos_err={pos_err:.4f}m ang_err={ang_err:.3f}rad")
                return True

            rospy.loginfo(f"Successfully moved to {p_now}, pos_err = {pos_err}")
            # optional pacing; planning dominates anyway
            time.sleep(0.0)

        rospy.logwarn("Max steps reached")
        return False

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_dir", default="./bc_out", help="Folder with bc_policy_best.pt & scalers.json")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args(rospy.myargv(sys.argv)[1:])

    ctrl = BCController(Path(args.policy_dir))
    succ = 0
    for ep in range(args.episodes):
        rospy.loginfo(f"=== Episode {ep+1}/{args.episodes} ===")
        succ += int(ctrl.rollout_once())
    rospy.loginfo(f"Done. Successes: {succ}/{args.episodes}")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
