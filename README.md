# Project Summary — Behavioral Cloning for Collision-Free Panda Arm Trajectories

This project trains a behavioral cloning agent to generate smooth, collision-free trajectories for the Franka Emika Panda arm in a cluttered environment. The agent learns directly from expert demonstrations generated using the MoveIt motion planning framework. The agent learns to predict joint-space motion increments that drive the arm toward a target pose without hitting obstacles. The panda arm is placed in a workspace populated with three randomly placed box obstacles. 

---

## Expert Rollout Generation

The `run_episode` function in ***data_gen.py*** handles the process of producing a single demonstration. First, it clears any previous obstacles and places three new boxes in the workspace. Each box’s position and size is randomized, but placement rules ensure that neither the starting position of the end-effector nor the target goal lies inside a box.

The start configuration is taken from the robot’s current joint positions and validated using MoveIt’s state validity service. A goal pose is then sampled within a bounded region of the workspace. The goal must be sufficiently far from the start to avoid trivial motions, and an inverse kinematics (IK) solution is computed using the start configuration as the seed. If either IK fails or the resulting configuration is invalid, the process retries with a new goal.

Once both start and goal configurations are found, a motion plan is requested from MoveIt. The resulting trajectory is interpolated to a uniform time step and checked for collisions at each intermediate waypoint. If the plan passes these checks, it is executed in simulation and resampled so that each step corresponds to the same fixed time interval. Joint-space increments (Δq) are computed by subtracting consecutive joint positions. These increments, along with the robot state, goal-relative pose, and obstacle descriptions, form the dataset entries.

---

## Observation and Action Definitions

**Observation** $o_t \in \mathbb{R}^{39}$:

$$
o_t = \big[ q_t \in \mathbb{R}^7,\ \dot{q}_t \in \mathbb{R}^7,\ \Delta x_t \in \mathbb{R}^3,\ \Delta r_t \in \mathbb{R}^4,\ b_t \in \mathbb{R}^{18} \big]
$$

- $q_t$: joint angles  
- $\dot{q}_t$: joint velocities  
- $\Delta x_t$: EE position error (goal − current)  
- $\Delta r_t$: EE orientation error (quaternion)  
- $b_t$: obstacle parameters — (center and size of 3 boxes)

---

**Action** $a_t \in \mathbb{R}^7$:

$$
a_t = \Delta q_t = q_{t+1} - q_t
$$

- Incremental change in joint angles per timestep

---

## Dataset

Each dataset contains:
- `obs`: an array of shape `(N, 39)` containing normalized observations.
- `acts`: an array of shape `(N, 7)` containing normalized actions (Δq).
- `dones`: boolean flags marking the ends of episodes.
- `episode_starts`: boolean flags marking the first step in each episode.

In addition, a `stats.json` file stores the feature means and standard deviations for observation normalization, as well as scaling factors for the actions. These statistics are used during both training and inference.

---
## Policy Architecture

The policy $\pi_\theta$ is a feedforward MLP mapping observations to joint increments:

$$
\pi_\theta: \mathbb{R}^{39} \to \mathbb{R}^7
$$

- **Input**: $o_t \in \mathbb{R}^{39}$ (normalized)  
- **Hidden layers**: 2 fully connected layers, each with 256 units and ReLU activation  
- **Output**: $\hat{a}_t \in [-1,1]^7$ (Tanh activation), representing scaled $\Delta q_t$  

