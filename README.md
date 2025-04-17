# Alma Robot Bipedal Locomotion Enhancement Using Reinforcement Learning

This project builds on the open-source framework provided by the [legged_gym repository](https://github.com/leggedrobotics/legged_gym), developed by the Robotic Systems Lab at ETH Zurich.

This project extends the Alma robot’s bipedal locomotion capabilities by leveraging its **manipulator as a stabilizing support leg**, freeing the front legs for advanced manipulation. To reach complex control behaviors like **goal-directed walking**, we implemented **progressive learning** within a **multi-task reinforcement learning** framework.

## 📌 Motivation

Quadrupedal robots like ANYmal are excellent for locomotion, but their manipulators are usually unused during motion. By using the manipulator as a stabilizing leg, Alma transitions to a **tripodal configuration** that:
- Maintains balance during bipedal motion.
- Frees two legs for manipulation.
- Improves payload handling during movement.

To train these behaviors, we needed:
- Modular task learning (linear velocity, angular velocity, goal reaching).
- A mechanism to schedule learning focus over time.
- Scalable reward computation for thousands of environments.

---

## 🧠 Key Contributions

1. **Multi-task reward architecture** using `ProgressiveRewardManager` with:
   - Task-specific rewards (e.g., `linear_velocity_tracking`, `angular_tracking`).
   - Shared shaping rewards (e.g., orientation, torque, collisions).
   - Task encoder in observations (optional).

2. **Progressive learning scheduler**:
   - Tracks learning progress of each task.
   - Dynamically updates training probability distribution.
   - Balances exploration (ε-greedy sampling) and exploitation.

3. **Dual-policy controller vs. Multi-task policy** comparison.

4. Addition of a wheel at the end of the manipulator to facilitate rotation and translation.
---

## 🔍 Results Overview

| Task                         | Method              | Performance                                |
|-----------------------------|---------------------|---------------------------------------------|
| Linear velocity tracking     | Multi-task & Dual   | Accurate with jump-based motion             |
| Angular velocity tracking    | Multi-task & Dual   | Smooth and quick with manipulator rotation  |
| Goal-reaching (x, y)         | Multi-task          | Unstable with circular trajectories         |
| Goal-reaching (x, y)         | Dual-policy         | Fast, direct, and realistic path            |
| Payload testing              | N/A                 | Up to **25 kg** carried stably              |

---

## 📊 Important Figures and videos

1. **Progressive Learning Results**
![Linear Velocity Tracking](figures/progressive_learning_1.jpg)
![Linear Velocity Tracking](figures/progressive_learning_2.jpg)
We can see that in the first iterations, the probability of sampling the goal task is low compared to the linear and angular 
velocity tracking, meaning that the learning of these two tasks is progressing faster ( high reward derivative).
However, after 4000 iterations, the progress of the goal task becomes higher and thus more environments are use to train the goal task.

2. **Progressive Learning VS Dual-Policy Videos**  
   <table>
  <tr>
    <td align="center">
      <a href="https://drive.google.com/drive/u/0/folders/1JkkqyRg4MWzeSpwkUdAH1vAVYwSQaRM1">
        <b>Progressive Learning Policy</b>
      </a>
    </td>
    <td align="center">
      <a href="https://drive.google.com/drive/u/0/folders/1JkkqyRg4MWzeSpwkUdAH1vAVYwSQaRM1">
        <b>Controller Using Separate Policies </b>
      </a>
    </td>
  </tr>
</table>

3. **Figure 9 & 10: Task Probability Distributions**  
   - 📍 Place under `Results → Multi-task Framework`  
   - Demonstrates how task probabilities evolve through training.

4. **Figure 11: Goal Comparison Setup**  
   - 📍 Place under `Results → Comparison between the two methods`  
   - Compares trajectories of multi-task vs dual-policy controller.

5. **Figure 12: Payload Torque Comparison**  
   - 📍 Place under `Results → Payload`  
   - Visualizes the advantage of manipulator support over ANYmal biped mode.

---

## 🗂️ Main Code Contributions

The core implementation of progressive learning and task control was made in the following locations:

### 1. `legged_gym/rewards/reward_manager_progressive_learning.py`
- Implements per-task reward logic with masking.
- Supports both shared and task-specific reward functions.
- Tracks episode-level statistics for logging and curriculum.

### 2. `legged_gym/envs/locomotion/alma_bipedal/`
- Samples and assigns new tasks per environment at reset using ε-greedy and task distributions.
- Updates reward manager dynamically to reflect current tasks.
- Addition of environment to train payload carrying


### 3. Modification of the Reinforcement Learning pipeline to be able to include the task codes and ids into the model, as well as to train different tasks at the same time.

---


## 📽️ Demonstration Videos

- [🦘 Linear Velocity Tracking](https://drive.google.com/file/d/1bIIWS_wGmsQUNnFIu5DeubK2D0iKDmMg/view?usp=sharing)  
- [↪️ Angular Velocity Tracking](https://drive.google.com/file/d/1vVctATsAsQNo9GKdM9feyHwfrl_I_gd3/view?usp=sharing)  
- [📚 Progressive Learning Evolution](https://drive.google.com/file/d/1EqcPE1w4d-CgCsMRLE3VwVi1fI80E5st/view?usp=sharing)  
- [🎮 Dual Policy Goal Controller](https://drive.google.com/file/d/1JeNCs7MOXyBJL6HjFkUrCkdEJaCItwsP/view?usp=sharing)

---


## 🧾 Main References

- Hwangbo et al., *Learning Agile and Dynamic Motor Skills for Legged Robots*
- Colas et al., *CURIOUS: Intrinsically Motivated Multi-goal RL*
- Schulman et al., *Proximal Policy Optimization Algorithms*

## ✅ Next Steps

- Integrate real-time perception (vision or lidar) for real-world deployment.
- Train on uneven terrain and staircases.
- Experiment with richer task encoding and transformer-based policies.
