import numpy as np
import mujoco as mj
import dm_control
from utils import *

# Inspired by: https://alefram.github.io/posts/Basic-inverse-kinematics-in-Mujoco
class LevenbergMarquardtIK:
    
    def __init__(self, model: dm_control.mujoco.wrapper.core.MjModel, 
                 data: dm_control.mujoco.wrapper.core.MjData, 
                 step_size: int, 
                 tol: int, 
                 alpha: int, 
                 jacp: np.array, 
                 jacr: np.array, 
                 damping: int, 
                 max_steps: int, 
                 physics: dm_control.mjcf.physics.Physics):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.damping = damping
        self.max_steps = max_steps
        self.physics = physics
    
    def calculate(self, target_positions: np.array, 
                  target_orientations: np.array, 
                  body_ids: list, 
                  evaluating=False):
        """
        Calculates joint angles given target positions and orientations by solving inverse kinematics.
        Uses the Levenbeg-Marquardt method for nonlinear optimization. 

        Parameters
        ----------
        target_positions: 3xn np.array containing n desired x,y,z positions
        target_orientations: 4xn np.array containing n desired quaternion orientations
        body_ids: list of length n containing the ids for every body

        Returns
        -------
        new_qpos: np.array of size self.physics.data.qpos containing desired positions in joint space

        Tips: 
            -To access the body id you can use: self.model.body([insert name of body]).id 
            -You should consider using clip_to_valid_state in utils.py to ensure that joint poisitons
            are possible 
        """
        #YOUR CODE HERE
        for i in range(self.max_steps):
            print("iteration:", i)
            J_blocks, error_blocks = self.ContructBlocks(target_positions, body_ids)

            # Stack Jacobians and errors
            J_full = np.vstack(J_blocks)     # shape: (3*num_fingers, nq)
            error_full = np.concatenate(error_blocks)  # shape: (3*num_fingers,)

            if np.linalg.norm(error_full) < self.tol:
                break

            # Solve damped least squares
            JTJ = J_full.T @ J_full
            delta_q = np.linalg.solve(JTJ + self.damping * np.eye(JTJ.shape[0]), J_full.T @ error_full)

            # Apply update
            self.data.qpos[1:] += self.step_size * delta_q
            self.data.qpos = clip_to_valid_state(self.physics, self.data.qpos)
            mj.mj_forward(self.physics.model.ptr, self.physics.data.ptr)

        return np.copy(self.data.qpos)
    
    def ContructBlocks(self, target_positions, body_ids):
        J_blocks = []
        error_blocks = []

        for i, body_id in enumerate(body_ids):
            # Forward kinematics
            mj.mj_forward(self.physics.model.ptr, self.physics.data.ptr)

            id_num = self.model.body(body_id).id
            # zero the values to prevent accumulation 
            self.jacp[:] = 0
            self.jacr[:] = 0
            
            # Compute Jacobian
            mj.mj_jacBodyCom(self.physics.model.ptr, self.physics.data.ptr, self.jacp, self.jacr, id_num)
            J_blocks.append(np.copy(self.jacp))  # shape: (3, nq)

            # Compute error
            current_pos = np.copy(self.data.body(body_id).xpos)
            error = target_positions[i] - current_pos
            error_blocks.append(error)

        return J_blocks, error_blocks

    
