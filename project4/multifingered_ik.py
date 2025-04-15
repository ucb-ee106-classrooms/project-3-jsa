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

        numbodies = len(body_ids)

        for ind in range(numbodies):
            # consider each body separately
            body_id = body_ids[ind]
            mj.mj_forward(self.model, self.data)
            
            current_pose = self.data.body(body_id).xpos
            target_pose = target_pose[ind]

            # only currently considering position error, not quat error yet
            error = np.subtract(target_pose, current_pose)

            while (np.linalg.norm(error) >= self.tol):
                # calc jacobian
                mj.mj_jac(self.model, self.data, self.jacp, self.jacr, target_pose, body_id)

                # calc grad
                grad = self.alpha * self.jacp.T @ error

                # calc step

    
