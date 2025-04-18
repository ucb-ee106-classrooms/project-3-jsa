import numpy as np
import dm_control
import mujoco as mj

class AllegroHandEnv:
    def __init__(self, physics: dm_control.mjcf.physics.Physics, 
                 q_h_slice: slice, 
                 object_name: str, 
                 num_fingers=4):
        self.physics = physics
        self.q_h_slice = q_h_slice
        self.num_fingers = num_fingers
        self.object_name = object_name

    def set_configuration(self, q_h: np.array):
        self.physics.data.qpos[self.q_h_slice] = q_h
        self.physics.forward()

    def get_contact_positions(self, body_names: list[str]):
        """
        Input: list of the names in the XML of the bodies that are in contact
        Returns: (num_contacts x 3) np.array containing 
        finger positions in workspace coordinates
        """
        #YOUR CODE HERE
        pos_list = []
        name_set = set(body_names)

        ncon = self.physics.data.ncon
        for i in range(ncon):
            c = self.physics.data.contact[i]

            # Geom indices → body indices → names
            bid1 = self.physics.model.geom_bodyid[c.geom1]
            bid2 = self.physics.model.geom_bodyid[c.geom2]
            bname1 = self.physics.model.body(bid1).name
            bname2 = self.physics.model.body(bid2).name

            if (bname1 in name_set) or (bname2 in name_set):
                pos_list.append(c.pos.copy())

        if len(pos_list) == 0:
            return np.zeros((0, 3))
        # Remove duplicates (within small tolerance)
        return np.unique(np.round(pos_list, 6), axis=0)

    def get_contact_normals(self, object_name: str = None): 
        """
        Input: contact data structure that contains MuJoCo contact information
        Returns the normal vector for each geom that's in contact with the ball
        
        Tip: 
            -See information about the mjContact_ struct here: https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjcontact
            -Get normals for all the geoms in contact with the ball, not just the fingertips
        """
        #YOUR CODE HERE
        if object_name is None:
            object_name = self.object_name

        normals = []
        ncon = self.physics.data.ncon
        for i in range(ncon):
            c = self.physics.data.contact[i]

            bid1 = self.physics.model.geom_bodyid[c.geom1]
            bid2 = self.physics.model.geom_bodyid[c.geom2]
            bname1 = self.physics.model.body(bid1).name
            bname2 = self.physics.model.body(bid2).name

            if (bname1 == object_name) or (bname2 == object_name):
                n = c.frame[6:9].copy()    # normal vector
                normals.append(n)

        if len(normals) == 0:
            return np.zeros((0, 3))
        return np.stack(normals, axis=0)
    
    def get_contacts_bool(self, body_names: list[str]) -> list[bool]:
        """
        True/False list, length == len(body_names), indicating whether each
        fingertip body is currently in contact with *anything*.
        """
        flags = [False] * len(body_names)
        name_to_idx = {n: i for i, n in enumerate(body_names)}

        ncon = self.physics.data.ncon
        for i in range(ncon):
            c = self.physics.data.contact[i]
            bid1 = self.physics.model.geom_bodyid[c.geom1]
            bid2 = self.physics.model.geom_bodyid[c.geom2]
            bname1 = self.physics.model.body(bid1).name
            bname2 = self.physics.model.body(bid2).name
            if bname1 in name_to_idx:
                flags[name_to_idx[bname1]] = True
            if bname2 in name_to_idx:
                flags[name_to_idx[bname2]] = True
        return flags
    
    def object_center(self) -> np.ndarray:
        try:
            return self.physics.named.data.xpos[self.object_name].copy()
        except KeyError:
            # if `object_name` is a site:
            return self.physics.named.data.xpos[self.object_name + '_site'].copy()
    


class AllegroHandEnvSphere(AllegroHandEnv):
    def __init__(self, physics: dm_control.mjcf.physics.Physics, 
                 sphere_center: int, 
                 sphere_radius: int, 
                 q_h_slice: slice, 
                 object_name: str):
        super().__init__(physics, q_h_slice, object_name)
        self.physics = physics
        self.sphere_center = sphere_center
        self.sphere_radius = sphere_radius
        self.q_h_slice = q_h_slice
        self.num_fingers = 4
    
    def sphere_surface_distance(self, pos: np.array, center: np.array, radius: int):
        """
        Returns the distance from pos to the surface of a sphere with a specified
        radius and center
        """
        d = np.linalg.norm(pos - center) - radius
        return d
    
    def fingertip_distances(self, finger_names: list[str]) -> np.ndarray:
        """
        Returns an array (len(finger_names),) of signed distances from each
        fingertip to the sphere surface.
        """
        tip_pos = self.get_contact_positions(finger_names)  # or use xpos lookup
        if len(tip_pos) == 0:
            # if no contacts yet, fall back to site positions
            tip_pos = np.vstack([
                self.physics.named.data.xpos[name].copy()
                for name in finger_names
            ])
        d = [self.sphere_surface_distance(p, self.sphere_center,
                                          self.sphere_radius) for p in tip_pos]
        return np.asarray(d)
    
    def contact_pos_normals(self, finger_names: list[str]):
        """
        Returns two arrays:
            positions : (#contacts, 3)
            normals   : (#contacts, 3)
        Only contacts involving *either* the sphere or the fingertips listed.
        """
        pos_all   = []
        normal_all = []

        ncon = self.physics.data.ncon
        for i in range(ncon):
            c = self.physics.data.contact[i]
            bid1 = self.physics.model.geom_bodyid[c.geom1]
            bid2 = self.physics.model.geom_bodyid[c.geom2]
            bname1 = self.physics.model.body(bid1).name
            bname2 = self.physics.model.body(bid2).name

            if (bname1 in finger_names or bname2 in finger_names or
                bname1 == self.object_name or bname2 == self.object_name):
                pos_all.append(c.pos.copy())
                normal_all.append(c.frame[6:9].copy())

        if len(pos_all) == 0:
            return (np.zeros((0, 3)), np.zeros((0, 3)))
        return (np.stack(pos_all,   axis=0),
                np.stack(normal_all, axis=0))
    