import numpy as np
from scipy.optimize import linprog, minimize
import AllegroHandEnv
import dm_control
import mujoco as mj
import grasp_synthesis
import types

"""
Note: this code gives a suggested structure for implementing grasp synthesis.
You may decide to follow it or not. 
"""

def synthesize_grasp(env: grasp_synthesis.AllegroHandEnv, 
                        q_h_init: np.array,
                        fingertip_names: list[str], 
                        max_iters=1000, 
                        lr=0.1, 
                        grad_eps: float = 1e-3,
                        beta: float = 10.0,
                        mu: float = 0.5,
                        num_cone_vec: int = 4,
                        q_plus_thresh: float = 1e-3):
    """
    Given an initial hand joint configuration, q_h_init, return adjusted joint angles that are touching
    the object and approximate force closure. This is algorithm 1 in the project specification.

    Parameters
    ----------
    env: AllegroHandEnv instance (can use to access physics)
    q_h_init: array of joint positions for the hand
    max_iters: maximum number of iterations for the optimization
    lr: learning rate for the gradient step

    Output
    ------
    New joint angles after contact and force closure adjustment
    """
    #YOUR CODE HERE
    q_h = q_h_init.copy()
    loss = np.array([])
    in_contact = False

    for it in range(max_iters):
        if all(env.get_contacts_bool(fingertip_names)):
            in_contact = True

        f = lambda q: joint_space_objective(
            env, q, fingertip_names, in_contact,
            beta=beta, friction_coeff=mu,
            num_friction_cone_approx=num_cone_vec,
            q_plus_thresh=q_plus_thresh
        )
        grad = numeric_gradient(f, q_h, eps=grad_eps)
        q_h -= lr * grad
        env.set_configuration(q_h)

        # loss updating
        loss = np.append(loss, np.array([f(q_h)]))
        if f(q_h) < 1e-6:
            break
        if it % 50 == 0:
            print(f"[{it:4d}] loss={f(q_h):8.3e} | in_contact={in_contact}")

    return q_h, loss

def joint_space_objective(env: grasp_synthesis.AllegroHandEnv, 
                          q_h: np.array,
                          fingertip_names: list[str], 
                          in_contact: bool, 
                          beta=10.0, 
                          friction_coeff=0.5, 
                          num_friction_cone_approx=4,
                          q_plus_thresh: float = 1e-3):
    """
    This function minimizes an objective such that the distance from the origin
    in wrench space as well as distance from fingers to object surface is minimized.
    This is algorithm 2 in the project specification. 

    Parameters
    ----------
    env: AllegroHandEnv instance (can use to access physics)
    q_h: array of joint positions for the hand
    fingertip_names: names of the fingertips as defined in the MJCF
    in_contact: helper variable to determine if the fingers are in contact with the object
    beta: weight coefficient on the surface penalty 
    friction_coeff: Friction coefficient for the ball
    num_friction_cone_approx: number of approximation vectors in the friction cone

    
    Output
    ------
    fc_loss + (beta * d) as written in algorithm 2
    """
    env.set_configuration(q_h)
    #YOUR CODE HERE
    D = np.sum(np.square(env.fingertip_distances(fingertip_names)))
    if not in_contact:
        return beta * D

    positions, normals = env.contact_pos_normals(fingertip_names)
    cones = build_friction_cones(normals,
                                 mu=friction_coeff,
                                 num_approx=num_friction_cone_approx)
    G = build_grasp_matrix(positions, cones, origin=env.object_center())
    q_plus = optimize_necessary_condition(G)
    fc_loss = q_plus if q_plus > q_plus_thresh else optimize_sufficient_condition(G)
    return fc_loss + beta * D


def numeric_gradient(function: types.FunctionType, 
                     q_h: np.array, 
                     eps=0.001):
    """
    This function approximates the gradient of the joint_space_objective

    Parameters
    ----------
    function: function we are taking the gradient of
    q_h: joint configuration of the hand 
    env: AllegroHandEnv instance 
    fingertip_names: names of the fingertips as defined in the MJCF
    in_contact: helper variable to determine if the fingers are in contact with the object
    eps: hyperparameter for the delta of the gradient 

    Output
    ------
    Approximate gradient of the inputted function
    """
    baseline = function(q_h)
    grad = np.zeros_like(q_h)
    for i in range(len(q_h)):
        q_h_pert = q_h.copy()
        q_h_pert[i] += eps
        val_pert = function(q_h_pert)
        grad[i] = (val_pert - baseline) / eps
    return grad


def build_friction_cones(normals: np.array, mu=0.5, num_approx=4):
    """
    This function builds a discrete friction cone around each normal vector. 

    Parameters
    ----------
    normals: nx3 np.array where n is the number of normal directions
        normal directions for each contact
    mu: friction coefficient
    num_approx: number of approximation vectors in the friction cone

    Output
    ------
    friction_cone_vectors: array of discretized friction cones represented 
    as vectors
    """
    #YOUR CODE HERE
    cones = []
    for n in normals:
        n = n / np.linalg.norm(n)
        # Pick two orthonormal tangents u,v (Gram‑Schmidt)
        u = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
        u -= n * np.dot(n, u)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        vecs = []
        for k in range(num_approx):
            theta = 2*np.pi*k/num_approx
            dir_tangent = np.cos(theta)*u + np.sin(theta)*v
            vec = n + mu * dir_tangent
            vecs.append(vec / np.linalg.norm(vec))   # unit vectors
        cones.append(np.stack(vecs, axis=0))
    return cones


def build_grasp_matrix(contact_positions: np.array, friction_cones: list, origin=np.zeros(3)):
    """
    Builds a grasp map containing wrenches along the discretized friction cones. 

    Parameters
    ----------
    contact_positions: nx3 np.array of contact positions where n is the number of contacts
    firction_cone: a list of lists as outputted by build_friction_cones. 
    origin: the torque reference. In this case, it's the object center.
    
    Return a 2D numpy array G with shape (6, sum_of_all_cone_directions).
    """
    #YOUR CODE HERE
    wrenches = []
    for p, cone in zip(contact_positions, friction_cones):
        for f_dir in cone:
            tau = np.cross(p - origin, f_dir)
            wrench = np.concatenate((f_dir, tau))
            wrenches.append(wrench)
    G = np.stack(wrenches, axis=1)
    return G


def optimize_necessary_condition(G: np.array, env: grasp_synthesis.AllegroHandEnv):
    """
    Returns the result of the L2 optimization on the distance from wrench origin to the
    wrench space of G

    Parameters
    ----------
    G: grasp matrix
    env: AllegroHandEnv instance (can use to access physics)

    Returns the minimum of the objective

    Hint: use scipy.optimize.minimize
    """
    #YOUR CODE HERE
    m = G.shape[1]
    if m == 0:
        return 1e3
    
    def objective(x):
        return np.linalg.norm(G @ x)

    x0 = np.ones(m) / m
    bounds = [(0, None) for _ in range(m)]

    res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'ftol': 1e-6, 'disp': False})
    if not res.success:
        raise ValueError("Optimization failed: " + res.message)
    return res.fun


def optimize_sufficient_condition(G: np.array, K=20):
    """
    Runs the optimization from the project spec to evaluate Q- distance. 

    Parameters
    ----------
    G: grasp matrix
    K: number of approximations to the norm ball

    Returns the Q- value

    Hints:
        -Use scipy.optimize.linprog
        -Here's a resource with the basics: https://realpython.com/linear-programming-python/
        -You'll have to find a way to represent the alpha's for the constraints
            -Consider including the alphas in the linprog objective with coefficients 0 
        -For the optimization method, do method='highs'
    """
    #YOUR CODE HERE
    m = G.shape[1]
    q_minus_vals = []

    rng = np.random.default_rng(42)

    for _ in range(K):
        q = rng.normal(size=6)
        q /= np.linalg.norm(q)

        # Decision variables:   [ r , α₁ … α_m ]      length = 1+m
        c = np.zeros(1 + m)
        c[0] = -1             # maximise r  <=>  minimise -r

        # Constraints:  [ r q - Gα = 0 ]  (6 eqns)
        A_eq = np.zeros((7, 1 + m))
        A_eq[:6, 0]   = q
        A_eq[:6, 1:]  = -G
        A_eq[6, 1:]   = 1.0      # Σ α = 1
        b_eq = np.zeros(7)
        b_eq[6] = 1.0

        # Bounds:  r ≥ 0 , α ≥ 0
        bounds = [(0, None)] * (1 + m)

        res = linprog(c,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds,
                      method='highs')
        if res.success:
            r_val = res.x[0]
            q_minus_vals.append(-r_val)   # remember Q⁻ is -max_r

    # If nothing succeeded, return large positive penalty
    return max(q_minus_vals) if q_minus_vals else 1e3