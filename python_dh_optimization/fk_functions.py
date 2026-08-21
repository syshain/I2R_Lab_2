import numpy as np
from scipy.spatial.transform import Rotation as R

def fk_from_pose(pose):
    x, y, z, roll, pitch, yaw = pose
    Rmat = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = [x, y, z]
    return T


def fk_from_zeta(q, zeta, n=6):
    """
    Standard D-H forward kinematics. q in radians. Returns 4x4 transform.
    Takes q as numpy array of joint angles in radians.
    zeta is a numpy array of DH parameters: [a_mm, alpha_rad, d_mm, theta_rad] * n.
    """
    a = zeta[0:n]
    alpha = zeta[n:2*n]
    d = zeta[2*n:3*n]
    theta_off = zeta[3*n:4*n]
    T = np.eye(4)

    for i in range(n):
        theta = q[i] + theta_off[i]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        T = T @ np.array([
            [ct, -st*ca,  st*sa, a[i]*ct],
            [st,  ct*ca, -ct*sa, a[i]*st],
            [0,    sa,     ca,   d[i]],
            [0,    0,      0,    1 ],
        ])
    return T
