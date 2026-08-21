"""SO(3) helpers — log/exp maps and transform vectorization."""

import numpy as np


def rotation_log_map(R):
    """Vector logarithm of a 3x3 rotation matrix.

    Returns omega in R^3 such that exp(hat(omega)) = R.
    ||omega|| is the geodesic angle on SO(3).
    """
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-8:
        skew = (R - R.T) / 2.0
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    scale = theta / (2.0 * np.sin(theta))
    return np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) * scale


def rotation_exp_map(omega):
    """Matrix exponential: builds 3x3 rotation from axis-angle vector."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def tee_cam_from_vec(v):
    """Build 4x4 T_ee_cam from 6-vector [px, py, pz, ox, oy, oz]."""
    T = np.eye(4)
    T[:3, 3] = v[:3]
    T[:3, :3] = rotation_exp_map(v[3:])
    return T


def vec_from_transform(T):
    """Extract 6-vector [position, axis-angle] from 4x4 transform."""
    pos = T[:3, 3]
    aa = rotation_log_map(T[:3, :3])
    return np.concatenate([pos, aa])
