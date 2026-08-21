"""Calibration model — T_base_board prediction, residuals, and Jacobian.

Frame chain:  Board <-T_cam_board<- Camera <-T_ee_cam<- EE <-FK(zeta)-> Base

At each pose i:  T_base_board_i = FK_i(zeta) @ T_ee_cam @ T_cam_board_i

The ArUco board is fixed in space, so T_base_board must be constant across
all poses.  We use pose 0 as the reference and define residuals for every
other pose as the deviation from that reference.

Combined parameter vector xi = [zeta(24), tee_cam_vec(6)] = 30 parameters.
"""

import numpy as np

from fk_functions import fk_from_zeta
from so3 import tee_cam_from_vec, vec_from_transform


def predict_all_Tbb(Q, Tcb_list, xi, n=6):
    """Compute T_base_board for every pose.

    Returns array of shape (l, 4, 4).
    """
    zeta = xi[:4 * n]
    T_ee_cam = tee_cam_from_vec(xi[4 * n:])
    l = Q.shape[0]
    Tbb = np.empty((l, 4, 4))
    for i in range(l):
        T_be = fk_from_zeta(Q[i], zeta, n)
        Tbb[i] = T_be @ T_ee_cam @ Tcb_list[i]
    return Tbb


def Tbb_as_vectors(Tbb):
    """Convert (l, 4, 4) transforms to (l, 6) vectors [pos, axis-angle]."""
    l = Tbb.shape[0]
    V = np.empty((l, 6))
    for i in range(l):
        V[i] = vec_from_transform(Tbb[i])
    return V


def compute_residuals_and_jacobian(xi, Q, Tcb_list, n=6, eps=1e-7):
    """Residuals and finite-difference Jacobian in one sweep.

    Pose 0 defines the reference T_base_board.  For each pose i >= 1:
        residual_i = V_0 - V_i

    J = d(V_i - V_0)/d(xi), computed by finite difference.
    Row block 0 is all zeros (V_0 - V_0 = 0 always).

    Returns (r_flat, J) where r_flat has length 6*l and J is (6*l, len(xi)).
    """
    l = Q.shape[0]
    np_params = len(xi)

    Tbb_cur = predict_all_Tbb(Q, Tcb_list, xi, n)
    V_cur = Tbb_as_vectors(Tbb_cur)       # (l, 6)
    V_ref = V_cur[0]                       # pose 0 reference

    r = (V_ref - V_cur).flatten()          # (6*l,)

    J = np.zeros((l * 6, np_params))
    for j in range(np_params):
        xp = xi.copy()
        xp[j] += eps
        Tbp = predict_all_Tbb(Q, Tcb_list, xp, n)
        Vp = Tbb_as_vectors(Tbp)
        delta = (Vp - Vp[0] - (V_cur - V_ref)).flatten()
        J[:, j] = delta / eps
    return r, J


def _orthogonalise_rotation(R):
    """Project a 3x3 matrix onto SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        Vt[-1, :] *= -1
        R_proj = U @ Vt
    return R_proj


def init_ee_cam_two_pose(Q, Tcb_list, zeta_nom, n=6):
    """Rough T_ee_cam estimate from two poses with largest baseline.

    Assumes T_bb = FK(pi) @ Tcb[pi] with TeeCam approx identity, then solves
    for TeeCam from both poses and averages.  Orthogonalises rotation via SVD.
    Falls back to zeros if anything fails.
    """
    try:
        l = Q.shape[0]
        T_be_all = np.array([fk_from_zeta(Q[i], zeta_nom, n) for i in range(l)])

        best_gap = 0
        best_pair = (0, 1)
        for i in range(min(l, 10)):
            for j in range(i + 1, min(l, 10)):
                gap = np.linalg.norm(Tcb_list[i][:3, 3] - Tcb_list[j][:3, 3])
                if gap > best_gap:
                    best_gap = gap
                    best_pair = (i, j)

        pi, pj = best_pair
        T_bb_guess = T_be_all[pi] @ Tcb_list[pi]
        Tee_i = np.linalg.inv(T_be_all[pi]) @ T_bb_guess @ np.linalg.inv(Tcb_list[pi])
        Tee_j = np.linalg.inv(T_be_all[pj]) @ T_bb_guess @ np.linalg.inv(Tcb_list[pj])
        Tee_avg = (Tee_i + Tee_j) / 2.0

        Tee_avg[:3, :3] = _orthogonalise_rotation(Tee_avg[:3, :3])
        return vec_from_transform(Tee_avg)
    except Exception:
        return np.zeros(6)


def init_ee_cam_horaud(Q, Tcb_list, zeta_nom, n=6):
    """Horaud et al. (1989) linear AX=XB hand-eye solver.

    For every pair (i, ref) with ref=0:
        A_ij = FK_ref^{-1} @ FK_i      (robot relative motion)
        B_ij = Tcb_ref @ Tcb_i^{-1}     (camera-relative board motion)

    Solve A_ij @ X = X @ B_ij for X = T_ee_cam using least squares
    over all pairs simultaneously.  Rotation solved first (vectorised 9-eq),
    then translation (linear 3-eq given R).

    Returns 6-vector [position, axis-angle].
    """
    try:
        l = Q.shape[0]
        T_be_all = np.array([fk_from_zeta(Q[i], zeta_nom, n) for i in range(l)])
        T_be_inv = np.array([np.linalg.inv(T_be_all[i]) for i in range(l)])
        Tcb_inv = np.array([np.linalg.inv(Tcb_list[i]) for i in range(l)])

        ref = 0
        # Collect A_k = A_{k,ref}, B_k = B_{k,ref} for k = 1..l-1
        Ak_list, Bk_list = [], []
        for k in range(1, l):
            Ak = T_be_inv[ref] @ T_be_all[k]
            Bk = Tcb_list[ref] @ Tcb_inv[k]
            Ak_list.append(Ak)
            Bk_list.append(Bk)

        m = len(Ak_list)
        if m < 2:
            return init_ee_cam_two_pose(Q, Tcb_list, zeta_nom, n)

        # --- Solve rotation: R_A @ X_R = X_R @ R_B  => vectorise ---
        # vec(A@X) = (I ⊗ A) vec(X),  vec(X@B) = (B^T ⊗ I) vec(X)
        # So: [(I⊗A_k) - (B_k^T⊗I)] vec(X_R) = 0  for each k
        from numpy import kron
        Mrhs = []
        for k in range(m):
            Ra = Ak_list[k][:3, :3]
            Rb = Bk_list[k][:3, :3]
            M_k = kron(np.eye(3), Ra) - kron(Rb.T, np.eye(3))
            Mrhs.append(M_k)
        Mrhs = np.vstack(Mrhs)  # (9*m, 9)

        # Solution = right singular vector corresponding to smallest singular value
        _, _, Vt_r = np.linalg.svd(Mrhs, full_matrices=False)
        x_vec = Vt_r[-1, :]          # (9,) — last row of Vt
        R_sol = x_vec.reshape(3, 3).T  # reshape: column-major from vec()
        R_sol = _orthogonalise_rotation(R_sol)

        # --- Solve translation given R ---
        # A @ [R p; 0 1] = [R p; 0 1] @ B
        # => Ra @ p - pb = Rp @ Rb @ t ... rearrange:
        # For each pair k: Ra_k @ p - p = Rp_k @ tb_k  where tb_k is B's translation
        # => (Ra_k - I) @ p = Rp_k @ tb_k - p_b_k  ... actually:
        # Full expand: Ra_k @ [R p] = [R p] @ [Rb kb]
        # Left side top-right: Ra_k @ p
        # Right side top-right: R @ kb_k + p_b_k  (but A has zero translation offset here)
        # Since A = inv(FK_ref) @ FK_k, A's translation is non-zero generally.
        # Let ta_k = Ak[:3,3], tb_k = Bk[:3,3]
        # Ra @ [R p] = [R p] @ [Rb tb]
        # => Ra @ p = R @ tb + ta   (since left side is Ra@[R;p cols] = Ra@p for trans part)
        # Wait, let me be careful:
        # A @ X = X @ B
        # [Ra ta; 0 1] @ [R p; 0 1] = [R p; 0 1] @ [Rb tb; 0 1]
        # LHS top-left: Ra @ R,  LHS top-right: Ra @ p + ta
        # RHS top-left: R @ Rb,  RHS top-right: R @ tb + p
        # Translation eq: Ra @ p + ta = R @ tb + p
        # => (Ra - I) @ p = R @ tb - ta

        Trhs = []
        Tlhs = []
        for k in range(m):
            Ra = Ak_list[k][:3, :3]
            ta = Ak_list[k][:3, 3]
            tb = Bk_list[k][:3, 3]
            Tlhs.append(Ra - np.eye(3))
            Trhs.append(R_sol @ tb - ta)

        Tlhs = np.vstack(Tlhs)   # (3*m, 3)
        Trhs = np.concatenate(Trhs)  # (3*m,)
        p_sol, _, _, _ = np.linalg.lstsq(Tlhs, Trhs, rcond=None)

        T_ee = np.eye(4)
        T_ee[:3, :3] = R_sol
        T_ee[:3, 3] = p_sol
        return vec_from_transform(T_ee)
    except Exception:
        return init_ee_cam_two_pose(Q, Tcb_list, zeta_nom, n)
