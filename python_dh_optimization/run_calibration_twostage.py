"""Two-Stage Calibration — Hand-Eye Only, then Joint DH + Hand-Eye.

Stage 1: Fix DH at nominal, calibrate T_ee_cam only (6 params).
Stage 2: Calibrate DH + T_ee_cam jointly (30 params), initialised from Stage 1 result.

Results from both stages are printed side-by-side and saved to calibration_results.npy.
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from so3 import tee_cam_from_vec, rotation_log_map
from calibration_model import (predict_all_Tbb, Tbb_as_vectors,
                               compute_residuals_and_jacobian,
                               init_ee_cam_two_pose)
from lm_optimizer import levenberg_marquardt

_SCRIPT_DIR = Path(__file__).resolve().parent
_CALIB_DIR = _SCRIPT_DIR.parent / 'Calibration'
DATA_FILE = str(_CALIB_DIR / 'calibration_data.npy')


# ================================================================
# Data loading
# ================================================================
def load_calibration_data(path):
    """Load and preprocess calibration data."""
    data = np.load(str(path), allow_pickle=True).item()
    Q = data['robot_config'].copy()[:, :6]          # (l, 6) radians
    zeta0 = data['zeta_nominal']                    # (24,)
    Tcb = data['T_cam_board']                       # (l, 4, 4)
    for i in range(len(Tcb)):
        Tcb[i][:3, 3] /= 1000.0                     # mm -> metres
    return Q, zeta0, Tcb


def build_weight_vector(l, wp=100.0, wo=1.0):
    """Row weights: position rows get wp, orientation rows get wo."""
    w = np.ones(l * 6)
    for i in range(l):
        w[i * 6:i * 6 + 3] = wp
        w[i * 6 + 3:i * 6 + 6] = wo
    return w


# ================================================================
# Residual builders — different parameter subsets
# ================================================================
def make_handeye_only_residual_fn(Q, Tcb_list, zeta_fixed, n=6):
    """Return residual_fn that only optimises T_ee_cam (6 params)."""
    def fn(tee_vec):
        xi_full = np.concatenate([zeta_fixed, tee_vec])
        r, J_full = compute_residuals_and_jacobian(xi_full, Q, Tcb_list, n=n)
        # Keep only columns corresponding to T_ee_cam (last 6)
        J_ee = J_full[:, 4 * n:]
        return r, J_ee
    return fn


def make_joint_residual_fn(Q, Tcb_list, n=6):
    """Return residual_fn that optimises all 30 params."""
    def fn(xi):
        return compute_residuals_and_jacobian(xi, Q, Tcb_list, n=n)
    return fn


# ================================================================
# Reporting helpers
# ================================================================
def print_stage_header(stage_num, label):
    sep = "=" * 65
    print("\n" + sep)
    print("  STAGE %d: %s" % (stage_num, label))
    print(sep)


def print_tbb_error_stats(Q, Tcb_list, xi_or_parts, n=6, zeta_fixed=None):
    """Compute and print T_base_board error statistics across all poses."""
    if zeta_fixed is not None:
        # hand-eye-only mode: xi is just the 6-vector
        xi_full = np.concatenate([zeta_fixed, xi_or_parts])
    else:
        xi_full = xi_or_parts

    Tbb = predict_all_Tbb(Q, Tcb_list, xi_full, n)
    l = Tbb.shape[0]

    mean_pos = np.mean(Tbb[:, :3, 3], axis=0)
    mean_rot = Rotation.from_matrix(Tbb[:, :3, :3]).mean().as_matrix()

    trans_devs = []
    ori_devs_deg = []
    for i in range(l):
        dp = np.linalg.norm(Tbb[i, :3, 3] - mean_pos) * 1000
        trans_devs.append(dp)
        rel_R = mean_rot.T @ Tbb[i, :3, :3]
        ang = np.arccos(np.clip((np.trace(rel_R) - 1) / 2, -1, 1))
        ori_devs_deg.append(np.degrees(ang))

    trans_devs = np.array(trans_devs)
    ori_devs = np.array(ori_devs_deg)
    pos_std = np.std(Tbb[:, :3, 3], axis=0) * 1000

    print("\n  T_base_board consistency (deviation from mean, %d poses):" % l)
    print("    Translation (mm):  mean=%.3f  std=%.3f  max=%.3f" %
          (trans_devs.mean(), trans_devs.std(), trans_devs.max()))
    print("    Orientation (deg): mean=%.3f  std=%.3f  max=%.3f" %
          (ori_devs.mean(), ori_devs.std(), ori_devs.max()))
    print("    Per-axis pos std:  [%.3f, %.3f, %.3f] mm" % tuple(pos_std))

    return {
        'trans_mean_mm': float(trans_devs.mean()),
        'trans_std_mm': float(trans_devs.std()),
        'trans_max_mm': float(trans_devs.max()),
        'ori_mean_deg': float(ori_devs.mean()),
        'ori_std_deg': float(ori_devs.std()),
        'ori_max_deg': float(ori_devs.max()),
        'pos_axis_std_mm': pos_std.tolist(),
    }


def print_ee_cam_summary(tee_vec, label=""):
    """Print estimated T_ee_cam with translation and Euler angle details."""
    Tee = tee_cam_from_vec(tee_vec)
    pos = Tee[:3, 3]
    R_ee = Tee[:3, :3]
    euler = Rotation.from_matrix(R_ee).as_euler('xyz', degrees=True)
    aa = rotation_log_map(R_ee)
    aa_deg = np.degrees(np.linalg.norm(aa))

    hdr = "  %sT_ee_cam:" % label if label else "  T_ee_cam:"
    print(hdr)
    print("    Translation : [%.4f, %.4f, %.4f] m  (%.1f, %.1f, %.1f mm)" %
          (pos[0], pos[1], pos[2], pos[0]*1000, pos[1]*1000, pos[2]*1000))
    print("    Euler xyz   : [%.2f, %.2f, %.2f] deg" % (euler[0], euler[1], euler[2]))
    print("    Axis-angle  : %.2f deg about [%.4f, %.4f, %.4f]" %
          (aa_deg, aa[0], aa[1], aa[2]))
    return Tee


def print_dh_table(zeta_init, zeta_cal, n=6):
    """Print calibrated DH parameters in a table."""
    names = (["a%d" % (i + 1) for i in range(n)] +
             ["alpha%d" % (i + 1) for i in range(n)] +
             ["d%d" % (i + 1) for i in range(n)] +
             ["theta%d" % (i + 1) for i in range(n)])
    units = (["m"] * n + ["rad"] * n + ["m"] * n + ["rad"] * n)
    delta = zeta_cal - zeta_init

    print("\n  DH Parameter Corrections:")
    print("  " + "-" * 58)
    print("  %-10s %12s %12s %+12s  %s" % ("param", "nominal", "calib", "delta", "unit"))
    print("  " + "-" * 58)
    for j in range(4 * n):
        prec = 6
        print("  %-10s %12.*f %12.*f %+12.*f  %s" %
              (names[j], prec, zeta_init[j], prec, zeta_cal[j], prec, delta[j], units[j]))
    print("  " + "=" * 58)


def print_comparison(stats_he, stats_joint):
    """Print side-by-side comparison of Stage 1 vs Stage 2 results."""
    print("\n" + "=" * 65)
    print("  COMPARISON: Hand-Eye Only  vs  Joint DH+HandEye")
    print("=" * 65)
    print("  %-28s %12s %12s %10s" % ("metric", "HE only", "Joint", "improvement"))
    print("  " + "-" * 65)

    metrics = [
        ('Trans mean (mm)', 'trans_mean_mm', 'trans_mean_mm'),
        ('Trans std (mm)', 'trans_std_mm', 'trans_std_mm'),
        ('Trans max (mm)', 'trans_max_mm', 'trans_max_mm'),
        ('Ori mean (deg)', 'ori_mean_deg', 'ori_mean_deg'),
        ('Ori std (deg)', 'ori_std_deg', 'ori_std_deg'),
        ('Ori max (deg)', 'ori_max_deg', 'ori_max_deg'),
    ]
    for label, k_he, k_j in metrics:
        v_he = stats_he[k_he]
        v_j = stats_joint[k_j]
        if v_he != 0:
            improvement = (v_he - v_j) / abs(v_he) * 100
            imp_str = "%.1f%%" % improvement
        else:
            imp_str = "---"
        print("  %-28s %12.3f %12.3f %10s" % (label, v_he, v_j, imp_str))
    print("=" * 65)


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    data_path = Path(DATA_FILE)
    if not data_path.exists():
        print("No data file found at " + DATA_FILE)
        raise SystemExit(1)

    # --- Load data ---
    Q, zeta0, Tcb_list = load_calibration_data(data_path)
    l = Q.shape[0]
    n_dof = 6
    w_rows = build_weight_vector(l, wp=1000.0, wo=50.0)
    print("Loaded %d configurations from %s" % (l, DATA_FILE))

    # --- Initialise T_ee_cam ---
    tee_init = init_ee_cam_two_pose(Q, Tcb_list, zeta0, n=n_dof)
    Tee_init = tee_cam_from_vec(tee_init)
    print("\nInitial T_ee_cam (two-pose estimate):")
    print(Tee_init)

    # Show baseline spread with nominal DH + init T_ee_cam
    xi_baseline = np.concatenate([zeta0, tee_init])
    print("\nBaseline T_base_board spread (nominal DH + init T_ee_cam):")
    print_tbb_error_stats(Q, Tcb_list, tee_init, n=n_dof, zeta_fixed=zeta0)

    ####################################################################
    # STAGE 1: Hand-Eye Only (DH fixed at nominal)
    ####################################################################
    print_stage_header(1, "Hand-Eye Calibration Only (DH fixed at nominal)")

    he_residual_fn = make_handeye_only_residual_fn(Q, Tcb_list, zeta0, n=n_dof)

    result_he = levenberg_marquardt(
        he_residual_fn, tee_init, w_rows,
        max_iter=200, tol=1e-6, lam0=1e-3,
        reg_weight=0.0, svd_thresh=0.0, verbose=True
    )

    tee_he = result_he['xi']
    Tee_he = print_ee_cam_summary(tee_he, label="(Stage 1 HE-only) ")
    stats_he = print_tbb_error_stats(Q, Tcb_list, tee_he, n=n_dof, zeta_fixed=zeta0)

    ####################################################################
    # STAGE 2: Joint DH + Hand-Eye (initialised from Stage 1)
    ####################################################################
    print_stage_header(2, "Joint DH + Hand-Eye Calibration")

    xi0_joint = np.concatenate([zeta0, tee_he])
    joint_residual_fn = make_joint_residual_fn(Q, Tcb_list, n=n_dof)

    result_joint = levenberg_marquardt(
        joint_residual_fn, xi0_joint, w_rows,
        max_iter=300, tol=1e-6, lam0=1e-3,
        reg_weight=1e-2, svd_thresh=1e-6, verbose=True
    )

    xi_joint = result_joint['xi']
    zeta_joint = xi_joint[:4 * n_dof]
    tee_joint = xi_joint[4 * n_dof:]

    print_dh_table(zeta0, zeta_joint, n=n_dof)
    Tee_joint = print_ee_cam_summary(tee_joint, label="(Stage 2 Joint) ")
    stats_joint = print_tbb_error_stats(Q, Tcb_list, xi_joint, n=n_dof)

    ####################################################################
    # Comparison
    ####################################################################
    print_comparison(stats_he, stats_joint)

    ####################################################################
    # Save
    ####################################################################
    cal_file = str(data_path.parent / 'calibration_results.npy')
    np.save(cal_file, {
        # Stage 1
        'T_ee_cam_he_only': Tee_he,
        'tee_cam_vec_he_only': tee_he,
        'he_iters': result_he['iters'],
        'he_converged': result_he['converged'],
        'he_trans_mean_mm': stats_he['trans_mean_mm'],
        'he_ori_mean_deg': stats_he['ori_mean_deg'],
        # Stage 2
        'zeta_calibrated': zeta_joint,
        'zeta_nominal': zeta0,
        'delta_zeta': zeta_joint - zeta0,
        'T_ee_cam_joint': Tee_joint,
        'tee_cam_vec_joint': tee_joint,
        'joint_iters': result_joint['iters'],
        'joint_converged': result_joint['converged'],
        'joint_trans_mean_mm': stats_joint['trans_mean_mm'],
        'joint_ori_mean_deg': stats_joint['ori_mean_deg'],
    })
    print("\nSaved to " + cal_file)
