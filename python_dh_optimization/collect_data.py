"""
Data Collection — UFACTORY Lite 6 DH Calibration

Moves the robot through pre-determined calibration configurations
At each configuration, records joint angles, aruco based pose, and full end-effector poses and saves to ../Data/calibration_data.npy.

Run once per calibration session. Then iterate on the optimizer without
touching the robot again.

Units stored in the output file:
    Q_configs  : radians
    X_measured : cols 0-2 = position (metres), cols 3-5 = orientation (radians)
                  [x, y, z, roll, pitch, yaw]
    Use metres and radians for SI units.
"""

import numpy as np
import time
from pathlib import Path
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R

from aruco_board_detector import Aruco3DBoardDetector as BoardDetector
from axes_artist import draw_frame_axes

from xarm.wrapper import XArmAPI

from fk_functions import fk_from_pose, fk_from_zeta

_SCRIPT_DIR = Path(__file__).resolve().parent
_CALIB_DIR = _SCRIPT_DIR.parent / 'Calibration'

ROBOT_IP = '192.168.1.153'  # the UFactory robot IP

OUTPUT_FILE = str(_CALIB_DIR / 'calibration_data.npy')
CAMERA_MATRIX = str(_CALIB_DIR / 'camera_matrix.npy')
CAMERA_DIST = str(_CALIB_DIR / 'dist_coeffs.npy')
BOARD_CONFIG = str(_CALIB_DIR / 'board_config.json')

n = 6 # By default, collect all 6 degrees of freedom

# ================================================================
# Nominal DH parameters (Standard D-H, Appendix 7 of user manual)
# ================================================================
# Columns: [theta_offset_deg, d_mm, alpha_deg, a_mm]

dh_nominal_raw = np.array([
    [  0,   243.3,  -90,    0],
    [-90,     0.0,  180,  200],
    [-90,     0.0,   90,   87],
    [  0,   227.6,   90,    0],
    [  0,     0.0,  -90,    0],
    [  0,    61.5,    0,    0],
])
# The above is in mm and degrees - convert to meters and radians below

a_nom     = dh_nominal_raw[:, 3] / 1000.0          # mm -> m
alpha_nom = np.deg2rad(dh_nominal_raw[:, 2])       # deg -> rad
d_nom     = dh_nominal_raw[:, 1] / 1000.0          # mm -> m
th_nom    = np.deg2rad(dh_nominal_raw[:, 0])       # deg -> rad
zeta_nominal = np.concatenate([a_nom, alpha_nom, d_nom, th_nom])

# ================================================================
if __name__ == '__main__':

    """
    UFactory robot setup steps
    1. Connect
    2. Clean any errors
    3. Enable robot motion
    4. Set mode to position control
    5. Set state to 0, that is enable the robot
    """

    print(f"Connecting to robot at {ROBOT_IP} …")
    arm = XArmAPI(ROBOT_IP)
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)

    # Move to home position
#    print("Moving to home …")
#    arm.move_gohome(wait=True)
#    time.sleep(3)
#
#    # Diagnostic at home
#    home_code, home_angles_rad = arm.get_servo_angle(is_radian=True)
#    T_home = fk_from_zeta(home_angles_rad, zeta_nominal, n)
#    # fk_from_zeta returns position in meters, convert to mm
#    pos_model_mm = T_home[:3, 3] * 1000.00  # in mm
#
#    # get_position() returns [x(mm), y(mm), z(mm), roll(rad), pitch(rad), yaw(rad)]
#    home_code, home_pose = arm.get_position(is_radian=True)
#    pos_measured_mm = np.array(home_pose[:3])
#    diag_err = np.linalg.norm(pos_measured_mm - pos_model_mm)
#
#    print(f"\n[Home: controller reports {np.rad2deg(home_angles_rad[:n])} deg]")
#    print(f"  Model FK   : [{pos_model_mm[0]:7.1f}, {pos_model_mm[1]:7.1f}, {pos_model_mm[2]:7.1f}] mm")
#    print(f"  Measured   : [{pos_measured_mm[0]:7.1f}, {pos_measured_mm[1]:7.1f}, {pos_measured_mm[2]:7.1f}] mm")
#    print(f"  Discrepancy: {diag_err:.1f} mm")
#    if diag_err > 50:
#        print("  WARNING: large discrepancy — check angle convention!\n")
#    else:
#        print("  OK\n")
#
    # Initialize aruco detector

    detector = BoardDetector(
        camera_index=1,
        calibration_folder=str(_CALIB_DIR),
        config_path=BOARD_CONFIG
    )

    if not detector.cap.isOpened():
        print(f"Failed to open camera {detector.camera_index}")
        raise SystemExit

    detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # --- Collect calibration poses ---
    print("Press 'c' to capture a pose, 's' to save, 'q' to quit.\n")
    Q_accepted = []
    X_measured = []
    pose_ids = []
    T_cam_board_list = []
    T_base_ee_zeta_list = []
    T_base_ee_actual_list = []
    zeta_nominal_val = zeta_nominal
    marker_ids_list = []
    reproj_errors = []
    pose_id = 0

    while True:

        ret, frame = detector.cap.read()
        if not ret:
            print("Failed to read frame")
            continue
        success, T_cam_board, rvec, tvec, reproj_error, corners, ids = detector.detect(frame)

        # Build IDs string first so it's available everywhere
        if ids is not None:
            ids_text = ",".join(map(str, ids.flatten().tolist()))
        else:
            ids_text = "none"

        # Draw all detected markers

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(
                frame, f"Detected IDs: {ids_text}",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
            )

        # ── Overlay ────────────────────────────────────────────────────────
        if success:
            drawn = draw_frame_axes(
                frame,
                detector.camera_matrix,
                detector.dist_coeffs,
                rvec, tvec,
                length=20.0
            )

            if not drawn:
                origin_2d, _ = cv2.projectPoints(
                    np.float32([[0, 0, 0]]),
                    rvec, tvec,
                    detector.camera_matrix,
                    detector.dist_coeffs
                )
                cv2.circle(
                    frame,
                    tuple(origin_2d[0][0].astype(int)),
                    5, (0, 255, 0), -1
                )

            cv2.putText(
                frame,
                f"Board detected [{ids_text}] — press 'c' to capture",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Reproj err: {reproj_error:.2f} px",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )
            y_text = 90

        else:
            cv2.putText(
                frame, "Board not detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )
            y_text = 60

        cv2.putText(
            frame, f"Poses saved: {len(Q_accepted)}",
            (10, y_text), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 0), 2
        )

        cv2.imshow("3D ArUco Board Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── CAPTURE ───────────────────────────────────────────────────────
        if key == ord('c'):
            if not success:
                print("Board not detected — cannot capture.")
                continue

            if reproj_error is None or reproj_error > 4.0:
                print(f"Pose rejected: reproj error too high "
                      f"({reproj_error:.2f} px)")
                continue

            code, raw_pose = arm.get_position(is_radian=True)
            if code != 0:
                print(f"get_position failed (code {code}), skipping.")
                continue
            raw_pose[:3] = [x/1000.00 for x in raw_pose[:3]] # convert to meters

            code, angles_rad = arm.get_servo_angle(is_radian=True)
            if code != 0:
                print(f"get_servo_angle failed (code {code}), skipping.")
                continue
            T_base_ee_zeta = fk_from_zeta(angles_rad[:n], zeta_nominal, n)
            T_base_ee_actual = fk_from_pose(raw_pose)

            pose_id += 1
            pose_ids.append(pose_id)
            T_cam_board_list.append(T_cam_board)
            Q_accepted.append(angles_rad)
            X_measured.append(raw_pose)
            T_base_ee_zeta_list.append(T_base_ee_zeta)
            T_base_ee_actual_list.append(T_base_ee_actual)
            marker_ids_list.append(ids.flatten().tolist() if ids is not None else [])
            reproj_errors.append(float(reproj_error))


            # ── Terminal log ───────────────────────────────────────────────
            print("\n"+"="*40)
            print(f"Pose {pose_id} captured and saved")
            print(f"Reprojection error : {reproj_error:.2f} px")
            print(f"Raw robot pose     : {raw_pose}")
            print("\n"+"="*40)

        elif key == ord('q'):
            break

        elif key == ord('s'):
            if pose_id >= 2:
                np.save(OUTPUT_FILE, {
                    'pose_ids':           pose_ids,
                    'robot_config':       np.array(Q_accepted),             # joint angles, RADIANS (l, 6)
                    'robot_pose':         np.array(X_measured),              # [x(m), y(m), z(m), roll(rad), pitch(rad), yaw(rad)]
                    'T_base_ee_zeta':     np.array(T_base_ee_zeta_list),    # FK from nominal DH (list of 4x4)
                    'T_base_ee_actual':   np.array(T_base_ee_actual_list),  # FK from controller pose (list of 4x4)
                    'T_cam_board':        np.array(T_cam_board_list),       # camera-to-board transforms (list of 4x4)
                    'marker_ids_used':    marker_ids_list,                  # which aruco markers were detected
                    'reprojection_error': np.array(reproj_errors),          # reprojection error in pixels
                    'zeta_nominal':       zeta_nominal_val,                 # nominal DH params
                    'n':                  pose_id,                          # number of configurations
                })
                print(f"Saved {pose_id} calibration poses to {OUTPUT_FILE}")
            else:
                print(f"Too few poses ({pose_id}/2 min). Press 'c' to capture more.")

    detector.cap.release()
    arm.disconnect()
    cv2.destroyAllWindows()
