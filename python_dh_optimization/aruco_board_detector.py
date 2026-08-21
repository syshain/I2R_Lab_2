import json
from pathlib import Path
import numpy as np
import cv2
import cv2.aruco as aruco
import scipy.spatial.transform as transform


class Aruco3DBoardDetector:
    """
    3D ArUco polyhedron detector.
    Loads corner coordinates from board_config.json.
    Scales from 30mm -> 40mm markers (scale = 4/3).
    Re-centers so origin = geometric center of polyhedron.
    """

    CORNER_ROLL = {
        0: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
    }

    def __init__(self, camera_index=1, calibration_folder=None, config_path=None):
        if calibration_folder is None:
            calibration_folder = str(Path(__file__).resolve().parent.parent / 'Calibration')
        if config_path is None:
            config_path = str(Path(calibration_folder) / 'board_config.json')

        self.camera_matrix = np.load(Path(calibration_folder) / 'camera_matrix.npy')
        self.dist_coeffs   = np.load(Path(calibration_folder) / 'dist_coeffs.npy')

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.detector = aruco.ArucoDetector(self.aruco_dict, params)

        self.cap          = cv2.VideoCapture(camera_index)
        self.camera_index = camera_index
        self.scale        = 4.0 / 3.0
        self.config_path  = config_path

        self._build_3d_model()

    # ─────────────────────────────────────────────────────────────────────
    def _build_3d_model(self):
        with open(self.config_path, 'r') as f:
            cfg = json.load(f)

        board   = cfg['toolList'][0]
        ids     = board['marker_ids']
        corners = board['marker_corners_m']

        # Scale + winding correction
        scaled = {}
        for mid, raw in zip(ids, corners):
            pts  = np.array(raw, dtype=np.float64) * self.scale
            roll = self.CORNER_ROLL.get(mid, 0)
            scaled[mid] = np.roll(pts, -roll, axis=0)

        # Re-center: subtract centroid of all corners
        all_pts  = np.vstack([scaled[mid] for mid in ids])
        centroid = all_pts.mean(axis=0)

        self.marker_world    = {}
        self.marker_ids_list = ids
        self.centroid_offset = centroid

        for mid in ids:
            self.marker_world[mid] = scaled[mid] - centroid

        print(f"[Model] Loaded {len(ids)} markers  "
              f"scale={self.scale:.4f}  "
              f"centroid=[{centroid[0]:.2f},"
              f"{centroid[1]:.2f},{centroid[2]:.2f}]")

    # ─────────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray):
        """
        Detect markers and solve for camera->object transform.

        Returns:
            success      : bool
            T_cam_board  : (4,4) homogeneous transform
            rvec         : (3,1)
            tvec         : (3,1)
            reproj_error : float (pixels)
            corners      : raw ArUco corners
            ids          : raw ArUco ids
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return False, None, None, None, None, corners, ids

        ids_flat      = ids.flatten()
        object_points = []
        image_points  = []
        matched_ids   = []

        for i, mid in enumerate(ids_flat):
            if mid in self.marker_world:
                roll      = self.CORNER_ROLL.get(mid, 0)
                world_pts = np.roll(self.marker_world[mid], -roll, axis=0)
                for j in range(4):
                    object_points.append(world_pts[j])
                    image_points.append(corners[i][0][j])
                matched_ids.append(int(mid))

        if len(object_points) < 4:
            return False, None, None, None, None, corners, ids

        object_points = np.array(object_points, dtype=np.float32)
        image_points  = np.array(image_points,  dtype=np.float32)

        # ── solvePnP ──────────────────────────────────────────────────────
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )

        if not success:
            return False, None, None, None, None, corners, ids

        # ── LM refinement ─────────────────────────────────────────────────
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points, image_points,
                self.camera_matrix, self.dist_coeffs,
                rvec, tvec
            )
        except Exception:
            pass

        # ── Reprojection error ─────────────────────────────────────────────
        proj, _ = cv2.projectPoints(
            object_points, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )
        reproj_error = float(np.mean(
            np.linalg.norm(
                proj.reshape(-1, 2) - image_points, axis=1
            )
        ))

        if reproj_error > 15.0:
            print(f"[REJECT] reproj_err={reproj_error:.2f}px > 15px")
            return False, None, None, None, reproj_error, corners, ids

        R_mat, _ = cv2.Rodrigues(rvec)
        T_cam_board         = np.eye(4, dtype=np.float64)
        T_cam_board[:3, :3] = R_mat
        T_cam_board[:3,  3] = tvec.flatten()

        return True, T_cam_board, rvec, tvec, reproj_error, corners, ids
