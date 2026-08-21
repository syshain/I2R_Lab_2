import numpy as np
import scipy.spatial.transform as transform

class VerticalArucoBoardDetector:
    def __init__(self, camera_index=4):
        # Load camera calibration
        self.camera_matrix = np.load('camera_matrix.npy')
        self.dist_coeffs = np.load('dist_coeffs.npy')

        # ============================================================
        # ARUCO SETUP for OpenCV 4.13
        # ============================================================
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # ============================================================
        # BOARD DEFINITION (Vertical stack: 2 rows, 1 column)
        # ============================================================
        self.marker_length = 100.0      # mm
        self.marker_separation = 25.0   # mm (edge-to-edge gap)

        # Build object points for solvePnP
        self._build_board_points()

        # Camera setup
        self.cap = None
        self.camera_index = camera_index
        self.calibration_data = []
