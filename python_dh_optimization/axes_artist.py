import numpy as np
import cv2

def draw_frame_axes(img, camera_matrix, dist_coeffs,
                          rvec, tvec, length=20.0, margin=20):
    """Draw frame axes if all points project within frame bounds."""
    axis_pts = np.float32([
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length]
    ]).reshape(-1, 3)

    origin = np.float32([[0, 0, 0]])

    img_origin, _ = cv2.projectPoints(
        origin, rvec, tvec, camera_matrix, dist_coeffs
    )
    img_pts, _ = cv2.projectPoints(
        axis_pts, rvec, tvec, camera_matrix, dist_coeffs
    )

    h, w = img.shape[:2]
    for pt in [img_origin[0][0]] + [p[0] for p in img_pts]:
        x, y = pt
        if not (margin <= x <= w - margin and margin <= y <= h - margin):
            return False

    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, length)
    return True
