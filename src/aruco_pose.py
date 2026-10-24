from typing import Protocol

import cv2
import numpy as np


class VideoSource(Protocol):
    def frame_available(self) -> bool: ...

    def frame(self) -> np.ndarray: ...


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array(
        [
            [-marker_size / 2, -marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [-marker_size / 2, marker_size / 2, 0],
        ],
        dtype=np.float32,
    )

    success = []
    rvecs = []
    tvecs = []

    for corner in corners:
        result, rotation, translation = cv2.solvePnP(
            marker_points,
            corner,
            mtx,
            distortion,
            True,
            cv2.SOLVEPNP_IPPE_SQUARE,
        )
        rvecs.append(rotation)
        tvecs.append(translation)
        success.append(result)

    return rvecs, tvecs, success


def checkArucoPresence(
    video: VideoSource, detector, mtx, dist, marker_size=0.5
):
    while True:
        if not video.frame_available():
            continue

        frame = video.frame().copy()
        frame.setflags(True)

        marker_corners, marker_ids, _ = detector.detectMarkers(frame)
        if marker_ids is not None:
            return my_estimatePoseSingleMarkers(
                marker_corners, marker_size, mtx, dist
            )
        return None
