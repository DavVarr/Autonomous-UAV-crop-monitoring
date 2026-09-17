import asyncio
import time
from dataclasses import dataclass
from math import cos, radians, sin

import cv2
import numpy as np
from mavsdk.mocap import (
    AngularVelocityBody, Covariance, Odometry, PositionBody, Quaternion, SpeedBody,
)

import fc_simulated as fc
from camera_simulated import my_estimatePoseSingleMarkers

CAMERA_TO_BODY = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


@dataclass(slots=True)
class ArucoObservation:
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    timestamp: float


class VisualOdometry:
    def __init__(self, video, detector, mtx, dist, marker_size=0.5, marker_id=None,
                 camera_to_body=CAMERA_TO_BODY, camera_position_body=None):
        self.video, self.detector = video, detector
        self.mtx, self.dist = np.asarray(mtx), np.asarray(dist)
        self.marker_size, self.marker_id = marker_size, marker_id
        self.camera_to_body = np.asarray(camera_to_body, float)
        self.camera_position_body = np.zeros(3) if camera_position_body is None else np.asarray(camera_position_body, float)
        self._latest = self._R_local_marker = self._p_local_marker = None

    def get_observation(self, max_age=0.15):
        obs = self._latest
        return obs if obs is not None and time.perf_counter() - obs.timestamp <= max_age else None

    def _detect(self, frame, timestamp):
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return None
        ids = np.asarray(ids).ravel()
        matches = np.arange(len(ids)) if self.marker_id is None else np.flatnonzero(ids == self.marker_id)
        if not len(matches):
            return None

        corner = corners[int(matches[0])]
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
            [corner], self.marker_size, self.mtx, self.dist)
        return ArucoObservation(
            np.asarray(corner).reshape(4, 2).copy(),
            np.asarray(rvecs[0], float).reshape(3),
            np.asarray(tvecs[0], float).reshape(3), timestamp)

    def _body_in_marker(self, obs):
        R_camera_marker, _ = cv2.Rodrigues(obs.rvec)
        R_marker_camera = R_camera_marker.T
        R_camera_body = self.camera_to_body.T
        R_marker_body = R_marker_camera @ R_camera_body
        p_marker_camera = -R_marker_camera @ obs.tvec
        p_marker_body = p_marker_camera - R_marker_body @ self.camera_position_body
        return p_marker_body, R_marker_body

    async def _initialize_frame(self, drone, obs):
        pv, attitude = await asyncio.gather(
            fc.get_drone_ned_position_velocity(drone), fc.get_drone_attitude_euler(drone))
        p_local_body = np.array([
            pv.position.north_m, pv.position.east_m, pv.position.down_m])
        R_local_body = _body_to_ned(attitude)
        p_marker_body, R_marker_body = self._body_in_marker(obs)

        self._R_local_marker = R_local_body @ R_marker_body.T
        self._p_local_marker = p_local_body - self._R_local_marker @ p_marker_body

    def _local_body_pose(self, obs):
        p_marker_body, R_marker_body = self._body_in_marker(obs)
        return (
            self._p_local_marker + self._R_local_marker @ p_marker_body,
            self._R_local_marker @ R_marker_body,
        )

    async def run(self, drone):
        last_frame = None
        while True:
            frame = self.video.frame()
            if frame is None or frame is last_frame:
                await asyncio.sleep(0.01)
                continue

            last_frame = frame
            obs = await asyncio.to_thread(self._detect, frame, time.perf_counter())
            self._latest = obs
            if obs is None:
                continue

            if self._R_local_marker is None:
                await self._initialize_frame(drone, obs)

            position, R = self._local_body_pose(obs)
            await drone.mocap.set_odometry(_odometry(position, R))


def _body_to_ned(attitude):
    r, p, y = map(radians, (attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg))
    cr, sr, cp, sp, cy, sy = cos(r), sin(r), cos(p), sin(p), cos(y), sin(y)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ])


def _quaternion(R):
    q = np.empty(4)
    trace = np.trace(R)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        q[:] = [s / 4, (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            q[:] = [(R[2, 1] - R[1, 2]) / s, s / 4,
                    (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            q[:] = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                    s / 4, (R[1, 2] + R[2, 1]) / s]
        else:
            s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            q[:] = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                    (R[1, 2] + R[2, 1]) / s, s / 4]
    q /= np.linalg.norm(q)
    return Quaternion(*map(float, q))


def _odometry(position, R):
    nan = float("nan")
    return Odometry(
        0, Odometry.MavFrame.LOCAL_FRD, PositionBody(*map(float, position)), _quaternion(R),
        SpeedBody(nan, nan, nan), AngularVelocityBody(nan, nan, nan),
        Covariance([nan]), Covariance([nan]), 0, Odometry.MavEstimatorType.VISION, 100)