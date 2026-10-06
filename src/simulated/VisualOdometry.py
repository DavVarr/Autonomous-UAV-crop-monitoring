import asyncio
import time
from dataclasses import dataclass

import cv2
import numpy as np
from mavsdk.mocap import AngleBody, Covariance, PositionBody, VisionPositionEstimate

import fc_simulated as fc
from camera_simulated import my_estimatePoseSingleMarkers


CAMERA_TO_BODY = np.array([
    [0., -1., 0.],
    [1.,  0., 0.],
    [0.,  0., 1.],
])


@dataclass(slots=True)
class ArucoObservation:
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    timestamp: float


class VisualOdometry:
    _next_reset_counter = 1

    def __init__(
        self, video, detector, mtx, dist, marker_size=0.5, marker_id=None,
        camera_to_body=CAMERA_TO_BODY, camera_position_body=None,
        position_sigma=0.05, attitude_sigma_deg=2.0,
    ):
        self.video, self.detector = video, detector
        self.mtx, self.dist = np.asarray(mtx, float), np.asarray(dist, float)
        self.marker_size, self.marker_id = float(marker_size), marker_id
        self.camera_to_body = np.asarray(camera_to_body, float)
        self.camera_position_body = (
            np.zeros(3) if camera_position_body is None
            else np.asarray(camera_position_body, float)
        )

        self._R_ned_marker = self._p_ned_marker = None
        self._latest = None

        self._ready = asyncio.Event()
        self._fusion_enabled = False
        self._valid = 0
        self._pose_covariance = _pose_covariance(
            position_sigma, np.deg2rad(attitude_sigma_deg)
        )

        self._reset_counter = VisualOdometry._next_reset_counter
        VisualOdometry._next_reset_counter = (
            VisualOdometry._next_reset_counter + 1
        ) & 0xFF

    async def wait_ready(self):
        await self._ready.wait()

    def get_observation(self, max_age=0.15):
        obs = self._latest
        if obs is None or time.perf_counter() - obs.timestamp > max_age:
            return None
        return obs

    @property
    def marker_position_ned(self):
        return (
            None if self._p_ned_marker is None
            else self._p_ned_marker.copy()
        )

    @property
    def marker_corners_ned(self):
        if self._p_ned_marker is None:
            return None

        h = self.marker_size / 2
        corners = np.array([
            [-h, -h, 0.], [h, -h, 0.],
            [h, h, 0.], [-h, h, 0.],
        ])
        return self._p_ned_marker + (self._R_ned_marker @ corners.T).T

    def _detect(self, frame, timestamp):
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return None

        ids = np.asarray(ids).ravel()
        if self.marker_id is None:
            i = 0
        else:
            matches = np.flatnonzero(ids == self.marker_id)
            if not len(matches):
                return None
            i = int(matches[0])

        corner = corners[i]
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
            [corner], self.marker_size, self.mtx, self.dist
        )

        return ArucoObservation(
            np.asarray(corner).reshape(4, 2).copy(),
            np.asarray(rvecs[0], float).reshape(3),
            np.asarray(tvecs[0], float).reshape(3),
            timestamp,
        )

    def _marker_in_body(self, obs):
        R_camera_marker, _ = cv2.Rodrigues(obs.rvec)

        p_body_marker = (
            self.camera_position_body
            + self.camera_to_body @ obs.tvec
        )
        R_body_marker = self.camera_to_body @ R_camera_marker

        return p_body_marker, R_body_marker

    async def _initialize_frame(self, drone, obs):
        pv, attitude = await asyncio.gather(
            fc.get_drone_ned_position_velocity(drone),
            fc.get_drone_attitude_euler(drone),
        )

        p_ned_body = np.array([
            pv.position.north_m,
            pv.position.east_m,
            pv.position.down_m,
        ])
        R_ned_body = _body_to_ned(attitude)
        p_body_marker, R_body_marker = self._marker_in_body(obs)

        # Fixed marker pose in the existing PX4 local-NED frame.
        self._R_ned_marker = R_ned_body @ R_body_marker
        self._p_ned_marker = p_ned_body + R_ned_body @ p_body_marker

    def _local_body_pose(self, obs, attitude):
        p_body_marker, R_body_marker = self._marker_in_body(obs)
        R_ned_body = _body_to_ned(attitude)

        return (
            self._p_ned_marker - R_ned_body @ p_body_marker,
            self._R_ned_marker @ R_body_marker.T,
        )

    def _vpe(self, position, R):
        roll, pitch, yaw = _euler321(R)
        return VisionPositionEstimate(
            0,
            PositionBody(*map(float, position)),
            AngleBody(float(roll), float(pitch), float(yaw)),
            self._pose_covariance,
            self._reset_counter,
        )

    async def run(self, drone):
        last_frame = None

        while True:
            frame = self.video.frame()
            if frame is None or frame is last_frame:
                await asyncio.sleep(0.005)
                continue

            last_frame = frame
            obs = await asyncio.to_thread(
                self._detect, frame, time.perf_counter()
            )

            if obs is None:
                continue

            self._latest = obs

            if self._R_ned_marker is None:
                await self._initialize_frame(drone, obs)

            self._valid += 1
            if self._valid >= 10:
                self._ready.set()


            attitude = await fc.get_drone_attitude_euler(drone)
            position, R = self._local_body_pose(obs, attitude)
            
            if self._fusion_enabled:
                await drone.mocap.set_vision_position_estimate(
                    self._vpe(position, R)
                )
    
    def enable_fusion(self):
        self._fusion_enabled = True

    def disable_fusion(self):
        self._fusion_enabled = False



def _body_to_ned(a):
    r, p, y = np.deg2rad([
        a.roll_deg, a.pitch_deg, a.yaw_deg
    ])
    cr, sr, cp, sp, cy, sy = (
        np.cos(r), np.sin(r), np.cos(p),
        np.sin(p), np.cos(y), np.sin(y),
    )

    return np.array([
        [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
        [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
        [-sp,   sr*cp,            cr*cp],
    ])


def _euler321(R):
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def _pose_covariance(
    position_sigma=0.05,
    attitude_sigma=np.deg2rad(2),
):
    ps = (
        [position_sigma] * 3
        if np.isscalar(position_sigma) else position_sigma
    )
    rs = (
        [attitude_sigma] * 3
        if np.isscalar(attitude_sigma) else attitude_sigma
    )

    cov = [0.0] * 21
    for i, sigma in zip(
        (0, 6, 11, 15, 18, 20),
        [*ps, *rs],
    ):
        cov[i] = float(sigma) ** 2

    return Covariance(cov)