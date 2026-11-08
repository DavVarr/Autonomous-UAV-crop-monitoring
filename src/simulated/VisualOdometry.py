import asyncio
import time
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from mavsdk_grpc.mocap import AngleBody, Covariance, PositionBody, VisionPositionEstimate

from aruco_pose import my_estimatePoseSingleMarkers
import simulated.fc_simulated as fc


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
    marker_id: int | None = None


class VisualOdometry:
    _next_reset_counter = 1

    def __init__(
        self, video, detector, mtx, dist, marker_size=0.5, marker_id=None,
        camera_to_body=CAMERA_TO_BODY, camera_position_body=None,
        position_sigma=0.05, attitude_sigma_deg=2.0, origin_altitude_m=0.0,
    ):
        self.video, self.detector = video, detector
        self.mtx, self.dist = np.asarray(mtx, float), np.asarray(dist, float)
        self.marker_size, self.marker_id = float(marker_size), marker_id
        self._origin_altitude_m = float(origin_altitude_m)
        if not np.isfinite(self._origin_altitude_m):
            raise ValueError("EV origin altitude must be finite")
        self.camera_to_body = np.asarray(camera_to_body, float)
        self.camera_position_body = (
            np.zeros(3) if camera_position_body is None
            else np.asarray(camera_position_body, float)
        )

        self._R_ned_marker = self._p_ned_marker = None
        self._latest = None
        self._last_corners = None
        self._telemetry_cache = None
        self._visual_yaw_reference = None
        self._visual_yaw = float("nan")
        self._yaw_target = None
        self._next_yaw_log = 0.0
        self._frames = self._published = 0
        self._yaw_log_counts = (0, 0, 0)
        self._capture_ms = self._detection_ms = 0.0
        self._frame_size = None
        self._last_vpe_position = np.full(3, np.nan)
        self._last_vpe_at = self._last_vpe_frame_at = self._max_publish_gap = 0.0
        self._last_vpe_attitude_us = 0

        self._ready = asyncio.Event()
        self._fusion_enabled = False
        self._initialize_after = None
        self._frame_ready = asyncio.Event()
        self._valid = 0
        self._pose_covariance = _pose_covariance(
            position_sigma, np.deg2rad(attitude_sigma_deg)
        )

        self._reset_counter = VisualOdometry._next_reset_counter
        VisualOdometry._next_reset_counter = (
            VisualOdometry._next_reset_counter + 1
        ) & 0xFF

    def enable_frame_initialization(self):
        self._initialize_after = time.perf_counter()

    async def wait_frame_ready(self):
        await self._wait_event(self._frame_ready, "marker-frame initialization")

    async def wait_ready(self):
        await self._wait_event(self._ready, "10 observations")

    async def _wait_event(self, event, stage):
        start = time.perf_counter()
        logging.info("VO waiting for %s", stage)
        while not event.is_set():
            try:
                await asyncio.wait_for(event.wait(), 2.0)
            except asyncio.TimeoutError:
                obs, cache = self._latest, self._telemetry_cache
                logging.info(
                    "VO waiting for %s %.1fs: valid=%d last_id=%s obs_age=%.3fs attitude_age=%.3fs position_age=%.3fs",
                    stage, time.perf_counter() - start, self._valid,
                    getattr(obs, "marker_id", None),
                    time.perf_counter() - obs.timestamp if obs is not None else float("inf"),
                    cache.age("attitude_euler") if cache is not None else float("inf"),
                    cache.age("position_velocity_ned") if cache is not None else float("inf"),
                )
        logging.info("VO %s ready after %.2fs", stage, time.perf_counter() - start)

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
        height, width = frame.shape[:2]
        regions = []
        if self._last_corners is not None:
            minimum = self._last_corners.min(axis=0)
            maximum = self._last_corners.max(axis=0)
            margin = np.maximum(maximum - minimum, 32) * 2
            x0 = max(0, int(np.floor(minimum[0] - margin[0])))
            y0 = max(0, int(np.floor(minimum[1] - margin[1])))
            x1 = min(width, int(np.ceil(maximum[0] + margin[0])))
            y1 = min(height, int(np.ceil(maximum[1] + margin[1])))
            if x0 > 0 or y0 > 0 or x1 < width or y1 < height:
                regions.append((frame[y0:y1, x0:x1], x0, y0))
        regions.append((frame, 0, 0))

        corner = None
        for region, offset_x, offset_y in regions:
            corners, ids, _ = self.detector.detectMarkers(region)
            if ids is None:
                continue

            ids = np.asarray(ids).ravel()
            if not len(ids):
                continue
            if self.marker_id is None:
                i = 0
            else:
                matches = np.flatnonzero(ids == self.marker_id)
                if not len(matches):
                    continue
                i = int(matches[0])

            corner = np.asarray(corners[i]).reshape(4, 2).copy()
            corner += (offset_x, offset_y)
            break

        if corner is None:
            return None

        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
            [corner], self.marker_size, self.mtx, self.dist
        )
        self._last_corners = corner.copy()

        return ArucoObservation(
            corner,
            np.asarray(rvecs[0], float).reshape(3),
            np.asarray(tvecs[0], float).reshape(3),
            timestamp,
            int(ids[i]),
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
        pv = await fc.get_drone_ned_position_velocity(drone)
        attitude = self.get_attitude()
        if attitude is None:
            return False

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
        return True

    def _local_body_pose(self, obs, attitude):
        p_body_marker, R_body_marker = self._marker_in_body(obs)
        R_ned_body = _body_to_ned(attitude)

        if self._visual_yaw_reference is not None:
            # Align only the marker's horizontal X direction; retain PX4 roll/pitch.
            marker_axis = R_ned_body @ R_body_marker[:, 0]
            delta = self._visual_yaw_reference - np.arctan2(marker_axis[1], marker_axis[0])
            c, s = np.cos(delta), np.sin(delta)
            R_ned_body = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ R_ned_body
            self._visual_yaw = np.arctan2(R_ned_body[1, 0], R_ned_body[0, 0])

        return (
            self._p_ned_marker - R_ned_body @ p_body_marker,
            self._R_ned_marker @ R_body_marker.T,
        )

    def enable_visual_yaw(self, target_position):
        # Heading is still fixed here; allow a complete detection cycle to anchor.
        obs, attitude = self.get_observation(max_age=0.5), self.get_attitude()
        if obs is None or attitude is None:
            return False
        _, R_body_marker = self._marker_in_body(obs)
        marker_axis = _body_to_ned(attitude) @ R_body_marker[:, 0]
        self._visual_yaw_reference = np.arctan2(marker_axis[1], marker_axis[0])
        self._visual_yaw = np.deg2rad(attitude.yaw_deg)
        self._yaw_target = np.asarray(target_position, float).copy()
        self._yaw_log_counts = (self._frames, self._valid, self._published)
        self._max_publish_gap = 0.0
        self._next_yaw_log = 0.0
        logging.info("Visual yaw anchored: id=%s fc_yaw=%.2f marker_heading=%.2f obs_age=%.3fs",
                     obs.marker_id, attitude.yaw_deg, np.degrees(self._visual_yaw_reference),
                     time.perf_counter() - obs.timestamp)
        return True

    def _log_yaw(self):
        now = time.perf_counter()
        if self._yaw_target is None or now < self._next_yaw_log:
            return
        self._next_yaw_log = now + 1.0
        cache, obs = self._telemetry_cache, self._latest
        pv = cache.latest("position_velocity_ned")
        attitude = self.get_attitude()
        position = ([pv.position.north_m, pv.position.east_m, pv.position.down_m]
                    if pv is not None else [np.nan] * 3)
        velocity = ([pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s]
                    if pv is not None else [np.nan] * 3)
        rpy = ([attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg]
               if attitude is not None else [np.nan] * 3)
        margin = (min(obs.corners.min(), np.min(np.asarray(self._frame_size) - 1 - obs.corners))
                  if obs is not None and self._frame_size is not None else np.nan)
        counts = (self._frames, self._valid, self._published)
        logging.info(
            "Yaw tracking: t=%.3f target=(%.3f,%.3f,%.3f) p=(%.3f,%.3f,%.3f) v=(%.3f,%.3f,%.3f) vpe=(%.3f,%.3f,%.3f) fc_rpy=(%.2f,%.2f,%.2f) visual_yaw=%.2f id=%s margin_px=%.1f obs_age=%.3fs publish_age=%.3fs frames/valid/sent=%d/%d/%d max_publish_gap=%.3fs capture_ms=%.1f detect_ms=%.1f vpe_frame_at=%.3f vpe_attitude_us=%d attitude_age=%.3fs position_age=%.3fs",
            now, *self._yaw_target, *position, *velocity, *self._last_vpe_position, *rpy,
            np.degrees(self._visual_yaw), getattr(obs, "marker_id", None), margin,
            now - obs.timestamp if obs is not None else float("inf"), now - self._last_vpe_at,
            *(a - b for a, b in zip(counts, self._yaw_log_counts)),
            max(self._max_publish_gap, now - self._last_vpe_at), self._capture_ms, self._detection_ms,
            self._last_vpe_frame_at, self._last_vpe_attitude_us,
            cache.age("attitude_euler"), cache.age("position_velocity_ned"),
        )
        self._yaw_log_counts, self._max_publish_gap = counts, 0.0

    def _vpe(self, position, R):
        # PX4's EV-height reference resets absolute altitude to -Z.
        # Convert only the published height; keep planner/marker positions local NED.
        position = np.array(position, dtype=float, copy=True)
        position[2] -= self._origin_altitude_m
        # Position-only EV. To restore attitude output, uncomment the Euler conversion
        # and use the commented AngleBody instead of NaNs.
        # roll, pitch, yaw = _euler321(R)
        return VisionPositionEstimate(
            0,
            PositionBody(*map(float, position)),
            # AngleBody(float(roll), float(pitch), float(yaw)),
            AngleBody(float("nan"), float("nan"), float("nan")),
            self._pose_covariance,
            self._reset_counter,
        )

    async def run(self, drone):
        self._telemetry_cache = getattr(drone, "_telemetry_cache", None)
        owns_cache = self._telemetry_cache is None
        if owns_cache:
            self._telemetry_cache = fc.TelemetryCache(drone)
        try:
            await self._telemetry_cache.start()
            await self._run(drone)
        finally:
            if owns_cache:
                await self._telemetry_cache.close()

    def get_attitude(self, max_age=0.5):
        if self._telemetry_cache is not None:
            return self._telemetry_cache.latest("attitude_euler", max_age)
        return None

    async def _run(self, drone):
        last_frame = None

        while True:
            self._log_yaw()
            capture_start = time.perf_counter()
            frame = self.video.frame()
            frame_time = time.perf_counter()
            self._capture_ms = (frame_time - capture_start) * 1000
            if frame is None or frame is last_frame:
                await asyncio.sleep(0.005)
                continue

            last_frame = frame
            self._frame_size = (frame.shape[1], frame.shape[0])
            obs = await asyncio.to_thread(
                self._detect, frame, frame_time
            )
            self._frames += 1
            self._detection_ms = (time.perf_counter() - frame_time) * 1000

            if obs is None:
                continue

            if self.get_attitude() is None:
                continue
            self._latest = obs
            self._valid += 1
            if self._valid >= 10:
                self._ready.set()

            if self._R_ned_marker is None:
                if (
                    self._initialize_after is None
                    or obs.timestamp < self._initialize_after
                ):
                    continue
                if not await self._initialize_frame(drone, obs):
                    continue
                self._frame_ready.set()



            attitude = self.get_attitude()
            if attitude is None:
                continue
            position, R = self._local_body_pose(obs, attitude)
            
            if self._fusion_enabled:
                await drone.mocap.set_vision_position_estimate(
                    self._vpe(position, R)
                )
                now = time.perf_counter()
                if self._last_vpe_at:
                    self._max_publish_gap = max(self._max_publish_gap, now - self._last_vpe_at)
                self._last_vpe_position = position.copy()
                self._last_vpe_at, self._last_vpe_frame_at = now, obs.timestamp
                self._last_vpe_attitude_us = getattr(attitude, "timestamp_us", 0)
                self._published += 1
    
    def enable_fusion(self):
        self._fusion_enabled = True

    def disable_fusion(self):
        self._fusion_enabled = False
        self._visual_yaw_reference = self._yaw_target = None



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