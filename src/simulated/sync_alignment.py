import time
from math import cos, radians, sin

import cv2
import numpy as np
import ruckig
from mavsdk import System
from mavsdk.offboard import AccelerationNed, PositionNedYaw, VelocityNedYaw

import fc_simulated as fc
import utilities_simulated as utilities
from camera_simulated import my_estimatePoseSingleMarkers

G = 9.81
CAMERA_TO_BODY = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def _rotate_frame(v, heading, to_ned=False):
    x, y = utilities.rotate_vector(v[0], v[1], heading if to_ned else -heading)
    return np.array([x, y, v[2]])


def _body_to_ned(attitude):
    r, p, y = map(radians, (attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg))
    cr, sr, cp, sp, cy, sy = cos(r), sin(r), cos(p), sin(p), cos(y), sin(y)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ])


def _virtual_corners(corners, mtx, attitude):
    yaw = radians(attitude.yaw_deg)
    level_to_ned = np.array([
        [cos(yaw), -sin(yaw), 0.],
        [sin(yaw),  cos(yaw), 0.],
        [0., 0., 1.],
    ])
    R = CAMERA_TO_BODY.T @ level_to_ned.T @ _body_to_ned(attitude) @ CAMERA_TO_BODY
    H = mtx @ R @ np.linalg.inv(mtx)
    q = (H @ np.column_stack((corners, np.ones(4))).T).T
    return None if np.any(q[:, 2] <= 1e-6) else q[:, :2] / q[:, 2, None]


def _observe(frame, detector, mtx, dist, marker_size, marker_id, attitude, heading):
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None

    ids = np.asarray(ids).ravel()
    matches = np.arange(len(ids)) if marker_id is None else np.flatnonzero(ids == marker_id)
    if len(matches) == 0:
        return None
    corner = corners[int(matches[0])]

    virtual = _virtual_corners(np.asarray(corner).reshape(4, 2), mtx, attitude)
    if virtual is None:
        return None

    rvecs, tvecs, _ = my_estimatePoseSingleMarkers([corner], marker_size, mtx, dist)
    rvec, tvec = np.asarray(rvecs[0]).ravel(), np.asarray(tvecs[0]).ravel()
    h = marker_size / 2
    marker_corners = np.array([[-h, -h, 0.], [h, -h, 0.], [h, h, 0.], [-h, h, 0.]])
    R_marker, _ = cv2.Rodrigues(rvec)
    corners_camera = marker_corners @ R_marker.T + tvec
    corners_ned = (_body_to_ned(attitude) @ CAMERA_TO_BODY @ corners_camera.T).T
    corners_aligned = np.array([_rotate_frame(c, heading) for c in corners_ned])
    return virtual, corners_aligned.mean(axis=0), corners_aligned


def _phase1_acceleration_limits(corners, direction, mtx, width, height,
                                reserve, minimum, maximum):
    direction = np.asarray(direction, float)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.full(2, minimum)
    direction /= norm

    fx, fy, cx, cy = mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]
    x, y = (corners[:, 0] - cx) / fx, (corners[:, 1] - cy) / fy
    dx, dy = direction

    rate = np.column_stack((
        fx * ((1 + x*x) * dy - x*y * dx),
        fy * (x*y * dy - (1 + y*y) * dx),
    )) / G

    lower = corners - reserve
    upper = [width - reserve, height - reserve] - corners

    # Already inside the reserve: analytically find the first boundary hit.
    if np.all(lower >= 0) and np.all(upper >= 0):
        margin = np.where(rate >= 0, upper, lower)
        bounds = np.divide(
            margin, np.abs(rate),
            out=np.full_like(rate, np.inf),
            where=np.abs(rate) > 1e-9,
        )
        magnitude = np.clip(bounds.min(), minimum, maximum)
    else:
        # Reserve is soft: still move slowly toward the marker.
        magnitude = minimum

    return np.abs(direction) * magnitude


def _make_trajectory(position, velocity, acceleration, target,
                     max_velocity, max_acceleration, max_jerk):
    dofs = len(position)
    inp, trajectory = ruckig.InputParameter(dofs), ruckig.Trajectory(dofs)
    inp.current_position = np.asarray(position).tolist()
    inp.current_velocity = np.asarray(velocity).tolist()
    inp.current_acceleration = np.asarray(acceleration).tolist()
    inp.target_position = np.asarray(target).tolist()
    inp.target_velocity = inp.target_acceleration = [0.] * dofs
    inp.max_velocity = np.broadcast_to(max_velocity, dofs).tolist()
    inp.max_acceleration = np.broadcast_to(max_acceleration, dofs).tolist()
    inp.max_jerk = np.broadcast_to(max_jerk, dofs).tolist()
    result = ruckig.Ruckig(dofs).calculate(inp, trajectory)
    return trajectory if result in (ruckig.Result.Working, ruckig.Result.Finished) else None


def _body_to_aligned(acceleration):
    down = np.array([-acceleration[0], -acceleration[1], G])
    down /= np.linalg.norm(down)
    right = np.cross(down, [1., 0., 0.])
    if np.linalg.norm(right) < 1e-6:
        return None
    right /= np.linalg.norm(right)
    return np.column_stack((np.cross(right, down), right, down))


def _project(corners, position, acceleration, mtx):
    R = _body_to_aligned(acceleration)
    if R is None:
        return None
    camera = (CAMERA_TO_BODY.T @ R.T @ (corners - position).T).T
    if np.any(camera[:, 2] <= 1e-6):
        return None
    return camera[:, :2] / camera[:, 2, None] * [mtx[0, 0], mtx[1, 1]] + mtx[:2, 2]


def _trajectory_safe(trajectory, marker_corners, mtx, width, height, dt):
    for t in np.r_[np.arange(dt, trajectory.duration, dt), trajectory.duration]:
        p, _, a = map(np.asarray, trajectory.at_time(float(t)))
        pixels = _project(marker_corners, p, a, mtx)
        if pixels is None or np.any(pixels < 0) or np.any(pixels > [width, height]):
            return False
    return True


async def align_to_aruco_ruckig_adaptive(
    drone: System, video, detector, mtx: np.ndarray, dist: np.ndarray,
    fx: float, fy: float, frame_width: int = 1280, frame_height: int = 960,
    marker_size: float = 0.5, marker_id=None, fov_margin: int = 50,
    phase1_minimum_acceleration: float = 0.08,
    phase1_maximum_acceleration: float = 1.5,
    maximum_velocity: float = 3.0, maximum_acceleration: float = 3.0,
    maximum_jerk: float = 2.0, target_altitude: float = 1.0,
    marker_loss_timeout: float = 2.0, trajectory_check_dt: float = 0.02,
) -> bool:
    heading = await fc.get_drone_heading(drone)
    altitude_hold = (await fc.get_drone_ned_position_velocity(drone)).position.down_m
    phase1_trajectory = phase1_start = None
    marker_world = marker_corners_world = loss_start = None

    while True:
        pv = await fc.get_drone_ned_position_velocity(drone)
        attitude = await fc.get_drone_attitude_euler(drone)
        now = time.perf_counter()

        position_ned = np.array([pv.position.north_m, pv.position.east_m, pv.position.down_m])
        velocity_ned = np.array([pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s])
        position, velocity = (_rotate_frame(x, heading) for x in (position_ned, velocity_ned))

        frame = video.frame()
        observation = None if frame is None else _observe(
            frame, detector, mtx, dist, marker_size, marker_id, attitude, heading
        )
        if observation is None:
            loss_start = now if loss_start is None else loss_start
            if now - loss_start > marker_loss_timeout:
                return False
            continue

        loss_start = None
        virtual_corners, marker, relative_corners = observation
        if marker_world is None:
            marker_world, marker_corners_world = position + marker, position + relative_corners

        full_target = marker_world - [0., 0., target_altitude]
        if phase1_trajectory is None:
            full_p, full_v, full_a = position, velocity, np.zeros(3)
        else:
            t = min(now - phase1_start, phase1_trajectory.duration)
            p_ref, v_ref, a_ref = map(np.asarray, phase1_trajectory.at_time(t))
            full_p, full_v, full_a = np.r_[p_ref, altitude_hold], np.r_[v_ref, 0.], np.r_[a_ref, 0.]

        trajectory = _make_trajectory(
            full_p, full_v, full_a, full_target,
            maximum_velocity, maximum_acceleration, maximum_jerk,
        )
        if trajectory is not None and _trajectory_safe(
            trajectory, marker_corners_world, mtx, frame_width, frame_height, trajectory_check_dt
        ):
            break

        if phase1_trajectory is None:
            p_ref, v_ref, a_ref = position[:2], np.zeros(2), np.zeros(2)

        limits = _phase1_acceleration_limits(
            virtual_corners, marker_world[:2] - p_ref, mtx,
            frame_width, frame_height, fov_margin,
            phase1_minimum_acceleration, phase1_maximum_acceleration,
        )
        limits = np.maximum(limits, np.abs(a_ref))

        phase1_trajectory = _make_trajectory(
            p_ref, v_ref, a_ref, marker_world[:2], maximum_velocity, limits, maximum_jerk
        )
        if phase1_trajectory is None:
            return False

        phase1_start = now
        p, v, a = map(np.asarray, phase1_trajectory.at_time(
            min(time.perf_counter() - phase1_start, phase1_trajectory.duration)))
        p = _rotate_frame(np.r_[p, altitude_hold], heading, True)
        v, a = (_rotate_frame(np.r_[x, np.nan], heading, True) for x in (v, a))

        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(*map(float, p), heading),
            VelocityNedYaw(*map(float, v), heading),
            AccelerationNed(*map(float, a)),
        )

    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        p, v, a = trajectory.at_time(min(elapsed, trajectory.duration))
        p, v, a = (_rotate_frame(np.asarray(x), heading, True) for x in (p, v, a))
        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(*map(float, p), heading),
            VelocityNedYaw(*map(float, v), heading),
            AccelerationNed(*map(float, a)),
        )
        if elapsed > trajectory.duration:
            return True
