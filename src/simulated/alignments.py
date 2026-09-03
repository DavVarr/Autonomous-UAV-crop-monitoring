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
        [sin(yaw), cos(yaw), 0.],
        [0., 0., 1.],
    ])
    rotation = CAMERA_TO_BODY.T @ level_to_ned.T @ _body_to_ned(attitude) @ CAMERA_TO_BODY
    H = mtx @ rotation @ np.linalg.inv(mtx)
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
    object_corners = np.array([[-h, -h, 0.], [h, -h, 0.], [h, h, 0.], [-h, h, 0.]])

    R_marker, _ = cv2.Rodrigues(rvec)
    corners_camera = object_corners @ R_marker.T + tvec
    corners_ned = (_body_to_ned(attitude) @ CAMERA_TO_BODY @ corners_camera.T).T
    corners_aligned = np.array([_rotate_frame(c, heading) for c in corners_ned])

    return virtual, corners_aligned.mean(axis=0), corners_aligned


def _phase1_acceleration(marker, corners, fx, fy, width, height,
                         reserve, minimum, maximum, scale):
    center = np.array([width, height]) / 2
    available = np.maximum(0., center - reserve - np.max(np.abs(corners - center), axis=0))
    limits = np.clip(G * available[::-1] / [fy, fx], minimum, maximum)

    direction = marker[:2]
    distance = np.linalg.norm(direction)
    if distance < 1e-6:
        return np.zeros(3)
    direction = direction / distance

    active = np.abs(direction) > 1e-6
    magnitude = min(maximum, np.min(limits[active] / np.abs(direction[active])))
    return np.r_[direction * magnitude * scale, 0.]


def _make_trajectory(position, velocity, acceleration, target,
                     max_velocity, max_acceleration, max_jerk):
    inp, trajectory = ruckig.InputParameter(3), ruckig.Trajectory(3)
    inp.current_position = position.tolist()
    inp.current_velocity = velocity.tolist()
    inp.current_acceleration = acceleration.tolist()
    inp.target_position = target.tolist()
    inp.target_velocity = inp.target_acceleration = [0.] * 3
    inp.max_velocity = [max_velocity] * 3
    inp.max_acceleration = [max_acceleration] * 3
    inp.max_jerk = [max_jerk] * 3

    result = ruckig.Ruckig(3).calculate(inp, trajectory)
    return trajectory if result in (ruckig.Result.Working, ruckig.Result.Finished) else None


def _body_to_aligned(acceleration):
    down = np.array([-acceleration[0], -acceleration[1], G - acceleration[2]])
    down /= np.linalg.norm(down)
    right = np.cross(down, [1., 0., 0.])
    if np.linalg.norm(right) < 1e-6:
        return None
    right /= np.linalg.norm(right)
    return np.column_stack((np.cross(right, down), right, down))


def _trajectory_safe(trajectory, marker_corners, mtx, width, height, dt):
    focal, center = np.array([mtx[0, 0], mtx[1, 1]]), mtx[:2, 2]
    times = np.r_[np.arange(dt, trajectory.duration, dt), trajectory.duration]

    for t in times:
        p, _, a = map(np.asarray, trajectory.at_time(float(t)))
        R = _body_to_aligned(a)
        if R is None:
            return False

        camera = (CAMERA_TO_BODY.T @ R.T @ (marker_corners - p).T).T
        if np.any(camera[:, 2] <= 1e-6):
            return False

        pixels = camera[:, :2] / camera[:, 2, None] * focal + center
        if np.any(pixels < 0) or np.any(pixels > [width, height]):
            return False

    return True


async def align_to_aruco_visual_hybrid(
    drone: System, video, detector, mtx: np.ndarray, dist: np.ndarray,
    fx: float, fy: float, frame_width: int = 1280, frame_height: int = 960,
    marker_size: float = 0.5, marker_id=None, fov_margin: int = 50,
    phase1_minimum_acceleration: float = 0.08,
    phase1_maximum_acceleration: float = 1.5,
    phase1_acceleration_scale: float = 1,
    maximum_velocity: float = 3.0,
    maximum_acceleration: float = 3.0,
    maximum_jerk: float = 4.0,
    target_altitude: float = 1.0,
    marker_loss_timeout: float = 2.0,
    trajectory_check_dt: float = 0.02,
) -> bool:
    heading = await fc.get_drone_heading(drone)
    initial = await fc.get_drone_ned_position_velocity(drone)
    altitude_hold = initial.position.down_m
    phase1_acceleration = None
    loss_start = None

    while True:
        pv = await fc.get_drone_ned_position_velocity(drone)
        attitude = await fc.get_drone_attitude_euler(drone)
        now = time.perf_counter()

        position_ned = np.array([pv.position.north_m, pv.position.east_m, pv.position.down_m])
        velocity_ned = np.array([pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s])
        position = _rotate_frame(position_ned, heading)
        velocity = _rotate_frame(velocity_ned, heading)

        frame = video.frame()
        observation = None if frame is None else _observe(
            frame, detector, mtx, dist, marker_size, marker_id, attitude, heading
        )
        if observation is None:
            loss_start = now if loss_start is None else loss_start
            await drone.offboard.set_position_ned(PositionNedYaw(*map(float, position_ned), heading))
            if now - loss_start > marker_loss_timeout:
                return False
            continue

        loss_start = None
        virtual_corners, marker, relative_corners = observation
        initial_acceleration = np.zeros(3) if phase1_acceleration is None else phase1_acceleration
        target = position + marker - [0., 0., target_altitude]
        trajectory = _make_trajectory(
            position, velocity, initial_acceleration, target,
            maximum_velocity, maximum_acceleration, maximum_jerk,
        )

        if trajectory is not None and _trajectory_safe(
            trajectory, position + relative_corners, mtx,
            frame_width, frame_height, trajectory_check_dt,
        ):
            break

        phase1_acceleration = _phase1_acceleration(
            marker, virtual_corners, fx, fy, frame_width, frame_height, fov_margin,
            phase1_minimum_acceleration, phase1_maximum_acceleration,
            phase1_acceleration_scale,
        )
        acceleration_ned = _rotate_frame(phase1_acceleration, heading, True)
        nan = float("nan")
        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(nan, nan, altitude_hold, heading),
            VelocityNedYaw(nan, nan, nan, heading),
            AccelerationNed(float(acceleration_ned[0]), float(acceleration_ned[1]), nan),
        )

    print("end phase 1, start phase 2", "horizontal error x,y:", marker[0], marker[1])
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        p, v, a = trajectory.at_time(min(elapsed, trajectory.duration))
        p, v, a = (_rotate_frame(np.asarray(x), heading, True) for x in (p, v, a))

        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(*map(float, p), heading),
            VelocityNedYaw(*map(float, v), heading), AccelerationNed(*map(float, a))
        )
        if elapsed > trajectory.duration:
            print("last setpoints:", p, v, a)
            return True