import time
from math import cos, radians, sin

import numpy as np
import ruckig
from mavsdk import System
from mavsdk.offboard import AccelerationNed, PositionNedYaw, VelocityNedYaw

import fc_simulated as fc
import utilities_simulated as utilities
from camera_simulated import my_estimatePoseSingleMarkers

G = 9.81
CAMERA_TO_BODY = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
])


def _aligned(vector, heading, to_ned=False):
    angle = heading if to_ned else -heading
    xy = utilities.rotate_vector(vector[0], vector[1], angle)
    return np.array([xy[0], xy[1], vector[2]], dtype=float)


def _body_to_ned(attitude):
    r, p, y = map(radians, (
        attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg,
    ))
    cr, sr = cos(r), sin(r)
    cp, sp = cos(p), sin(p)
    cy, sy = cos(y), sin(y)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp,     sr * cp,                cr * cp],
    ])


def _yaw_rotation(yaw_deg):
    yaw = radians(yaw_deg)
    return np.array([
        [cos(yaw), -sin(yaw), 0.0],
        [sin(yaw),  cos(yaw), 0.0],
        [0.0,       0.0,      1.0],
    ])


def _virtual_corners(corners, mtx, mtx_inv, attitude):
    actual_body_to_ned = _body_to_ned(attitude)
    level_body_to_ned = _yaw_rotation(attitude.yaw_deg)
    camera_to_virtual = (
        CAMERA_TO_BODY.T
        @ level_body_to_ned.T
        @ actual_body_to_ned
        @ CAMERA_TO_BODY
    )
    homography = mtx @ camera_to_virtual @ mtx_inv
    pixels = np.column_stack((corners, np.ones(4)))
    projected = (homography @ pixels.T).T
    if np.any(projected[:, 2] <= 1e-6):
        return None
    return projected[:, :2] / projected[:, 2, None]


def _observe(frame, detector, mtx, mtx_inv, dist, marker_size,
             marker_id, attitude, heading):
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None

    ids = np.asarray(ids).reshape(-1)
    if marker_id is None:
        index = 0
    else:
        matches = np.flatnonzero(ids == marker_id)
        if len(matches) == 0:
            return None
        index = int(matches[0])

    corners_px = np.asarray(corners[index]).reshape(4, 2)
    virtual = _virtual_corners(corners_px, mtx, mtx_inv, attitude)
    if virtual is None:
        return None

    _, tvecs, _ = my_estimatePoseSingleMarkers(
        [corners[index]], marker_size, mtx, dist,
    )
    tvec = np.asarray(tvecs[0]).reshape(3)
    marker_body = np.array([-tvec[1], tvec[0], tvec[2]])
    marker_ned = _body_to_ned(attitude) @ marker_body
    return virtual, _aligned(marker_ned, heading)


def _horizontal_acceleration(corners, fx, fy, width, height, reserve,
                             scale, minimum, maximum):
    center = np.array([width / 2.0, height / 2.0])
    available = np.maximum(
        0.0,
        center - reserve - np.max(np.abs(corners - center), axis=0),
    )
    # Forward acceleration uses vertical pixels; right acceleration uses X.
    limits = scale * G * np.array([available[1] / fy, available[0] / fx])
    return np.clip(limits, minimum, maximum)


def _down_velocity(marker, velocity, corners, fx, fy, width, height,
                   reserve, margin_time, maximum):
    """First-order FOV constraint derived from q_dot.

    q_dot = (-closing_speed + q * down_velocity) / height
    q_dot <= (q_limit - q) / margin_time
    """
    cx, cy = width / 2.0, height / 2.0
    h = max(float(marker[2]), 0.05)

    qx = float(np.max(np.abs(corners[:, 0] - cx)) / fx)
    qy = float(np.max(np.abs(corners[:, 1] - cy)) / fy)
    qx_limit = (cx - reserve) / fx
    qy_limit = (cy - reserve) / fy

    closing_x = float(np.sign(marker[1]) * velocity[1])
    closing_y = float(np.sign(marker[0]) * velocity[0])

    vx = (closing_x + h * (qx_limit - qx) / margin_time) / max(qx, 1e-3)
    vy = (closing_y + h * (qy_limit - qy) / margin_time) / max(qy, 1e-3)
    return float(np.clip(min(vx, vy), 0.0, maximum))


def _trajectory(position, velocity, acceleration, target,
                max_xy_velocity, max_down_velocity, max_climb_velocity,
                max_xy_acceleration, max_z_acceleration, max_jerk):
    inp = ruckig.InputParameter(3)
    traj = ruckig.Trajectory(3)

    inp.current_position = position.tolist()
    inp.current_velocity = velocity.tolist()
    inp.current_acceleration = acceleration.tolist()
    inp.target_position = target.tolist()
    inp.target_velocity = [0.0, 0.0, 0.0]
    inp.target_acceleration = [0.0, 0.0, 0.0]

    xy_v = np.maximum(max_xy_velocity, np.abs(velocity[:2]) + 1e-3)
    xy_a = np.maximum(max_xy_acceleration, np.abs(acceleration[:2]) + 1e-3)
    down_v = max(max_down_velocity, max(0.0, velocity[2]) + 1e-3)
    z_a = max(max_z_acceleration, abs(acceleration[2]) + 1e-3)

    inp.max_velocity = [float(xy_v[0]), float(xy_v[1]), float(down_v)]
    inp.min_velocity = [
        -float(max_xy_velocity),
        -float(max_xy_velocity),
        -float(max_climb_velocity),
    ]
    inp.max_acceleration = [float(xy_a[0]), float(xy_a[1]), float(z_a)]
    inp.max_jerk = [float(max_jerk)] * 3

    inp.synchronization = ruckig.Synchronization.Time
    inp.per_dof_synchronization = [
        ruckig.Synchronization.Phase,
        ruckig.Synchronization.Phase,
        ruckig.Synchronization.No,
    ]

    result = ruckig.Ruckig(3).calculate(inp, traj)
    if result in (ruckig.Result.Working, ruckig.Result.Finished):
        return traj
    return None


async def align_to_aruco_ruckig_adaptive(
    drone: System,
    video,
    detector,
    mtx: np.ndarray,
    dist: np.ndarray,
    fx: float,
    fy: float,
    frame_width: int = 1280,
    frame_height: int = 960,
    marker_size: float = 0.5,
    marker_id=None,
    fov_margin: int = 50,
    horizontal_acceleration_scale: float = 0.8,
    minimum_horizontal_acceleration: float = 0.10,
    maximum_horizontal_acceleration: float = 3.0,
    maximum_horizontal_velocity: float = 3.0,
    maximum_down_velocity: float = 2.0,
    maximum_climb_velocity: float = 1.0,
    maximum_vertical_acceleration: float = 3.0,
    maximum_jerk: float = 2.0,
    margin_time: float = 1.0,
    target_altitude: float = 1.0,
    position_tolerance: float = 0.10,
    velocity_tolerance: float = 0.10,
    marker_loss_timeout: float = 2.0,
) -> bool:
    """Continuously replan to the real final target using virtual FOV limits."""
    heading = await fc.get_drone_heading(drone)
    mtx_inv = np.linalg.inv(mtx)
    last_acceleration = np.zeros(3)
    loss_start = None

    while True:
        pv = await fc.get_drone_ned_position_velocity(drone)
        attitude = await fc.get_drone_attitude_euler(drone)
        state_time = time.perf_counter()

        position_ned = np.array([
            pv.position.north_m, pv.position.east_m, pv.position.down_m,
        ])
        velocity_ned = np.array([
            pv.velocity.north_m_s,
            pv.velocity.east_m_s,
            pv.velocity.down_m_s,
        ])
        position = _aligned(position_ned, heading)
        velocity = _aligned(velocity_ned, heading)

        frame = video.frame()
        observation = None if frame is None else _observe(
            frame, detector, mtx, mtx_inv, dist, marker_size,
            marker_id, attitude, heading,
        )

        if observation is None:
            now = time.perf_counter()
            loss_start = now if loss_start is None else loss_start
            await drone.offboard.set_position_ned(PositionNedYaw(
                float(position_ned[0]), float(position_ned[1]),
                float(position_ned[2]), heading,
            ))
            last_acceleration[:] = 0.0
            if now - loss_start > marker_loss_timeout:
                return False
            continue

        loss_start = None
        virtual_corners, marker = observation
        error = np.array([marker[0], marker[1], marker[2] - target_altitude])

        if (
            np.max(np.abs(error)) <= position_tolerance
            and np.linalg.norm(velocity) <= velocity_tolerance
        ):
            await drone.offboard.set_position_ned(PositionNedYaw(
                float(position_ned[0]), float(position_ned[1]),
                float(position_ned[2]), heading,
            ))
            return True

        max_xy_acceleration = _horizontal_acceleration(
            virtual_corners, fx, fy, frame_width, frame_height, fov_margin,
            horizontal_acceleration_scale, minimum_horizontal_acceleration,
            maximum_horizontal_acceleration,
        )
        max_down_velocity_now = _down_velocity(
            marker, velocity, virtual_corners, fx, fy,
            frame_width, frame_height, fov_margin,
            margin_time, maximum_down_velocity,
        )

        target = position + marker - np.array([0.0, 0.0, target_altitude])
        traj = _trajectory(
            position, velocity, last_acceleration, target,
            maximum_horizontal_velocity, max_down_velocity_now,
            maximum_climb_velocity, max_xy_acceleration,
            maximum_vertical_acceleration, maximum_jerk,
        )
        if traj is None:
            print(
                "Ruckig failed: "
                f"a_xy={max_xy_acceleration}, "
                f"v_down={max_down_velocity_now:.2f}"
            )
            return False

        sample_time = min(time.perf_counter() - state_time, traj.duration)
        p_sp, v_sp, a_sp = map(np.asarray, traj.at_time(sample_time))
        last_acceleration = a_sp.copy()

        p_ned = _aligned(p_sp, heading, to_ned=True)
        v_ned = _aligned(v_sp, heading, to_ned=True)
        a_ned = _aligned(a_sp, heading, to_ned=True)

        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(
                float(p_ned[0]), float(p_ned[1]), float(p_ned[2]), heading,
            ),
            VelocityNedYaw(
                float(v_ned[0]), float(v_ned[1]), float(v_ned[2]), heading,
            ),
            AccelerationNed(
                float(a_ned[0]), float(a_ned[1]), float(a_ned[2]),
            ),
        )


