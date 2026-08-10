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


def _rotate_frame(vector, heading, to_ned=False):
    angle = heading if to_ned else -heading
    xy = utilities.rotate_vector(vector[0], vector[1], angle)
    return np.array([xy[0], xy[1], vector[2]], dtype=float)


def _body_to_ned(attitude):
    roll, pitch, yaw = map(radians, (
        attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg,
    ))
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
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


def _virtual_corners(corners_px, mtx, attitude):
    body_to_ned = _body_to_ned(attitude)
    level_body_to_ned = _yaw_rotation(attitude.yaw_deg)

    camera_to_virtual = (
        CAMERA_TO_BODY.T
        @ level_body_to_ned.T
        @ body_to_ned
        @ CAMERA_TO_BODY
    )
    homography = mtx @ camera_to_virtual @ np.linalg.inv(mtx)

    pixels = np.column_stack((corners_px, np.ones(len(corners_px))))
    projected = (homography @ pixels.T).T
    if np.any(projected[:, 2] <= 1e-6):
        return None
    return projected[:, :2] / projected[:, 2, None]


def _observe(frame, detector, mtx, dist, marker_size, marker_id,
             attitude, heading):
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
    virtual = _virtual_corners(corners_px, mtx, attitude)
    if virtual is None:
        return None

    _, tvecs, _ = my_estimatePoseSingleMarkers(
        [corners[index]], marker_size, mtx, dist,
    )
    tvec = np.asarray(tvecs[0]).reshape(3)

    # OpenCV camera [right, image-down, optical] -> body FRD.
    marker_body = np.array([-tvec[1], tvec[0], tvec[2]])
    marker_ned = _body_to_ned(attitude) @ marker_body
    marker_aligned = _rotate_frame(marker_ned, heading)
    return virtual, marker_aligned


def _margin(corners, width, height, reserve):
    center = np.array([width / 2.0, height / 2.0])
    limits = center - reserve
    used = np.max(np.abs(corners - center), axis=0)
    return float(np.min(np.clip((limits - used) / limits, 0.0, 1.0)))


def _phase1_acceleration(marker, corners, fx, fy, width, height,
                         reserve, minimum, maximum, scale):
    center = np.array([width / 2.0, height / 2.0])
    available = np.maximum(
        0.0,
        center - reserve - np.max(np.abs(corners - center), axis=0),
    )

    # Forward acceleration creates pitch (vertical pixels); right creates roll.
    axis_limits = np.array([
        np.clip(G * available[1] / fy, minimum, maximum),
        np.clip(G * available[0] / fx, minimum, maximum),
    ])

    direction = np.asarray(marker[:2], dtype=float)
    distance = np.linalg.norm(direction)
    if distance < 1e-6:
        return np.zeros(3)
    direction /= distance

    magnitude = maximum
    for axis in range(2):
        if abs(direction[axis]) > 1e-6:
            magnitude = min(magnitude, axis_limits[axis] / abs(direction[axis]))

    return np.array([*(direction * magnitude * scale), 0.0])


def _velocity_compensated_margin(marker, velocity, corners, fx, fy, width,
                                 height, reserve, lookahead):
    current_margin = _margin(corners, width, height, reserve)
    distance = float(np.linalg.norm(marker[:2]))
    if distance < 1e-6:
        return current_margin

    direction = marker[:2] / distance
    closing_speed = float(np.dot(velocity[:2], direction))
    if closing_speed <= 0.0:
        return current_margin

    horizon = min(lookahead, distance / closing_speed)
    predicted = marker - velocity * horizon
    if predicted[2] <= 0.05:
        return current_margin

    center = np.array([width / 2.0, height / 2.0])
    current_center = center + np.array([
        fx * marker[1] / marker[2],
        -fy * marker[0] / marker[2],
    ])
    predicted_center = center + np.array([
        fx * predicted[1] / predicted[2],
        -fy * predicted[0] / predicted[2],
    ])
    predicted_corners = corners + predicted_center - current_center
    predicted_margin = _margin(predicted_corners, width, height, reserve)
    return max(current_margin, predicted_margin)


def _make_trajectory(position, velocity, acceleration, target,
                     max_velocity, max_acceleration, max_jerk):
    inp = ruckig.InputParameter(3)
    trajectory = ruckig.Trajectory(3)
    inp.current_position = position.tolist()
    inp.current_velocity = velocity.tolist()
    inp.current_acceleration = acceleration.tolist()
    inp.target_position = target.tolist()
    inp.target_velocity = [0.0] * 3
    inp.target_acceleration = [0.0] * 3
    inp.max_velocity = [max_velocity] * 3
    inp.max_acceleration = [max_acceleration] * 3
    inp.max_jerk = [max_jerk] * 3
    
    result = ruckig.Ruckig(3).calculate(inp, trajectory)
    if result in (ruckig.Result.Working, ruckig.Result.Finished):
        return trajectory
    return None

async def reset_xy_controller(drone, down, heading):
    nan = float("nan")

    await drone.offboard.set_position_velocity_acceleration_ned(
        PositionNedYaw(nan, nan, float(down), heading),
        VelocityNedYaw(nan, nan, nan, heading),
        AccelerationNed(0.0, 0.0, nan),
    )

    # Wait for fresh telemetry, not a fixed sleep.
    await fc.get_drone_ned_position_velocity(drone)
async def align_to_aruco_visual_hybrid(
    drone: System, video, detector, mtx: np.ndarray, dist: np.ndarray,
    fx: float, fy: float, frame_width: int = 1280, frame_height: int = 960,
    marker_size: float = 0.5, marker_id=None, fov_margin: int = 50,
    release_margin_fraction: float = 0.25,
    phase1_minimum_acceleration: float = 0.08,
    phase1_maximum_acceleration: float = 1.5,
    phase1_acceleration_scale: float = 1,
    release_velocity_lookahead: float = 1.2,
    maximum_velocity: float = 3.0,
    maximum_acceleration: float = 3.0,
    maximum_jerk: float = 2.0,
    target_altitude: float = 1.0,
    marker_loss_timeout: float = 2.0,
) -> bool:
    heading = await fc.get_drone_heading(drone)
    initial = await fc.get_drone_ned_position_velocity(drone)
    altitude_hold = float(initial.position.down_m)
    phase1_acceleration = None
    loss_start = None

    while True:
        pv = await fc.get_drone_ned_position_velocity(drone)
        attitude = await fc.get_drone_attitude_euler(drone)
        now = time.perf_counter()

        position_ned = np.array([
            pv.position.north_m, pv.position.east_m, pv.position.down_m,
        ])
        velocity_ned = np.array([
            pv.velocity.north_m_s, pv.velocity.east_m_s,
            pv.velocity.down_m_s,
        ])
        position = _rotate_frame(position_ned, heading)
        velocity = _rotate_frame(velocity_ned, heading)

        frame = video.frame()
        observation = None if frame is None else _observe(
            frame, detector, mtx, dist, marker_size, marker_id,
            attitude, heading,
        )

        if observation is None:
            loss_start = now if loss_start is None else loss_start
            await drone.offboard.set_position_ned(PositionNedYaw(
                *map(float, position_ned), heading,
            ))
            if now - loss_start > marker_loss_timeout:
                return False
            continue

        loss_start = None
        virtual_corners, marker = observation
        release_margin = _velocity_compensated_margin(
            marker, velocity, virtual_corners, fx, fy,
            frame_width, frame_height, fov_margin,
            release_velocity_lookahead,
        )

        if release_margin < release_margin_fraction:
            phase1_acceleration = _phase1_acceleration(
                marker, virtual_corners, fx, fy,
                frame_width, frame_height, fov_margin,
                phase1_minimum_acceleration, phase1_maximum_acceleration,
                phase1_acceleration_scale,
            )
            acceleration_ned = _rotate_frame(
                phase1_acceleration, heading, to_ned=True,
            )
            nan = float("nan")
            await drone.offboard.set_position_velocity_acceleration_ned(
                PositionNedYaw(nan, nan, altitude_hold, heading),
                VelocityNedYaw(nan, nan, nan, heading),
                AccelerationNed(
                    float(acceleration_ned[0]),
                    float(acceleration_ned[1]),
                    nan,
                ),
            )
            continue

        initial_acceleration = (
            np.zeros(3) if phase1_acceleration is None else phase1_acceleration
        )
        #velocity = np.zeros(3) if phase1_acceleration is None else velocity
        target = position + marker - np.array([0.0, 0.0, target_altitude])
        print(velocity)
        trajectory = _make_trajectory(
            position, velocity, initial_acceleration, target,
            maximum_velocity, maximum_acceleration, maximum_jerk,
        )
        if trajectory is None:
            return False
        break
    print("end phase 1, start phase 2", "horizontal error x,y:", marker[0], marker[1])
    print("tilt:", attitude.roll_deg, attitude.pitch_deg, "yaw:", attitude.yaw_deg)

    start = time.perf_counter()
    
    while True:
        elapsed = time.perf_counter() - start
        p, v, a = trajectory.at_time(min(elapsed, trajectory.duration))
        p = _rotate_frame(np.asarray(p), heading, to_ned=True)
        v = _rotate_frame(np.asarray(v), heading, to_ned=True)
        a = _rotate_frame(np.asarray(a), heading, to_ned=True)

        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(*map(float, p), heading),
            VelocityNedYaw(*map(float, v), heading),
            AccelerationNed(*map(float, a)),
        )
        if elapsed > trajectory.duration:
            print("last setpoints:", p, v, a)
            return True
