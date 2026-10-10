import asyncio
import time
from math import cos, radians, sin

import cv2
import numpy as np
import ruckig
from mavsdk import System
from mavsdk.offboard import AccelerationNed, PositionNedYaw, VelocityNedYaw

import fc_simulated as fc
import utilities_simulated as utilities
from VisualOdometry import VisualOdometry

G = 9.81


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


def _virtual_corners(corners, mtx, attitude, camera_to_body):
    yaw = radians(attitude.yaw_deg)
    level_to_ned = np.array([
        [cos(yaw), -sin(yaw), 0.],
        [sin(yaw),  cos(yaw), 0.],
        [0., 0., 1.],
    ])
    R = camera_to_body.T @ level_to_ned.T @ _body_to_ned(attitude) @ camera_to_body
    H = mtx @ R @ np.linalg.inv(mtx)
    q = (H @ np.column_stack((corners, np.ones(4))).T).T
    return None if np.any(q[:, 2] <= 1e-6) else q[:, :2] / q[:, 2, None]


def _observe(obs, mtx, marker_size, attitude, heading, camera_to_body):
    if obs is None:
        return None
    virtual = _virtual_corners(obs.corners, mtx, attitude, camera_to_body)
    if virtual is None:
        return None

    h = marker_size / 2
    marker_corners = np.array([[-h, -h, 0.], [h, -h, 0.], [h, h, 0.], [-h, h, 0.]])
    R_marker, _ = cv2.Rodrigues(obs.rvec)
    corners_camera = marker_corners @ R_marker.T + obs.tvec
    corners_ned = (_body_to_ned(attitude) @ camera_to_body @ corners_camera.T).T
    corners_aligned = np.array([_rotate_frame(c, heading) for c in corners_ned])
    return virtual, corners_aligned.mean(axis=0), corners_aligned


def _boundary_angle(point, direction, normal, camera_to_body, limit):
    # Minimal tilt producing acceleration along direction.
    axis = np.array([direction[1], -direction[0], 0.])
    q = camera_to_body @ normal

    pk, kq = point @ axis, axis @ q
    A = point @ q - pk * kq
    B = point @ np.cross(axis, q)
    D = pk * kq

    r = np.hypot(A, B)
    if r < 1e-12 or abs(D) > r + 1e-12:
        return None

    phi = np.arctan2(B, A)
    delta = np.arccos(np.clip(-D / r, -1., 1.))
    roots = np.mod([phi - delta, phi + delta], 2 * np.pi)
    roots = roots[(roots > 1e-9) & (roots <= limit + 1e-9)]
    return None if not len(roots) else roots.min()

def _phase1_acceleration_limits(corners, direction, mtx, camera_to_body,
                                width, height, reserve, minimum, maximum):
    direction = np.asarray(direction, float)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.full(2, minimum)
    direction /= norm

    fx, fy, cx, cy = mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]

    # corners are 3-D vectors in the level heading-aligned frame.
    # Project them into a hypothetical level camera. This is the 3-D
    # equivalent of the previous virtual-corner compensation.
    camera = (camera_to_body.T @ corners.T).T
    if np.any(camera[:, 2] <= 1e-6):
        return np.abs(direction) * minimum
    pixels = camera[:, :2] / camera[:, 2, None] * [fx, fy] + [cx, cy]

    lower = pixels - reserve
    upper = [width - reserve, height - reserve] - pixels

    if np.all(lower >= 0) and np.all(upper >= 0):
        boundaries = [
            np.array([1., 0., -(reserve - cx) / fx]),
            np.array([1., 0., -((width - reserve) - cx) / fx]),
            np.array([0., 1., -(reserve - cy) / fy]),
            np.array([0., 1., -((height - reserve) - cy) / fy]),
        ]

        theta_max = np.arctan(maximum / G)
        for point in corners:
            for normal in boundaries:
                theta = _boundary_angle(
                    point, direction, normal, camera_to_body, theta_max)
                if theta is not None:
                    theta_max = min(theta_max, theta)

        magnitude = np.clip(G * np.tan(theta_max), minimum, maximum)
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


def _project(corners, position, acceleration, mtx, camera_to_body):
    R = _body_to_aligned(acceleration)
    if R is None:
        return None
    camera = (camera_to_body.T @ R.T @ (corners - position).T).T
    if np.any(camera[:, 2] <= 1e-6):
        return None
    return camera[:, :2] / camera[:, 2, None] * [mtx[0, 0], mtx[1, 1]] + mtx[:2, 2]


def _trajectory_safe(trajectory, marker_corners, mtx, camera_to_body, width, height, dt,margin):
    for t in np.r_[np.arange(dt, trajectory.duration, dt), trajectory.duration]:
        p, _, a = map(np.asarray, trajectory.at_time(float(t)))
        pixels = _project(marker_corners, p, a, mtx, camera_to_body)
        if pixels is None or np.any(pixels < margin) or np.any(pixels > [width - margin, height - margin]):
            return False
    return True

def _wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

async def align_to_aruco_ruckig_adaptive(
    drone: System, vision: VisualOdometry,
    frame_width=1280, frame_height=960, fov_margin=30, initial_state = None,
    phase1_minimum_acceleration=0.08, phase1_maximum_acceleration=1.5,
    maximum_velocity=3.0, maximum_acceleration=3.0, maximum_jerk=2.0,
    target_altitude=1.0, marker_loss_timeout=2.0, trajectory_check_dt=0.02,
) -> bool:
    mtx, C = vision.mtx, vision.camera_to_body
    heading = await fc.get_drone_heading(drone)
    altitude_hold = (await fc.get_drone_ned_position_velocity(drone)).position.down_m
    phase1_trajectory = phase1_start = None
    marker_ned = vision.marker_position_ned
    marker_corners_ned = vision.marker_corners_ned
    if marker_ned is None or marker_corners_ned is None:
        return False

    marker_world = _rotate_frame(marker_ned, heading)
    marker_corners_world = np.array([
        _rotate_frame(c, heading) for c in marker_corners_ned
    ])

    if initial_state:
        handoff_p = _rotate_frame(
            np.array([initial_state["p"][0], initial_state["p"][1], 0.]),
            heading
        )[:2]
        handoff_v = _rotate_frame(
            np.array([initial_state["v"][0], initial_state["v"][1], 0.]),
            heading
        )[:2]
        handoff_a = _rotate_frame(
            np.array([initial_state["a"][0], initial_state["a"][1], 0.]),
            heading
        )[:2]


    loss_start = None

    while True:
        pv = await fc.get_drone_ned_position_velocity(drone)
        attitude = await fc.get_drone_attitude_euler(drone)
        now = time.perf_counter()

        position_ned = np.array([pv.position.north_m, pv.position.east_m, pv.position.down_m])
        velocity_ned = np.array([pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s])
        position, velocity = (_rotate_frame(x, heading) for x in (position_ned, velocity_ned))

        observation = _observe(
            vision.get_observation(), mtx, vision.marker_size, attitude, heading, C)
        if observation is None:
            loss_start = now if loss_start is None else loss_start
            if now - loss_start > marker_loss_timeout:
                return False
            continue

        loss_start = None
        virtual_corners, _, relative_corners= observation

        full_target = marker_world - [0., 0., target_altitude]
        if phase1_trajectory is None:
            if initial_state:
                full_p = np.r_[handoff_p, position[2]]
                full_v = np.r_[handoff_v, 0.]
                full_a = np.r_[handoff_a, 0.]
            else:
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
            trajectory, marker_corners_world, mtx, C, frame_width, frame_height, trajectory_check_dt,fov_margin):
            break

        if phase1_trajectory is None:
            p_ref, v_ref, a_ref = full_p[:2], full_v[:2], full_a[:2]

        limits = _phase1_acceleration_limits(
            relative_corners, marker_world[:2] - p_ref, mtx, C,
            frame_width, frame_height, fov_margin,
            phase1_minimum_acceleration, phase1_maximum_acceleration,
        )
        limits = np.maximum(limits, np.abs(a_ref))

        phase1_trajectory = _make_trajectory(
            p_ref, v_ref, a_ref, marker_world[:2], maximum_velocity, limits, maximum_jerk
        )
        if phase1_trajectory is None:
            return False
        #print("phase1")
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
    #print("phase2")
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
            break


    marker_forward = marker_corners_ned[0] - marker_corners_ned[3]
    target_yaw = np.arctan2(marker_forward[1], marker_forward[0])

    attitude = await fc.get_drone_attitude_euler(drone)
    yaw_setpoint = np.deg2rad(attitude.yaw_deg)

    max_yaw_rate = np.deg2rad(85.)
    yaw_gain = 2.0                   # 1/s
    tolerance = np.deg2rad(1.)

    last = time.perf_counter()
    final_position = _rotate_frame(full_target, heading, True)
    while True:
        now = time.perf_counter()
        dt = now - last
        last = now

        error = _wrap_pi(target_yaw - yaw_setpoint)
        yaw_rate = np.clip(yaw_gain * error, -max_yaw_rate, max_yaw_rate)

        # Never step past the target.
        step = np.sign(error) * min(abs(yaw_rate) * dt, abs(error))
        yaw_setpoint += step

        await drone.offboard.set_position_ned(
            PositionNedYaw(
                *map(float, final_position), np.degrees(_wrap_pi(yaw_setpoint))))

        attitude = await fc.get_drone_attitude_euler(drone)
        actual_error = _wrap_pi(
            target_yaw - np.deg2rad(attitude.yaw_deg)
        )

        if abs(actual_error) < tolerance and abs(error) < tolerance:
            return True
