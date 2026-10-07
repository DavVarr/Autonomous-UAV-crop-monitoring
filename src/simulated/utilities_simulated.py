import asyncio
import ruckig
import time
from mavsdk.telemetry import PositionNed, Position
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, AccelerationNed
from camera_simulated import checkArucoPresence, Video
from math import fabs, degrees, radians, pi, cos, sin, sqrt, atan2, asin
import numpy as np
import fc_simulated as fc
from mavsdk import System

def sign(num):
    if num == 0: return 0
    return -1 if num < 0 else 1

def look_for_aruco(video: Video, detector, mtx, dist):
    for i in range(10):
        result = checkArucoPresence(video, detector, mtx, dist)
        if not result is None:
            return result
    return None

def get_circle_coordinates(starting_position: PositionNed, heading: float, radius: float) -> list[PositionNed]:
    coordinates = []
    for i in range(0, 10):
        theta = (2 * pi * i) / 10 + radians(heading)
        new_north = starting_position.north_m + radius * cos(theta)
        new_east = starting_position.east_m + radius * sin(theta)
        coordinates.append(PositionNed(new_north, new_east, starting_position.down_m))
    return coordinates

async def get_position_corrections(
        current_forward_distance: float,
        current_right_distance: float,
        current_forward_speed: float,
        current_right_speed: float,
        old_forward_correction: float,
        old_right_correction: float,
        old_forward_wind_speed: float,
        old_right_wind_speed: float,
        alfa:float,
        beta:float
    ):

    forward_wind_speed = old_forward_correction - current_forward_speed
    forward_wind_speed = forward_wind_speed * beta + (1 - beta) * old_forward_wind_speed

    right_wind_speed = old_right_correction - current_right_speed
    right_wind_speed = right_wind_speed * beta + (1 - beta) * old_right_wind_speed

    current_forward_correction = -forward_wind_speed + (min(max(fabs(alfa * current_forward_distance), 0.1), 0.4) * sign(current_forward_distance))
    current_right_correction = -right_wind_speed + (min(max(fabs(alfa * current_right_distance), 0.1), 0.4) * sign(current_right_distance))

    #print(f"Forward speed: {current_forward_speed}")
    #print(f"Forward correction: {current_forward_correction}")
    #print(f"Old Forward correction: {old_forward_correction}")
    #print(f"Right correction: {current_right_correction}")
    #print(f"Forward wind: {forward_wind_speed}")
    #print(f"Right wind: {right_wind_speed}")
    #print("")

    return [current_forward_correction, current_right_correction, forward_wind_speed, right_wind_speed]

def get_aruco_distances_and_yaw(rvecs, tvecs) -> list[float] | None:
    """Convert an already-computed marker pose to the aligned distances.

    This function performs no image acquisition and no ArUco detection. Pass
    it the ``rvecs`` and ``tvecs`` returned by
    ``camera_simulated.my_estimatePoseSingleMarkers`` so corners, pose, and
    FOV measurements all come from the same frame.
    """
    if rvecs is None or tvecs is None or len(rvecs) == 0 or len(tvecs) == 0:
        return None

    rvec = np.asarray(rvecs[0], dtype=float).reshape(3)
    tvec = np.asarray(tvecs[0], dtype=float).reshape(3)

    forward_distance = -float(tvec[1])
    right_distance = float(tvec[0])
    vertical_distance = float(tvec[2])
    yaw_difference = normalize_heading(degrees(float(rvec[2])))

    return [
        forward_distance,
        right_distance,
        vertical_distance,
        yaw_difference,
    ]


def detect_aruco_distances_and_yaw(video: Video, detector, mtx, dist):
    """Legacy convenience wrapper for older alignment/search functions.

    New code should detect once and call ``get_aruco_distances_and_yaw`` with
    the returned pose. This wrapper is kept only so unrelated old project
    functions can still use their previous video-based behavior.
    """
    aruco_result = checkArucoPresence(video, detector, mtx, dist)
    if aruco_result is None:
        return None
    rvecs, tvecs, _ = aruco_result
    return get_aruco_distances_and_yaw(rvecs, tvecs)


async def calculate_target_ned_position(drone: System, aruco_distances: list[float]) -> PositionNedYaw:
    current_ned_position: PositionNed = await fc.get_drone_ned_position(drone)
    current_heading = await fc.get_drone_heading(drone)


    ned_vector = rotate_vector(aruco_distances[0], aruco_distances[1], current_heading)

    target_position = PositionNedYaw(
        ned_vector[0] + current_ned_position.north_m,
        ned_vector[1] + current_ned_position.east_m,
        current_ned_position.down_m,
        current_heading)
    
    return target_position

async def calculate_target_global_position(drone: System, aruco_distances: list[float]) -> Position:
    current_global_position = await fc.get_drone_global_position(drone)
    current_heading = await fc.get_drone_heading(drone)

    # 270 degrees added to compensate for camera placement
    rotated_vector = rotate_vector(aruco_distances[0], aruco_distances[1], 0)

    distance = sqrt(pow(rotated_vector[0], 2) + pow(rotated_vector[1], 2))
    angle = degrees(atan2(rotated_vector[1], rotated_vector[0])) % 360
    direction = (current_heading + angle) % 360

    target_position = calculate_new_coordinates(current_global_position, distance, direction)
    
    return target_position

def rotate_vector(x, y, theta):
    theta_rad = np.radians(theta)

    v = np.array([x, y])
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])

    v_rotated = np.dot(R, v)
    return v_rotated

def calculate_new_coordinates(initial_position: Position, distance: float, direction: float):
    lat = radians(initial_position.latitude_deg)
    lon = radians(initial_position.longitude_deg)
    direction = radians(direction)
    
    earth_radius = 6378137.0
    
    new_lat = asin(sin(lat) * cos(distance / earth_radius) +
                          cos(lat) * sin(distance / earth_radius) * cos(direction))
    
    new_lon = lon + atan2(sin(direction) * sin(distance / earth_radius) * cos(lat),
                                 cos(distance / earth_radius) - sin(lat) * sin(new_lat))
    
    new_lat = degrees(new_lat)
    new_lon = degrees(new_lon)
    
    return Position(new_lat, new_lon, initial_position.absolute_altitude_m, initial_position.relative_altitude_m)

def normalize_heading(heading: float):
    if heading < -180: return heading + 360
    elif heading > 180: return heading - 360
    else: return heading


async def look_for_aruco_orbit(drone : System, video : Video, detector, mtx, dist,
                         center_ned : PositionNed, heading, radius, speed):
    omega = speed / radius
    accumulated_angle = 0.0
    prev_radial = None

    # Fly to orbit start point
    start = PositionNedYaw(
        center_ned.north_m + radius,
        center_ned.east_m,
        center_ned.down_m,
        heading
    )
    await fc.fly_to_ned(drone, start, 0.3)

    while accumulated_angle < 2 * pi:  # one full orbit
        current_ned = await fc.get_drone_ned_position(drone)

        dx = current_ned.north_m - center_ned.north_m
        dy = current_ned.east_m  - center_ned.east_m
        current_radius = sqrt(dx**2 + dy**2)

        if current_radius < 1e-3:
            continue

        radial_x = dx / current_radius
        radial_y = dy / current_radius

        # Accumulate swept angle using cross product between consecutive radials
        if prev_radial is not None:
            # cross product gives sine of angle between vectors
            # dot product gives cosine
            cross = prev_radial[0] * radial_y - prev_radial[1] * radial_x
            dot   = prev_radial[0] * radial_x + prev_radial[1] * radial_y
            delta_angle = atan2(cross, dot)  # signed angle increment
            accumulated_angle += fabs(delta_angle)

        prev_radial = (radial_x, radial_y)

        # Tangential velocity
        vn = -radial_y * speed
        ve =  radial_x * speed

        # Radial correction
        radius_error = radius - current_radius
        vn += radius_error * radial_x
        ve += radius_error * radial_y

        # Centripetal acceleration feedforward
        an = -radial_x * speed**2 / radius
        ae = -radial_y * speed**2 / radius

        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(float('nan'), float('nan'), center_ned.down_m, heading),
            VelocityNedYaw(vn, ve, 0.0, heading),
            AccelerationNed(an, ae, 0.0)
        )

        result = detect_aruco_distances_and_yaw(video, detector, mtx, dist)
        if result is not None:
            target_ned = await calculate_target_ned_position(drone,result)
            handoff = {}
            transition_task = asyncio.create_task(_smooth_move_to_target(
                drone, target_ned, heading, (an, ae), speed,handoff
            ))
            transition_task.handoff = handoff
            transition_task.target_xy = np.array([target_ned.north_m,target_ned.east_m])
            #await asyncio.sleep(0.5)
            return result,transition_task

    return None, None # completed full orbit without finding aruco

async def _smooth_move_to_target(drone, target, heading, acceleration,
                                 max_velocity, handoff ,max_acceleration=3, max_jerk=2.0):
    pv = await fc.get_drone_ned_position_velocity(drone)
    p0 = np.array([pv.position.north_m, pv.position.east_m])
    v0 = np.array([pv.velocity.north_m_s, pv.velocity.east_m_s])
    a0 = np.asarray(acceleration, float)
    target_xy = np.array([target.north_m, target.east_m])

    inp, trajectory = ruckig.InputParameter(2), ruckig.Trajectory(2)
    inp.current_position = p0.tolist()
    inp.current_velocity = v0.tolist()
    inp.current_acceleration = a0.tolist()
    inp.target_position = target_xy.tolist()
    inp.target_velocity = inp.target_acceleration = [0., 0.]
    inp.max_velocity = [max(max_velocity, abs(v0[0])),
                        max(max_velocity, abs(v0[1]))]
    inp.max_acceleration = [max(max_acceleration, abs(a0[0])),
                            max(max_acceleration, abs(a0[1]))]
    inp.max_jerk = [max_jerk, max_jerk]

    result = ruckig.Ruckig(2).calculate(inp, trajectory)
    if result not in (ruckig.Result.Working, ruckig.Result.Finished):
        return
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        p, v, a = map(np.asarray, trajectory.at_time(
            min(elapsed, trajectory.duration)))

        handoff["p"] = p.copy()
        handoff["v"] = v.copy()
        handoff["a"] = a.copy()
        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(float(p[0]), float(p[1]), target.down_m, heading),
            VelocityNedYaw(float(v[0]), float(v[1]), 0., heading),
            AccelerationNed(float(a[0]), float(a[1]), 0.),
        )
        if elapsed >= trajectory.duration:
            return
        

async def wait_gps_ctrl(drone, value=4, timeout=2.0):
    start = time.perf_counter()

    while await drone.param.get_param_int("EKF2_GPS_CTRL") != value:
        if time.perf_counter() - start > timeout:
            raise TimeoutError(
                f"EKF2_GPS_CTRL did not become {value}"
            )
        await asyncio.sleep(0.05)