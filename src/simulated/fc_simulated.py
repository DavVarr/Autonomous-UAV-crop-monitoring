from mavsdk import System
from mavsdk.telemetry import Position, PositionNed, FlightMode, Battery, PositionVelocityNed, EulerAngle
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, AccelerationNed, VelocityBodyYawspeed, PositionGlobalYaw
from mavsdk.mission import MissionItem, MissionProgress
from math import radians, cos, fabs, sqrt

import numpy as np
import ruckig
import time
import asyncio

async def get_drone_home_position(drone: System) -> Position:
    async for position in drone.telemetry.home():
        return position

async def get_drone_global_position(drone: System) -> Position:
    async for position in drone.telemetry.position():
        return position

async def get_drone_ned_position_velocity(drone: System) -> PositionVelocityNed:
    async for position_velocity in drone.telemetry.position_velocity_ned():
        return position_velocity
    
async def get_drone_ned_position(drone: System) -> PositionNed:
    async for position_velocity in drone.telemetry.position_velocity_ned():
        return position_velocity.position

async def get_drone_altitude(drone: System) -> float:
    async for altitude in drone.telemetry.altitude():
        return altitude.altitude_amsl_m

async def get_drone_heading(drone: System) -> float:
    async for heading in drone.telemetry.heading():
        return heading.heading_deg

async def get_drone_attitude_euler(drone: System):
    """Return the latest roll/pitch/yaw telemetry sample in degrees."""
    async for attitude in drone.telemetry.attitude_euler():
        return attitude

async def get_drone_attitude_euler(drone: System) -> EulerAngle:
    """Return the latest body attitude (roll, pitch, yaw) from telemetry."""
    async for attitude in drone.telemetry.attitude_euler():
        return attitude
    
async def get_drone_flight_mode(drone: System) -> FlightMode:
    async for mode in drone.telemetry.flight_mode():
        return mode
    
async def get_drone_mission_progress(drone: System) -> MissionProgress:
    async for progress in drone.mission.mission_progress():
        return progress
    
async def get_drone_remaining_battery(drone: System) -> Battery:
    async for battery in drone.telemetry.battery():
        return battery

async def get_drone_ned_current_position(drone: System):
    home_position: Position = get_drone_home_position(drone)
    current_position: Position = get_drone_global_position(drone)
    heading = get_drone_heading(drone)

    # Convert reference position to radians
    ref_latitude_rad = radians(home_position.latitude_deg)
    ref_longitude_rad = radians(home_position.longitude_deg)

    # Convert current position to radians
    current_latitude_rad = radians(current_position.latitude_deg)
    current_longitude_rad = radians(current_position.longitude_deg)

    # Earth radius (approximate value for WGS84 ellipsoid)
    R = 6378137.0  # meters

    # Calculate NED coordinates
    north_m = (current_latitude_rad - ref_latitude_rad) * R
    east_m = (current_longitude_rad - ref_longitude_rad) * R * cos(ref_latitude_rad)
    down_m = home_position.absolute_altitude_m - current_position.absolute_altitude_m

    return PositionNedYaw(north_m, east_m, down_m, heading)

async def check_global_position_reached(current_position: Position, target_position: Position):
    TOLERANCE_DEG = 0.000008  # Tolerance for latitude and longitude
    TOLERANCE_M = 1  # Tolerance for altitude

    diff_lat = fabs(current_position.latitude_deg - target_position.latitude_deg) - TOLERANCE_DEG
    diff_lon = fabs(current_position.longitude_deg - target_position.longitude_deg) - TOLERANCE_DEG
    diff_alt = fabs(current_position.relative_altitude_m - target_position.relative_altitude_m) - TOLERANCE_M
    if diff_lat <= 0 and diff_lon <= 0 and diff_alt <= 0:
        print("Position reached")
        return True
    return False

async def check_reached_checkpoint(drone: System, item: MissionItem):
    target_position = Position(item.latitude_deg, item.longitude_deg, None, item.relative_altitude_m)
    async for position in drone.telemetry.position():
        if await check_global_position_reached(position, target_position):
            return True

async def check_ned_position_reached(drone: System, target_position: PositionNedYaw, tolerance: float):
    position_velocity = await get_drone_ned_position(drone)
    distance_to_target = ((position_velocity.north_m - target_position.north_m) ** 2 +
                            (position_velocity.east_m - target_position.east_m) ** 2 +
                            (position_velocity.down_m - target_position.down_m) ** 2) ** 0.5

    if distance_to_target < tolerance:
        #print("Position reached")
        return True
    else:
        return False

async def check_ned_position_reached_with_velocity(drone: System, target_position: PositionNedYaw, tolerance: float, velocity_threshold: float):
    position_velocity = await get_drone_ned_position_velocity(drone)
    distance_to_target = ((position_velocity.position.north_m - target_position.north_m) ** 2 +
                            (position_velocity.position.east_m - target_position.east_m) ** 2 +
                            (position_velocity.position.down_m - target_position.down_m) ** 2) ** 0.5
    velocity_magnitude = (position_velocity.velocity.north_m_s ** 2 +
                          position_velocity.velocity.east_m_s ** 2 +
                          position_velocity.velocity.down_m_s ** 2) ** 0.5

    if distance_to_target < tolerance and velocity_magnitude < velocity_threshold:
        return True
    else:
        return False
    
async def set_local_speeds(forward: float, right: float, down: float, yaw: float, drone: System):
    movement = VelocityBodyYawspeed(forward, right, down, yaw)
    await drone.offboard.set_velocity_body(movement)

async def move_locally(forward: float, right: float, down: float, yaw: float, drone: System):
    highest_value = max(abs(forward), abs(right), abs(down))
    print(f"Forward: {forward}, Right: {right}, Down: {down}, Yaw: {yaw}, Highest: {highest_value}")
    if highest_value == 0: return

    normalized_forward = forward/highest_value
    normalized_right = right/highest_value
    normalized_down = down/highest_value

    if yaw is not None:
        current_yaw = await get_drone_heading(drone)
        if current_yaw == yaw : yaw = 0
        elif yaw > current_yaw : yaw = yaw - current_yaw
        else: yaw = yaw - current_yaw + 360
    else :
        yaw = 0
    normalized_yaw = yaw/highest_value

    movement = VelocityBodyYawspeed(normalized_forward, normalized_right, normalized_down, normalized_yaw)
    stop = VelocityBodyYawspeed(0,0,0,0)
    await drone.offboard.set_velocity_body(movement)
    await asyncio.sleep(highest_value)
    await drone.offboard.set_velocity_body(stop)
    await asyncio.sleep(1)

async def fly_to_ned(drone: System, target: PositionNedYaw, threshold: float):
    """Send position setpoint and wait until within threshold metres."""
    while True:
        await drone.offboard.set_position_ned(target)
        current = await get_drone_ned_position(drone)
        dist = sqrt(
            (target.north_m - current.north_m) ** 2 +
            (target.east_m  - current.east_m)  ** 2
        )
        if dist < threshold:
            return
        await asyncio.sleep(0.1)

def horizontal_distance(lat1, lon1, lat2, lon2):
    """Quick flat-earth approximation, good enough for small distances"""
    R = 6378137.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = dlat**2 + (cos(radians(lat1)) * dlon)**2
    return R * sqrt(a)

async def fly_to_global(drone: System,
                         target_lat, target_lon, target_alt,
                         heading=float('nan'),
                         max_velocity=10.0, max_acceleration=3.0, max_jerk=4.0):
    M_PER_DEG_LAT = 111_320.0
 
    gpos = await get_drone_global_position(drone)
    pos  = await get_drone_ned_position(drone)
    m_per_deg_lon = M_PER_DEG_LAT * cos(radians(gpos.latitude_deg))
 
    target = PositionNedYaw(
        pos.north_m + (target_lat - gpos.latitude_deg) * M_PER_DEG_LAT,
        pos.east_m  + (target_lon - gpos.longitude_deg) * m_per_deg_lon,
        pos.down_m  + (gpos.relative_altitude_m - target_alt),   # NED down is positive
        heading
    )
 
    await fly_to_ned_smooth(drone, target, threshold=0.2, max_velocity=max_velocity,
                             max_acceleration=max_acceleration, max_jerk=max_jerk)
    #await asyncio.sleep(0.5) 



async def fly_to_ned_smooth(drone: System,
                            target: PositionNedYaw,
                            threshold: float = 0.3,
                            max_velocity=3.0,max_acceleration=3.0,max_jerk=2.0):

    otg = ruckig.Ruckig(3)
    traj = ruckig.Trajectory(3)
    inp = ruckig.InputParameter(3)
    pos = await get_drone_ned_position(drone)
    inp.current_position = [pos.north_m, pos.east_m, pos.down_m]
    inp.current_velocity = [0.0, 0.0, 0.0]
    inp.current_acceleration = [0.0, 0.0, 0.0]
    inp.target_position     = [target.north_m, target.east_m, target.down_m]
    inp.target_velocity     = [0.0, 0.0, 0.0]
    inp.target_acceleration = [0.0, 0.0, 0.0]
    inp.max_velocity        = [max_velocity] * 3
    inp.max_acceleration    = [max_acceleration] * 3
    inp.max_jerk            = [max_jerk] * 3
    t1 = time.perf_counter()
    otg.calculate(inp, traj)
    t2 = time.perf_counter()
    print(t2-t1)
    start_time = time.perf_counter()
    
    while True:

        current_time = time.perf_counter()
        t = current_time - start_time


        p, v, a  = traj.at_time(t)
        #await drone.offboard.set_position_ned(PositionNedYaw(p[0], p[1], p[2], target.yaw_deg))
        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(p[0], p[1], p[2], target.yaw_deg),
            VelocityNedYaw(v[0], v[1], v[2], target.yaw_deg),
            AccelerationNed(a[0], a[1], a[2])
        )
        
        if t >= traj.duration:
            print("Trajectory successfully completed!")
            break
    while True:
        pos_vel_reached = await check_ned_position_reached_with_velocity(drone, PositionNedYaw(*inp.target_position, float("nan")), tolerance=threshold, velocity_threshold=0.1)
        if pos_vel_reached:
            return
async def fly_to_ned_recomp(drone: System,
                            target: PositionNedYaw,
                            threshold: float = 0.3):
    
    def _trajectory(position, velocity, acceleration, target,
                max_xy_velocity, max_down_velocity, max_climb_velocity,
                max_xy_acceleration, max_z_acceleration, max_jerk):
        inp = ruckig.InputParameter(3)
        traj = ruckig.Trajectory(3)

        inp.current_position = position.tolist()
        inp.current_velocity = velocity.tolist()
        inp.current_acceleration = acceleration.tolist()
        inp.target_position     = [target.north_m, target.east_m, target.down_m]
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
    last_acceleration = np.array([0.0, 0.0, 0.0])
    while True:

        pv = await get_drone_ned_position_velocity(drone)

        position_ned = np.array([
            pv.position.north_m, pv.position.east_m, pv.position.down_m,
        ])
        velocity_ned = np.array([
            pv.velocity.north_m_s,
            pv.velocity.east_m_s,
            pv.velocity.down_m_s,
        ])
        traj = _trajectory(
            position_ned, velocity_ned, last_acceleration, target,
            3, 3,
            3, 3,
            3, 2,
        )
        p, v, a  = traj.at_time(0.05)
        last_acceleration = np.array(a)
        #await drone.offboard.set_position_ned(PositionNedYaw(p[0], p[1], p[2], target.yaw_deg))
        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(p[0], p[1], p[2], target.yaw_deg),
            VelocityNedYaw(v[0], v[1], v[2], target.yaw_deg),
            AccelerationNed(a[0], a[1], a[2])
        )
        target_np = np.array([target.north_m, target.east_m, target.down_m])
        error = target_np - position_ned
        if (
            np.max(np.abs(error)) <= 0.15
            and np.linalg.norm(velocity_ned) <= 0.10
        ):
            await drone.offboard.set_position_ned(PositionNedYaw(
                float(position_ned[0]), float(position_ned[1]),
                float(position_ned[2]), float("nan"),
            ))
            return True


def make_position_smoother(trajectory, max_lag=0.25):
    """
    Returns a function:
        position_sp, velocity_sp, acceleration_sp = follower(current_position)

    The virtual trajectory clock slows when the drone falls behind.
    """

    virtual_time = 0.0
    last_time = time.perf_counter()

    start = np.asarray(trajectory.at_time(0.0)[0])
    end = np.asarray(trajectory.at_time(trajectory.duration)[0])
    direction = end - start
    direction /= max(np.linalg.norm(direction), 1e-6)

    def follower(current_position):
        nonlocal virtual_time, last_time

        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        position, velocity, acceleration = trajectory.at_time(virtual_time)
        position = np.asarray(position)

        # Positive if the reference is ahead of the drone.
        lag = np.dot(
            position - np.asarray(current_position),
            direction,
        )

        # PX4-like trajectory time stretching.
        rate = 1.0 - np.clip(lag / max_lag, 0.0, 1.0)
        if rate < 1: print(f"Trajectory follower lag: {lag:.3f} m, rate: {rate:.3f}")
        virtual_time = min(
            virtual_time + dt * rate,
            trajectory.duration,
        )

        return trajectory.at_time(virtual_time)

    return follower