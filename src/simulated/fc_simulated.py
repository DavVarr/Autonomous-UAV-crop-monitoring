from mavsdk import System
from mavsdk.telemetry import Position, PositionNed, FlightMode, Battery
from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed
from mavsdk.mission import MissionItem, MissionProgress
from math import radians, cos, fabs
import asyncio

async def get_drone_home_position(drone: System) -> Position:
    async for position in drone.telemetry.home():
        return position

async def get_drone_global_position(drone: System) -> Position:
    async for position in drone.telemetry.position():
        return position

async def get_drone_ned_position(drone: System) -> PositionNed:
    async for position_velocity in drone.telemetry.position_velocity_ned():
        return position_velocity.position

async def get_drone_altitude(drone: System) -> float:
    async for altitude in drone.telemetry.altitude():
        return altitude.altitude_amsl_m

async def get_drone_heading(drone: System) -> float:
    async for heading in drone.telemetry.heading():
        return heading.heading_deg
    
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
        print("Position reached")
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