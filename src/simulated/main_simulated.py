import asyncio
import time
import logging
from math import fabs, pi, sin, cos
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError, PositionNedYaw, PositionGlobalYaw
from mavsdk.mission import MissionItem, MissionPlan, MissionError
from mavsdk.action import OrbitYawBehavior
from mavsdk.telemetry import Position
import numpy as np
import cv2.aruco as aruco

from camera_simulated import Video
import utilities_simulated as utilities
import fc_simulated as fc

#Defining constants
ARUCO_SEARCH_ROUNDS = 3
ARUCO_SEARCH_TRIES = 3
ARUCO_SEARCH_RADIUS_MULTIPLIER = 2
ARUCO_SEARCH_STOP_ADDED_TIME = 15
ARUCO_SEARCH_SPOTTED_ADDED_TIME = 5

ALIGNMENT_TARGET_ALTITUDE = 1
ALIGNMENT_MAX_ARUCO_DISTANCE = 11
ALIGNMENT_LINEUP_DISTANCE = 1
ALIGNMENT_MAX_ERROR_DISTANCE = 0.5
ALIGNMENT_YAW_MAX_DISTANCE = 355
ALIGNMENT_YAW_MIN_DISTANCE = 5
ALIGNMENT_MANOUVRES_FACTOR = 0.3
ALIGNMENT_TRANSLATION_FACTOR = 0.6
ALIGNMENT_MAX_DETECTION_FAILURES = 200
ALIGNMENT_MAX_PROCEDURE_FAILURES = 2

TAKE_OFF_ALTITUDE = 3.0
async def run():

    #Prepare logs
    logging.basicConfig(filename="log.txt", level=logging.INFO, format="%(asctime)s %(message)s")

    drone = System()
    print("Connecting...")
    await drone.connect()

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            logging.info("Drone connection estabilished")
            break


    waypoints = [
        (47.398036222362471, 8.5450146439425509, 3),
        (47.398039859999997, 8.5455725400000002, 3),
    ]
    


    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break


    video = Video()

    # intrinsic matrix of the simulated camera, set with a resolution of 1280 × 960
    mtx = np.array([[539.936368, 0, 640.0],
                    [0, 539.936368, 480.0],
                    [0, 0, 1]])
    dist = np.array([0.0, 0.0, 0.0, 0.0])

    #Preparing Aruco detector
    detector = buildArucoDetector()
    print("-- Arming")
    logging.info("Arming and taking off")
    await drone.action.arm()
    await asyncio.sleep(1)
    await drone.offboard.set_position_ned(PositionNedYaw(
        0,0,-TAKE_OFF_ALTITUDE,float('nan')
    ))
    try:
        await drone.offboard.start()
        print("-- Offboard engaged")
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        await drone.action.return_to_launch()
        return

    while True:
        pos = await fc.get_drone_global_position(drone)
        if fabs(pos.relative_altitude_m - TAKE_OFF_ALTITUDE) <= 0.5:
            break

    print("-- Starting navigation")
    battery = await fc.get_drone_remaining_battery(drone)
    logging.info(f"Mission started with {battery.remaining_percent}% of battery, {battery.voltage_v} Volts")
    for (lat, lon, alt) in waypoints:
        print(f"Flying to waypoint {lat}, {lon}")
        logging.info(f"Flying to waypoint {lat}, {lon}")
       
        await fc.fly_to_global(drone,lat,lon,alt)
        await asyncio.sleep(1)
        logging.info("Waypoint reached, starting aruco search")
        
        arucoFound = None
        
        for s in range(ARUCO_SEARCH_TRIES):
            arucoFound = await aruco_search(drone, video, detector, mtx, dist)
            if not arucoFound is None:
                break

        if not arucoFound is None:
            print("Aruco located!")
            result = await align_to_aruco_ned(drone, video, detector, mtx, dist)

            if result is False:
                print("Aruco alignment failed!")
            else:
                print("Aruco alignment completed!")
                await asyncio.sleep(3)

            print("Going to next waypoint")
                
        else:
            print("Aruco not found, going to next waypoint")

    print("Mission completed! Return to home")
    logging.info(f"Mission completed, return to home")
    await drone.offboard.stop()
    await drone.action.return_to_launch()

async def aruco_search(drone: System, video: Video, detector: aruco.ArucoDetector, mtx, dist):
    print("Starting aruco search")
    battery = await fc.get_drone_remaining_battery(drone)
    logging.info(f"Started aruco search with {battery.remaining_percent}% of battery, {battery.voltage_v} Volts")
    round = 1
    initial_altitude = await fc.get_drone_altitude(drone)
    center_ned = await fc.get_drone_ned_position(drone)
    heading = await fc.get_drone_heading(drone)
    while round < ARUCO_SEARCH_ROUNDS:

        arucoFound = utilities.look_for_aruco(video, detector, mtx, dist)
        if not arucoFound is None:
            return arucoFound
        
        radius = ARUCO_SEARCH_RADIUS_MULTIPLIER * round

        print("Starting orbit")
        
        result = await utilities.look_for_aruco_orbit(
            drone, video, detector, mtx, dist,
            center_ned, heading, radius, 1,
        )
        if result is not None:
            return result
        round += 1
   
    center_ned_yaw = PositionNedYaw(
        center_ned.north_m,
        center_ned.east_m,
        center_ned.down_m,
        heading
    )
    await fc.fly_to_ned(drone, center_ned_yaw, 0.5)
    print("Aruco search failed, awaiting the drone to navigate to the initial position")
    await asyncio.sleep(5)
    return None


# TO-DO, update target position in a loop of aruco detection.
async def align_to_aruco_ned_smooth(drone: System, video: Video, detector: aruco.ArucoDetector, mtx, dist):
    print("Starting NED alignment procedure")
    battery = await fc.get_drone_remaining_battery(drone)
    logging.info(f"Started NED aruco alignment with {battery.remaining_percent}% of battery, {battery.voltage_v} Volts")

    aruco_detection_failures = 0
    procedure_failures = 0
    lined_up = False

    initial_ned_position = await fc.get_drone_ned_position(drone)
    initial_heading = await fc.get_drone_heading(drone)
    heading = initial_heading

    aruco_distances = utilities.get_aruco_distances_and_yaw(video, detector, mtx, dist)

    # --- Marker detected ---

    target_ned = await utilities.calculate_target_ned_position(drone, aruco_distances)
    
    vertical_delta = aruco_distances[2] - ALIGNMENT_TARGET_ALTITUDE

    current_ned = await fc.get_drone_ned_position(drone)
    target_ned.down_m = current_ned.down_m + vertical_delta
    await fc.fly_to_ned_smooth(drone,target_ned)
    
    # Hold in place: send current NED position as setpoint
   
    await asyncio.sleep(3)

    # Climb back to initial altitude before resuming mission
    await drone.offboard.set_position_ned(PositionNedYaw(
        target_ned.north_m,
        target_ned.east_m,
        initial_ned_position.down_m,
        heading
    ))
    await asyncio.sleep(7)
    return True



async def align_to_aruco_ned(drone: System, video: Video, detector: aruco.ArucoDetector, mtx, dist):
    print("Starting NED alignment procedure")
    battery = await fc.get_drone_remaining_battery(drone)
    logging.info(f"Started NED aruco alignment with {battery.remaining_percent}% of battery, {battery.voltage_v} Volts")

    aruco_detection_failures = 0
    procedure_failures = 0
    lined_up = False

    initial_ned_position = await fc.get_drone_ned_position(drone)
    initial_heading = await fc.get_drone_heading(drone)
    heading = initial_heading


    
    while True:
        aruco_distances = utilities.get_aruco_distances_and_yaw(video, detector, mtx, dist)

        # --- Detection failure handling ---
        if aruco_distances is None:
            aruco_detection_failures += 1
            print(f"Aruco not found for {aruco_detection_failures} times")

            if aruco_detection_failures >= ALIGNMENT_MAX_DETECTION_FAILURES:
                if procedure_failures < ALIGNMENT_MAX_PROCEDURE_FAILURES:
                    print("Aruco detection failed, retrying from initial NED position")
                    await drone.offboard.set_position_ned(PositionNedYaw(
                        initial_ned_position.north_m,
                        initial_ned_position.east_m,
                        initial_ned_position.down_m,
                        initial_heading
                    ))
                    aruco_detection_failures = 0
                    procedure_failures += 1
                    print("Awaiting drone to navigate to initial position")
                    await asyncio.sleep(10)
                else:
                    print("Aruco detection failed multiple times, aborting alignment")
                    return False
            continue

        # --- Marker detected ---
        aruco_detection_failures = 0

        target_ned = await utilities.calculate_target_ned_position(drone, aruco_distances)
        
        vertical_delta = aruco_distances[2] - ALIGNMENT_TARGET_ALTITUDE

        current_ned = await fc.get_drone_ned_position(drone)
        delta_north = target_ned.north_m - current_ned.north_m
        delta_east = target_ned.east_m - current_ned.east_m
        target_ned.north_m = current_ned.north_m + (delta_north * ALIGNMENT_TRANSLATION_FACTOR)
        target_ned.east_m = current_ned.east_m + (delta_east * ALIGNMENT_TRANSLATION_FACTOR)
        if fabs(aruco_distances[0]) < ALIGNMENT_LINEUP_DISTANCE and fabs(aruco_distances[1]) < ALIGNMENT_LINEUP_DISTANCE:
            lined_up = True

            # Once lined up, start correcting altitude and yaw gradually
            corrected_down = current_ned.down_m + (fabs(vertical_delta * ALIGNMENT_MANOUVRES_FACTOR) * utilities.sign(vertical_delta))
            heading = (await fc.get_drone_heading(drone) + (aruco_distances[3] * ALIGNMENT_MANOUVRES_FACTOR)) % 360
            
            target_ned.down_m = corrected_down
            target_ned.yaw_deg = heading

        else:
            # Not yet lined up horizontally: hold initial altitude
            if not lined_up: target_ned.down_m = initial_ned_position.down_m

        # --- Termination check ---
        yaw_ok = aruco_distances[3] < ALIGNMENT_YAW_MIN_DISTANCE or aruco_distances[3] > ALIGNMENT_YAW_MAX_DISTANCE
        position_ok = (
            fabs(aruco_distances[0]) < ALIGNMENT_MAX_ERROR_DISTANCE and
            fabs(aruco_distances[1]) < ALIGNMENT_MAX_ERROR_DISTANCE and
            fabs(vertical_delta)     < ALIGNMENT_MAX_ERROR_DISTANCE and
            yaw_ok
        )

        if position_ok:
            # Hold in place: send current NED position as setpoint
            await drone.offboard.set_position_ned(PositionNedYaw(
                current_ned.north_m,
                current_ned.east_m,
                current_ned.down_m,
                heading
            ))
            await asyncio.sleep(3)

            # Climb back to initial altitude before resuming mission
            await drone.offboard.set_position_ned(PositionNedYaw(
                current_ned.north_m,
                current_ned.east_m,
                initial_ned_position.down_m,
                heading
            ))
            await asyncio.sleep(7)
            return True

        await drone.offboard.set_position_ned(target_ned)

async def align_to_aruco(drone: System, video: Video, detector: aruco.ArucoDetector, mtx, dist, item: MissionItem):
    print("Starting alignment procedure")
    battery = await fc.get_drone_remaining_battery(drone)
    logging.info(f"Started aruco alignment with {battery.remaining_percent}% of battery, {battery.voltage_v} Volts")
    aruco_detection_failures = 0
    procedure_failures = 0
    lined_up = False
    initial_position = await fc.get_drone_global_position(drone)
    heading = await fc.get_drone_heading(drone)
    initial_heading = heading

    while True:
        aruco_distances = utilities.get_aruco_distances_and_yaw(video, detector, mtx, dist)
        if aruco_distances is None:
            aruco_detection_failures += 1
            print(f"Aruco not found for {aruco_detection_failures} times")
            if aruco_detection_failures >= ALIGNMENT_MAX_DETECTION_FAILURES and procedure_failures <= ALIGNMENT_MAX_PROCEDURE_FAILURES:
                print("Aruco detection failed, retry from inital position")
                await drone.action.goto_location(initial_position.latitude_deg, initial_position.longitude_deg, initial_position.absolute_altitude_m, initial_heading)
                aruco_detection_failures = 0
                procedure_failures += 1
                print("Awaiting the drone to navigate to the initial position")
                await asyncio.sleep(10)
            elif aruco_detection_failures >= ALIGNMENT_MAX_DETECTION_FAILURES and procedure_failures >= ALIGNMENT_MAX_PROCEDURE_FAILURES:
                print("Aruco detection failed multiple times, aborting alignment")
                return False
            continue

        aruco_detection_failures = 0
        current_position = await fc.get_drone_global_position(drone)

        target_coordinates = await utilities.calculate_target_global_position(drone, aruco_distances)
        vertical_delta = aruco_distances[2] - ALIGNMENT_TARGET_ALTITUDE
        
        if fabs(aruco_distances[0]) < ALIGNMENT_LINEUP_DISTANCE and fabs(aruco_distances[1]) < ALIGNMENT_LINEUP_DISTANCE:
            lined_up = True
            
            # Weighing the results for smoother manouvres
            target_coordinates.absolute_altitude_m = current_position.absolute_altitude_m - (fabs(vertical_delta * ALIGNMENT_MANOUVRES_FACTOR) * utilities.sign(vertical_delta))
            heading = (await fc.get_drone_heading(drone) + (aruco_distances[3] * ALIGNMENT_MANOUVRES_FACTOR)) % 360
        else:
            if not lined_up: target_coordinates.absolute_altitude_m = initial_position.absolute_altitude_m

        if fabs(aruco_distances[0]) < ALIGNMENT_MAX_ERROR_DISTANCE and fabs(aruco_distances[1]) < ALIGNMENT_MAX_ERROR_DISTANCE and fabs(vertical_delta) < ALIGNMENT_MAX_ERROR_DISTANCE and (aruco_distances[3] < ALIGNMENT_YAW_MIN_DISTANCE or aruco_distances[3] > ALIGNMENT_YAW_MAX_DISTANCE):
            await drone.action.hold()
            await asyncio.sleep(3)
            await drone.action.goto_location(target_coordinates.latitude_deg, target_coordinates.longitude_deg, initial_position.absolute_altitude_m, initial_heading)
            await asyncio.sleep(7)
            return True

        await drone.action.goto_location(target_coordinates.latitude_deg, target_coordinates.longitude_deg, target_coordinates.absolute_altitude_m, heading)

def buildArucoDetector():
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    arucoParams = aruco.DetectorParameters()
    return aruco.ArucoDetector(arucoDict, arucoParams)

if __name__ == "__main__":
    # Start the main function
    asyncio.run(run())