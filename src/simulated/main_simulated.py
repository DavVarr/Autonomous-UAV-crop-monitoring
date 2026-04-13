import asyncio
import time
import logging
from math import fabs
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError
from mavsdk.mission import MissionItem, MissionPlan, MissionError
from mavsdk.action import OrbitYawBehavior
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
ALIGNMENT_MAX_DETECTION_FAILURES = 200
ALIGNMENT_MAX_PROCEDURE_FAILURES = 2

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

    print("Preparing mission")
    mission_items = []
    mission_items.append(MissionItem(
                                    47.3977509,
                                    8.5456069,
                                    #Baylands37.4142,
                                    #Baylands-121.9961,
                                    3,
                                    10,
                                    False,
                                    float('nan'),
                                    float('nan'),
                                    MissionItem.CameraAction.NONE,
                                    1,
                                    float('nan'),
                                    float('nan'),
                                    float('nan'),
                                    float('nan'),
                                    MissionItem.VehicleAction.TAKEOFF))
    mission_items.append(MissionItem(
                                    47.398036222362471,
                                    8.5450146439425509,
                                    #Baylands37.4142,
                                    #Baylands-121.9961,
                                    3,
                                    10,
                                    False,
                                    float('nan'),
                                    float('nan'),
                                    MissionItem.CameraAction.NONE,
                                    1,
                                    float('nan'),
                                    float('nan'),
                                    float('nan'),
                                    float('nan'),
                                    MissionItem.VehicleAction.NONE))
    mission_items.append(MissionItem(
                                    47.398039859999997,
                                    8.5455725400000002,
                                    #Baylands37.4142,
                                    #Baylands-121.9951,
                                    3,
                                    10,
                                    False,
                                    float('nan'),
                                    float('nan'),
                                    MissionItem.CameraAction.NONE,
                                    1,
                                    float('nan'),
                                    float('nan'),
                                    float('nan'),
                                    float('nan'),
                                    MissionItem.VehicleAction.NONE))
    mission_items.append(MissionItem(
                                    47.3977509,
                                    8.5456069,
                                    #Baylands37.4142,
                                    #Baylands-121.9961,
                                    3,
                                    10,
                                    False,
                                    float('nan'),
                                    float('nan'),
                                    MissionItem.CameraAction.NONE,
                                    1,
                                    float('nan'),
                                    float('nan'),
                                    float('nan'),
                                    float('nan'),
                                    MissionItem.VehicleAction.LAND))

    mission_plan = MissionPlan(mission_items)

    # Tuning parameters
    await drone.mission.set_return_to_launch_after_mission(True)

    print("-- Uploading mission")
    await drone.mission.clear_mission()
    await asyncio.sleep(1)
    await drone.mission.upload_mission(mission_plan)

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    # Start offboard mode
    print("-- Setting initial setpoint")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    try:
        await drone.offboard.start()
        print("Offboard mode engaged!")
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code:"
            f" {error._result.result}")
        return

    video = Video()

    # intrinsic matrix of the simulated camera, set with a resolution of 640 × 1232
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
    await drone.action.takeoff()
    await asyncio.sleep(3)

    print("-- Starting mission")
    try:
        await drone.mission.start_mission()
        print("Mission started!")
        battery = await fc.get_drone_remaining_battery(drone)
        logging.info(f"Mission started with {battery.remaining_percent}% of battery, {battery.voltage_v} Volts")
    except MissionError as error:
        print(f"Starting mission failed with error code:"
            f" {error._result.result}")
        return
    
    for i, item in enumerate(mission_items):

        if item.vehicle_action == MissionItem.VehicleAction.TAKEOFF or item.vehicle_action == MissionItem.VehicleAction.LAND:
            continue

        while True:
            progress = await fc.get_drone_mission_progress(drone)
            if (progress.current) == i + 1:
                break
        
        print("Checkpoint reached, pausing mission")
        logging.info(f"Aruco checkpoint reached, pausing mission")
        await drone.mission.pause_mission()
        
        arucoFound = None
        for s in range(ARUCO_SEARCH_TRIES):
            arucoFound = await aruco_search(drone, video, detector, mtx, dist, item)
            if not arucoFound is None:
                break

        if not arucoFound is None:
            print("Aruco located!")
            result = await align_to_aruco(drone, video, detector, mtx, dist, item)

            if result is False:
                print("Aruco alignment failed!")
            else:
                print("Aruco alignment completed!")
                await asyncio.sleep(3)

            print("Resuming mission")
            await drone.mission.start_mission()
                
        else:
            print("Aruco not found, resuming mission")
            await drone.mission.start_mission()

    print("Mission completed! Return to home")
    logging.info(f"Mission completed, return to home")
    await drone.action.return_to_launch()

async def aruco_search(drone: System, video: Video, detector: aruco.ArucoDetector, mtx, dist, item: MissionItem):
    print("Starting aruco search")
    battery = await fc.get_drone_remaining_battery(drone)
    logging.info(f"Started aruco search with {battery.remaining_percent}% of battery, {battery.voltage_v} Volts")
    round = 1
    initial_altitude = await fc.get_drone_altitude(drone)

    while round < ARUCO_SEARCH_ROUNDS:

        arucoFound = utilities.look_for_aruco(video, detector, mtx, dist)
        if not arucoFound is None:
            return arucoFound
        
        radius = ARUCO_SEARCH_RADIUS_MULTIPLIER * round

        print("Starting orbit")
        await drone.action.do_orbit(radius,
                                    1,
                                    OrbitYawBehavior.HOLD_INITIAL_HEADING,
                                    item.latitude_deg,
                                    item.longitude_deg,
                                    initial_altitude)
        
        target_time = time.time() + ARUCO_SEARCH_STOP_ADDED_TIME * round

        while time.time() < target_time:
            arucoFound = utilities.look_for_aruco(video, detector, mtx, dist)
            # Double check to avoid problems due to camera angle
            if not arucoFound is None:
                print("Spotted marker")
                await drone.action.hold()
                target_time = target_time + ARUCO_SEARCH_SPOTTED_ADDED_TIME
                await asyncio.sleep(2)

                arucoFound = utilities.look_for_aruco(video, detector, mtx, dist)
                if not arucoFound is None:
                    return arucoFound
                else:
                    await drone.action.do_orbit(radius,
                                    1,
                                    OrbitYawBehavior.HOLD_INITIAL_HEADING,
                                    item.latitude_deg,
                                    item.longitude_deg,
                                    initial_altitude)
                    continue
        round += 1

    await drone.action.goto_location(item.latitude_deg, item.longitude_deg, initial_altitude, item.yaw_deg)
    print("Aruco search failed, awaiting the drone to navigate to the initial position")
    await asyncio.sleep(5)
    return None

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