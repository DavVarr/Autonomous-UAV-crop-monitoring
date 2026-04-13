import asyncio
from time import perf_counter, time
import logging
from math import fabs
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError
from mavsdk.mission import MissionItem, MissionError
from mavsdk.action import OrbitYawBehavior
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import numpy as np
import cv2.aruco as aruco

import utilities
import fc

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
ALIGNMENT_MAX_DETECTION_FAILURES = 20
ALIGNMENT_MAX_PROCEDURE_FAILURES = 2

ARUCO_DETECTOR_THRESHOLD_WINDOW_MIN = 15
ARUCO_DETECTOR_THRESHOLD_WINDOW_MAX = 45
ARUCO_DETECTOR_THRESHOLD_WINDOW_STEP = 15
ARUCO_DETECTOR_MIN_MARKER_PERIMETER_RATE = 0.1

async def run():

    #Prepare logs
    logging.basicConfig(filename="log.txt", level=logging.INFO, format="%(asctime)s %(message)s")

    drone = System()
    print("Connecting...")
    await drone.connect(system_address="serial:///dev/serial0:57600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            logging.info("Drone connection estabilished")
            break

    await drone.mission.set_return_to_launch_after_mission(True)

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    # Get mission items already loaded into the vehicle with QGround Control
    mission_plan = await drone.mission.download_mission()
    mission_items = mission_plan.mission_items

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

    # Preparing camera
    camera = Picamera2()
    main_stream = {"size": (1640, 1232)}
    lores_stream = {"size": (1640, 1232)}
    video_config = camera.create_video_configuration(main_stream, lores_stream, encode="lores")
    camera.configure(video_config)
    encoder = H264Encoder(10000000)

    # intrinsic matrix of the Raspberry Pi camera, set with a resolution of 1640 × 1232
    mtx = np.array([[1607, 0, 820.0],
                    [0, 1607, 616.0],
                    [0, 0, 1]])
    dist = np.array([0.0, 0.0, 0.0, 0.0])

    #Preparing Aruco detector
    detector = buildArucoDetector()

    camera.start_recording(encoder, 'video.h264')
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
        #battery = await fc.get_drone_remaining_battery(drone)
        #logging.info(f"Mission started with {battery.remaining_percent}% of battery, {battery.voltage_v} Volts")
    except MissionError as error:
        print(f"Starting mission failed with error code:"
            f" {error._result.result}")
        return

    for i, item in enumerate(mission_items):

        if item.vehicle_action == MissionItem.VehicleAction.TAKEOFF or item.vehicle_action == MissionItem.VehicleAction.LAND:
            continue

        while True:
            mission_progress_start = perf_counter()
            progress = await fc.get_drone_mission_progress(drone)
            mission_progress_end = perf_counter()
            logging.info(f"Getting mission progress took {mission_progress_end - mission_progress_start} seconds")
            print(f"Current item: {i}, progress: {progress.current}")
            if (progress.current) == i + 1:
                break

        print("Checkpoint reached, pausing mission")
        logging.info(f"Aruco checkpoint reached, pausing mission")
        pause_mission_start = perf_counter()
        await drone.mission.pause_mission()
        pause_mission_end = perf_counter()
        logging.info(f"Pausing mission took {pause_mission_end - pause_mission_start} seconds")

        logging.info("Starting aruco search")
        arucoFound = None
        for s in range(ARUCO_SEARCH_TRIES):
            arucoFound = await aruco_search(drone, camera, detector, mtx, dist, item)
            if not arucoFound is None:
                break

        if not arucoFound is None:
            print("Aruco located!")
            logging.info("Aruco search successful")
            result = await align_to_aruco(drone, camera, detector, mtx, dist, item)

            if result is False:
                print("Aruco alignment failed!")
                logging.info("Aruco alignment failed!")
            else:
                print("Aruco alignment completed!")
                logging.info("Aruco alignment completed!")

            print("Resuming mission")
            logging.info("Resuming mission")
            await drone.mission.start_mission()

        else:
            print("Aruco not found, resuming mission")
            logging.info("Aruco not found, resuming mission")
            await drone.mission.start_mission()

    print("Mission completed! Return to home")
    logging.info(f"Mission completed, return to home")
    await drone.action.return_to_launch()

    camera.stop_recording()
    camera.close()

async def aruco_search(drone: System, camera: Picamera2, detector, mtx, dist, item: MissionItem):
    print("Starting aruco search")

    #get_battery_start = perf_counter()
    #battery = await fc.get_drone_remaining_battery(drone)
    #get_battery_end = perf_counter()
    #logging.info(f"Getting battery took {get_battery_end - get_battery_start} seconds")
    #logging.info(f"Started aruco search with {battery.remaining_percent}% of battery, {battery.voltage_v} Volts")

    round = 1

    get_altitude_start = perf_counter()
    initial_altitude = await fc.get_drone_altitude(drone)
    get_altitude_end = perf_counter()
    logging.info(f"Getting altitude took {get_altitude_end - get_altitude_start} seconds")

    while round < ARUCO_SEARCH_ROUNDS:

        print(f"Search round number {round}")
        logging.info(f"Search round number {round}")
        look_for_aruco_start = perf_counter()
        arucoFound = utilities.look_for_aruco_image(camera, detector, mtx, dist)
        look_for_aruco_end = perf_counter()

        if not arucoFound is None:
            logging.info(f"Finding Aruco before search took {look_for_aruco_end - look_for_aruco_start} seconds")
            return arucoFound

        radius = ARUCO_SEARCH_RADIUS_MULTIPLIER * round

        print("Starting orbit")
        do_orbit_start = perf_counter()
        await drone.action.do_orbit(radius,
                                    1,
                                    OrbitYawBehavior.HOLD_INITIAL_HEADING,
                                    item.latitude_deg,
                                    item.longitude_deg,
                                    initial_altitude)
        do_orbit_end = perf_counter()
        logging.info(f"Starting orbit took {do_orbit_end - do_orbit_start} seconds")

        target_time = time() + ARUCO_SEARCH_STOP_ADDED_TIME * round

        while time() < target_time:
            arucoFound = utilities.look_for_aruco_image(camera, detector, mtx, dist)
            # Double check to avoid problems due to camera angle
            if not arucoFound is None:
                print("Spotted marker")
                logging.info("Spotted marker")
                await drone.action.hold()
                target_time = target_time + ARUCO_SEARCH_SPOTTED_ADDED_TIME
                await asyncio.sleep(2)

                arucoFound = utilities.look_for_aruco_image(camera, detector, mtx, dist)
                if not arucoFound is None:
                    print("Aruco found")
                    logging.info("Aruco found")
                    return arucoFound
                else:
                    print("False positive, resuming orbit")
                    logging.info("False positive, resuming orbit")
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
    logging.info("Aruco search failed")
    await asyncio.sleep(5)
    return None

async def align_to_aruco(drone: System, camera: Picamera2, detector, mtx, dist, item: MissionItem):
    print("Starting alignment procedure")
    logging.info("Starting alignment procedure")
    #battery = await fc.get_drone_remaining_battery(drone)
    #logging.info(f"Started aruco alignment with {battery.remaining_percent}% of battery, {battery.voltage_v} Volts")

    aruco_detection_failures = 0
    procedure_failures = 0
    lined_up = False

    get_initial_position_start = perf_counter()
    initial_position = await fc.get_drone_global_position(drone)
    get_initial_position_end = perf_counter()
    logging.info(f"Getting current global position took {get_initial_position_end - get_initial_position_start} seconds")

    heading = await fc.get_drone_heading(drone)
    initial_heading = heading

    while True:
        get_aruco_distances_start = perf_counter()
        aruco_distances = utilities.get_aruco_distances_and_yaw_image(camera, detector, mtx, dist)
        get_aruco_distances_end = perf_counter()
        logging.info(f"Getting Aruco distances took {get_aruco_distances_end - get_aruco_distances_start} seconds")

        if aruco_distances is None:
            aruco_detection_failures += 1
            print(f"Aruco not found for {aruco_detection_failures} times")
            if aruco_detection_failures >= ALIGNMENT_MAX_DETECTION_FAILURES and procedure_failures <= ALIGNMENT_MAX_PROCEDURE_FAILURES:
                print("Aruco detection failed, retry from inital position")
                await drone.action.goto_location(initial_position.latitude_deg, initial_position.longitude_deg, initial_position.absolute_altitude_m, initial_heading)
                aruco_detection_failures = 0
                procedure_failures += 1
                print("Awaiting the drone to navigate to the initial position")
                await asyncio.sleep(5)
            elif aruco_detection_failures >= ALIGNMENT_MAX_DETECTION_FAILURES and procedure_failures >= ALIGNMENT_MAX_PROCEDURE_FAILURES:
                print("Aruco detection failed multiple times, aborting alignment")
                return False
            continue

        # Security check to prevent false positives from crashing the drone
        # Calculated as the maximum distance of a marker in the field of view at top altitude of 9 metres
        if aruco_distances[0] > ALIGNMENT_MAX_ARUCO_DISTANCE or aruco_distances[1] > ALIGNMENT_MAX_ARUCO_DISTANCE: continue

        aruco_detection_failures = 0
        current_position = await fc.get_drone_global_position(drone)

        calculate_target_position_start = perf_counter()
        target_coordinates = await utilities.calculate_target_global_position(drone, aruco_distances)
        calculate_target_position_end = perf_counter()
        logging.info(f"Calculating target global position took {calculate_target_position_end - calculate_target_position_start} seconds")

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

        goto_start = perf_counter()
        await drone.action.goto_location(target_coordinates.latitude_deg, target_coordinates.longitude_deg, target_coordinates.absolute_altitude_m, heading)
        goto_end = perf_counter()
        logging.info(f"Issuing goto command took {goto_end - goto_start} seconds")


def buildArucoDetector():
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    arucoParams = aruco.DetectorParameters()

    # Detector parameters tuning
    arucoParams.adaptiveThreshWinSizeMax = ARUCO_DETECTOR_THRESHOLD_WINDOW_MAX
    arucoParams.adaptiveThreshWinSizeMin = ARUCO_DETECTOR_THRESHOLD_WINDOW_MIN
    arucoParams.adaptiveThreshWinSizeStep = ARUCO_DETECTOR_THRESHOLD_WINDOW_STEP
    # The marker side is 50 pixels at 8 meters of distance, so 170 pixels (safety margin) divided by 1640
    arucoParams.minMarkerPerimeterRate = ARUCO_DETECTOR_MIN_MARKER_PERIMETER_RATE
    
    return aruco.ArucoDetector(arucoDict, arucoParams)

if __name__ == "__main__":
    # Start the main function
    asyncio.run(run())