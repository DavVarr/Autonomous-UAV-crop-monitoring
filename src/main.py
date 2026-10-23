import asyncio
import json
import os
from time import perf_counter, time
import logging
from math import fabs
from mavsdk_grpc import System
from mavsdk_grpc.offboard import VelocityNedYaw, OffboardError, PositionNedYaw
from mavsdk_grpc.mission import MissionItem, MissionError
from mavsdk_grpc.action import OrbitYawBehavior
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import numpy as np
import cv2.aruco as aruco
from contextlib import suppress

import camera as camera_module
import simulated.fc_simulated as fc
import simulated.utilities_simulated as utilities
from simulated.fov_traj_alignment import align_to_aruco_ruckig_adaptive
from simulated.VisualOdometry import VisualOdometry

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

TAKE_OFF_ALTITUDE = 3.0

async def run():

    #Prepare logs
    logging.basicConfig(filename="log.txt", level=logging.INFO, format="%(asctime)s %(message)s")

    with open(os.path.join(os.path.dirname(__file__), "..", "config.json")) as config_file:
        config = json.load(config_file)

    if len(config["waypoints"]) == 0:
        raise ValueError("At least one waypoint must be set in config.json")

    drone = System()
    print("Connecting...")
    await drone.connect(system_address=config["system_address"])

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            logging.info("Drone connection estabilished")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    # Preparing camera
    camera = Picamera2()
    frame_width, frame_height = config["camera"]["resolution"]
    main_stream = {"size": (frame_width, frame_height)}
    lores_stream = {"size": (frame_width, frame_height)}
    video_config = camera.create_video_configuration(main_stream, lores_stream, encode="lores")
    camera.configure(video_config)
    encoder = H264Encoder(10000000)

    mtx = np.array(config["camera"]["intrinsic_matrix"])
    dist = np.array(config["camera"]["distortion_coefficients"])
    camera_to_body = np.array(config["camera"]["camera_to_body"])
    camera_position_body = np.array(config["camera"]["camera_position_body"])
    marker_size = config["aruco_size"]

    #Preparing Aruco detector
    detector = buildArucoDetector()

    camera.start_recording(encoder, 'video.h264')
    video = camera_module.Video(camera)
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

    for waypoint in config["waypoints"]:
        print(f"Flying to waypoint {waypoint['latitude']}, {waypoint['longitude']}")
        logging.info(f"Flying to waypoint {waypoint['latitude']}, {waypoint['longitude']}")
        await fc.fly_to_global(
            drone,
            waypoint["latitude"],
            waypoint["longitude"],
            waypoint["altitude"],
            heading=waypoint["heading"]
        )
        await asyncio.sleep(1)

        arucoFound = None
        for s in range(ARUCO_SEARCH_TRIES):
            arucoFound, transition_task = await aruco_search(
                drone, video, detector, mtx, dist, marker_size)
            if not arucoFound is None:
                break

        if not arucoFound is None:
            print("Aruco located!")
            vision = VisualOdometry(
                video, detector, mtx, dist, marker_size=marker_size, marker_id=None,
                camera_to_body=camera_to_body, camera_position_body=camera_position_body)
            vision_task = asyncio.create_task(vision.run(drone))

            result = await align_noGPS_with_failsafe(
                drone, vision, transition_task, frame_width, frame_height)

            if result is False:
                print("Aruco alignment failed!")
            else:
                print("Aruco alignment completed!")

            vision_task.cancel()
            with suppress(asyncio.CancelledError):
                await vision_task

        else:
            print("Aruco not found, going to next waypoint")

    print("Mission completed! Return to home")
    logging.info(f"Mission completed, return to home")
    await drone.offboard.stop()
    await drone.action.return_to_launch()

    camera.stop_recording()
    camera.close()

async def _wait_marker_loss(vision, timeout=2.0):
    while True:
        if vision.get_observation(max_age=timeout) is None:
            return

        await asyncio.sleep(0.05)

async def _wait_alignment_entry(vision, transition_task,
                                maximum_angle=np.deg2rad(30.),
                                low_speed=0.2,
                                stable_time=0.15):
    good_since = None
    cos_limit = np.cos(maximum_angle)

    while True:
        obs = vision.get_observation(max_age=0.15)
        state = transition_task.handoff
        good = False

        if obs is not None and all(k in state for k in ("p", "v")):
            p = state["p"]
            v = state["v"]

            to_target = transition_task.target_xy - p
            speed = np.linalg.norm(v)
            distance = np.linalg.norm(to_target)

            if speed < low_speed or distance < 0.1:
                motion_ok = True
            else:
                direction_cos = np.dot(v, to_target) / (speed * distance)
                motion_ok = direction_cos >= cos_limit

            good = motion_ok

        if good:
            if good_since is None:
                good_since = perf_counter()
            elif perf_counter() - good_since >= stable_time:
                return
        else:
            good_since = None

        await asyncio.sleep(0.05)

async def align_noGPS_with_failsafe(drone, vision : VisualOdometry, transition_task, frame_width, frame_height):
    gps_ctrl = await drone.param.get_param_int("EKF2_GPS_CTRL")

    await vision.wait_ready()
    pv = await fc.get_drone_ned_position_velocity(drone)
    return_down = pv.position.down_m

    initial_state = None

    if transition_task is not None:
        await _wait_alignment_entry(vision, transition_task)
        initial_state = transition_task.handoff
        transition_task.cancel()
        with suppress(asyncio.CancelledError):
            await transition_task

    vision.enable_frame_initialization()
    await vision.wait_frame_ready()

    print("-- Disabling GPS position fusion")
    await drone.param.set_param_int("EKF2_GPS_CTRL", 4)
    await utilities.wait_gps_ctrl(drone)
    vision.enable_fusion()

    operation_task = asyncio.create_task(align_ruckig_fov_safe(
            drone, vision, initial_state, return_down, frame_width, frame_height)
    )
    loss_task = asyncio.create_task( _wait_marker_loss(vision, 2.0))

    done, _ = await asyncio.wait(
        (operation_task, loss_task),
        return_when=asyncio.FIRST_COMPLETED,
    )

    if loss_task in done:
        print("-- ArUco lost for 2 seconds, aborting")

        operation_task.cancel()
        with suppress(asyncio.CancelledError):
            await operation_task

        vision.disable_fusion()

        print("-- Restoring GPS fusion")
        await drone.param.set_param_int("EKF2_GPS_CTRL", gps_ctrl)

        print("-- Recovering altitude using GPS")
        pos = await fc.get_drone_ned_position(drone)
        await fc.fly_to_ned_smooth(
            drone, PositionNedYaw(pos.north_m, pos.east_m, return_down,float("nan")), velocity_threshold=0.3, max_jerk=4
        )
        return False

    success = await operation_task

    print("-- Restoring GPS fusion")
    await drone.param.set_param_int(
        "EKF2_GPS_CTRL", gps_ctrl
    )
    vision.disable_fusion()

    loss_task.cancel()
    with suppress(asyncio.CancelledError):
        await loss_task

    return success

async def align_ruckig_fov_safe(drone,vision,initial_state,return_down,frame_width,frame_height):
    success = await align_to_aruco_ruckig_adaptive(drone,vision,
        frame_width=frame_width, frame_height=frame_height, initial_state=initial_state
    )
    if not success:
        return False
    await asyncio.sleep(5)
    print("-- Alignment complete, returning to altitude")
    pos = await fc.get_drone_ned_position(drone)
    await fc.fly_to_ned_smooth(
        drone, PositionNedYaw(pos.north_m, pos.east_m, return_down, float("nan")), velocity_threshold=0.3, max_jerk=4
    )
    return True

async def aruco_search(drone: System, video: camera_module.Video, detector, mtx, dist, marker_size):
    print("Starting aruco search")
    battery = await fc.get_drone_remaining_battery(drone)
    logging.info(f"Started aruco search with {battery.remaining_percent}% of battery, {battery.voltage_v} Volts")
    round = 1
    initial_altitude = await fc.get_drone_altitude(drone)
    center_ned = await fc.get_drone_ned_position(drone)
    heading = await fc.get_drone_heading(drone)
    while round < ARUCO_SEARCH_ROUNDS:

        arucoFound = utilities.look_for_aruco(video, detector, mtx, dist, marker_size)
        if not arucoFound is None:
            return arucoFound, None

        radius = ARUCO_SEARCH_RADIUS_MULTIPLIER * round

        print("Starting orbit")

        result, transition_task = await utilities.look_for_aruco_orbit(
            drone, video, detector, mtx, dist,
            center_ned, heading, radius, 1, marker_size,
        )
        if result is not None:
            return result, transition_task
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
    return None, None


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
