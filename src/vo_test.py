# vo_bench_test.py
# Bench test: send ArUco visual odometry to PX4 and compare with PX4's own estimate.
# Does NOT arm, does NOT start offboard. Carry the drone by hand over the marker.

import asyncio
import json
import os
import time
from contextlib import suppress

import cv2.aruco as aruco
import numpy as np
from mavsdk_grpc import System
from picamera2 import Picamera2

from aruco_pose import scale_intrinsic_matrix
import simulated.fc_simulated as fc
from ev_delay_test import TimedVideo, DelayProbeVO
PRINT_PERIOD = 1.0


def build_detector():
    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 15
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 15
    params.minMarkerPerimeterRate = 0.1
    return aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_1000), params)


async def read_params(drone):
    for name in ("EKF2_EV_CTRL", "EKF2_GPS_CTRL"):
        try:
            print(f"   {name} = {await drone.param.get_param_int(name)}")
        except Exception as e:
            print(f"   {name}: could not read ({e})")
    print("   (EKF2_EV_CTRL bits: 1=horiz pos, 2=vert pos, 4=velocity, 8=yaw; must be non-zero)")


async def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "config.json")) as f:
        config = json.load(f)

    drone = System()
    print("Connecting...")
    await drone.connect(system_address=config["system_address"])
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected")
            break

    print("-- PX4 parameters:")
    await read_params(drone)

    # Camera: same configuration as the flight code
    w, h = config["camera"]["resolution"]
    camera = Picamera2()
    camera.configure(camera.create_video_configuration(
        {"size": (w, h), "format": "YUV420"},
        {"size": (w, h), "format": "YUV420"}))
    camera.start()
    video = TimedVideo(camera)

    vision = DelayProbeVO(
        video, build_detector(),
        scale_intrinsic_matrix(
            config["camera"]["intrinsic_matrix"],
            config["camera"]["calibration_resolution"],
            config["camera"]["resolution"],
        ),
        np.array(config["camera"]["distortion_coefficients"]),
        marker_size=config["aruco_size"], marker_id=None,
        camera_to_body=np.array(config["camera"]["camera_to_body"]),
        camera_position_body=np.array(config["camera"]["camera_position_body"]),
        probe_csv="probe.csv"
    )
    vision_task = asyncio.create_task(vision.run(drone))

    try:
        print("-- Hold the drone still with the marker in view...")
        await vision.wait_ready()
        print("-- Marker detected. Anchoring the marker frame to PX4's local NED...")
        vision.enable_frame_initialization()
        await vision.wait_frame_ready()
        vision.enable_fusion()
        print("-- Sending vision estimates. Now move the drone by hand. Ctrl+C to stop.\n")

        t0 = time.perf_counter()
        last_count, last_t = 0, t0

        while True:
            await asyncio.sleep(PRINT_PERIOD)
            now = time.perf_counter()

            rate = (vision.sent_count - last_count) / (now - last_t)
            last_count, last_t = vision.sent_count, now

            sent = vision.last_sent
            obs = vision.get_observation(max_age=0.5)
            if sent is None or obs is None:
                print(f"[{now - t0:6.1f}s] no marker / nothing sent  (rate {rate:.1f} Hz)")
                continue

            pv = await fc.get_drone_ned_position_velocity(drone)
            att = await fc.get_drone_attitude_euler(drone)

            vp = np.array([sent.position_body.x_m,
                           sent.position_body.y_m,
                           sent.position_body.z_m])
            px = np.array([pv.position.north_m, pv.position.east_m, pv.position.down_m])
            dyaw = (np.degrees(sent.angle_body.yaw_rad) - att.yaw_deg + 180) % 360 - 180

            print(f"[{now - t0:6.1f}s] rate {rate:5.1f} Hz | "
                  f"dist to marker {np.linalg.norm(obs.tvec):5.2f} m | "
                  f"VISION NED {np.round(vp, 2)} | PX4 NED {np.round(px, 2)} | "
                  f"|diff| {np.linalg.norm(vp - px):.2f} m | yaw diff {dyaw:+.1f} deg")
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        vision.close_probe()
        vision.disable_fusion()
        vision_task.cancel()
        with suppress(asyncio.CancelledError):
            await vision_task
        camera.stop()
        camera.close()
        print("\nStopped.")


if __name__ == "__main__":
    asyncio.run(main())