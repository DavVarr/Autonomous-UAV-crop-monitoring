"""Measure camera, ArUco, and alignment-trajectory latency without flight control.

Run from the repository root on the companion computer:
    python3 src/vision_pipeline_test.py --duration 60

The script starts the production camera stream and video recording, runs the
production VisualOdometry detector, and evaluates the production trajectory
planner against a fixed synthetic drone pose. It never connects to or commands
a real drone.
"""
import argparse
import asyncio
import csv
from datetime import datetime
import json
from pathlib import Path
import queue
import time
from types import SimpleNamespace

import cv2.aruco as aruco
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

import camera as camera_module
from aruco_pose import scale_intrinsic_matrix
from simulated import fov_traj_alignment as alignment
from simulated.VisualOdometry import VisualOdometry


class TimedVideo(camera_module.Video):
    """Video source that records sensor exposure and capture timing."""

    def __init__(self, camera):
        super().__init__(camera)
        self.clock_id = None
        self.first_capture_ns = None
        self.last_frame = None
        self.frame_number = 0

    def _select_clock(self, sensor_timestamp_ns):
        candidates = []
        for clock_id in (time.CLOCK_MONOTONIC, time.CLOCK_BOOTTIME):
            age_ns = time.clock_gettime_ns(clock_id) - sensor_timestamp_ns
            if 0 <= age_ns < 60_000_000_000:
                candidates.append((age_ns, clock_id))
        if not candidates:
            raise RuntimeError(
                "Camera SensorTimestamp does not match CLOCK_MONOTONIC or "
                "CLOCK_BOOTTIME within 60 seconds; cannot measure frame age"
            )
        return min(candidates)[1]

    def frame(self):
        request_start_ns = time.perf_counter_ns()
        request = self.camera.capture_request()
        try:
            metadata = request.get_metadata()
            frame = camera_module._main_frame(self.camera, request)
        finally:
            request.release()
        request_end_ns = time.perf_counter_ns()

        sensor_timestamp_ns = int(metadata["SensorTimestamp"])
        exposure_us = int(metadata.get("ExposureTime", 0))
        if self.clock_id is None:
            self.clock_id = self._select_clock(sensor_timestamp_ns)

        exposure_midpoint_ns = sensor_timestamp_ns + exposure_us * 500
        if self.first_capture_ns is None:
            self.first_capture_ns = exposure_midpoint_ns

        self.last_frame = {
            "number": self.frame_number,
            "capture_ns": exposure_midpoint_ns,
            "capture_s": (exposure_midpoint_ns - self.first_capture_ns) / 1e9,
            "request_ms": (request_end_ns - request_start_ns) / 1e6,
            "capture_return_age_ms": (
                time.clock_gettime_ns(self.clock_id) - exposure_midpoint_ns
            ) / 1e6,
        }
        self.frame_number += 1
        return frame


class TimedVisualOdometry(VisualOdometry):
    """Run the real VO detector and send per-frame work to the monitor."""

    def __init__(self, *args, frame_results, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_results = frame_results

    def _detect(self, frame, timestamp):
        frame_info = self.video.last_frame
        if frame_info is None:
            raise RuntimeError("Visual odometry ran before a camera frame arrived")

        detect_start_ns = time.clock_gettime_ns(self.video.clock_id)
        observation = super()._detect(frame, timestamp)
        detect_end_ns = time.clock_gettime_ns(self.video.clock_id)
        self.frame_results.put({
            **frame_info,
            "observation": observation,
            "detect_start_ns": detect_start_ns,
            "detect_ms": (detect_end_ns - detect_start_ns) / 1e6,
            "detect_done_age_ms": (
                detect_end_ns - frame_info["capture_ns"]
            ) / 1e6,
        })
        return observation


class FixedPoseTelemetry:
    """Only implements the read-only telemetry used by VisualOdometry."""

    def __init__(self, start_down_m, roll_deg, pitch_deg, heading_deg):
        self._position_velocity = SimpleNamespace(
            position=SimpleNamespace(
                north_m=0.0, east_m=0.0, down_m=start_down_m
            ),
            velocity=SimpleNamespace(
                north_m_s=0.0, east_m_s=0.0, down_m_s=0.0
            ),
        )
        self._attitude = SimpleNamespace(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=heading_deg,
        )

    async def position_velocity_ned(self):
        yield self._position_velocity

    async def attitude_euler(self):
        yield self._attitude


class FixedPoseDrone:
    """A fake telemetry-only object; it has no action or offboard interface."""

    def __init__(self, start_down_m, roll_deg, pitch_deg, heading_deg):
        self.telemetry = FixedPoseTelemetry(
            start_down_m, roll_deg, pitch_deg, heading_deg
        )


def build_detector():
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 15
    parameters.adaptiveThreshWinSizeMax = 45
    parameters.adaptiveThreshWinSizeStep = 15
    parameters.minMarkerPerimeterRate = 0.1
    return aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_1000),
        parameters,
    )


def compute_trajectory_diagnostic(vision, observation, pose, width, height):
    """Evaluate the same production trajectory helpers used during alignment."""
    heading = pose.yaw_deg
    camera_to_body = vision.camera_to_body
    marker_ned = vision.marker_position_ned
    marker_corners_ned = vision.marker_corners_ned
    if marker_ned is None or marker_corners_ned is None:
        return None

    marker_world = alignment._rotate_frame(marker_ned, heading)
    marker_corners_world = np.array([
        alignment._rotate_frame(corner, heading)
        for corner in marker_corners_ned
    ])
    position_ned = np.array([0.0, 0.0, pose.down_m])
    position = alignment._rotate_frame(position_ned, heading)
    velocity = np.zeros(3)

    observed = alignment._observe(
        observation, vision.mtx, vision.marker_size, pose, heading,
        camera_to_body,
    )
    if observed is None:
        return None
    _, _, relative_corners = observed

    target_altitude = 1.0
    target = marker_world - [0.0, 0.0, target_altitude]
    trajectory = alignment._make_trajectory(
        position, velocity, np.zeros(3), target,
        max_velocity=3.0, max_acceleration=3.0, max_jerk=2.0,
    )

    direct_safe = trajectory is not None and alignment._trajectory_safe(
        trajectory, marker_corners_world, vision.mtx, camera_to_body,
        width, height, 0.02, 30,
    )
    phase1_duration = None
    if not direct_safe:
        limits = alignment._phase1_acceleration_limits(
            relative_corners,
            marker_world[:2] - position[:2],
            vision.mtx,
            camera_to_body,
            width,
            height,
            30,
            0.08,
            1.5,
        )
        phase1 = alignment._make_trajectory(
            position[:2], velocity[:2], np.zeros(2), marker_world[:2],
            max_velocity=3.0, max_acceleration=limits, max_jerk=2.0,
        )
        if phase1 is not None:
            phase1_duration = phase1.duration

    return {
        "target_distance_m": float(np.linalg.norm(target - position)),
        "direct_duration_s": (
            None if trajectory is None else float(trajectory.duration)
        ),
        "direct_safe": direct_safe,
        "phase1_duration_s": phase1_duration,
    }


CSV_FIELDS = (
    "frame", "capture_s", "frame_interval_ms", "capture_request_ms",
    "capture_return_age_ms", "capture_to_detect_start_ms", "detect_ms",
    "detect_done_age_ms", "detected", "trajectory_ms",
    "capture_to_trajectory_ms", "target_distance_m", "direct_duration_s",
    "direct_safe", "phase1_duration_s",
)


def write_result(writer, record, previous_capture_ns, vision, pose,
                 width, height):
    trajectory_start_ns = time.perf_counter_ns()
    trajectory_result = None
    if record["observation"] is not None and vision.marker_position_ned is not None:
        trajectory_result = compute_trajectory_diagnostic(
            vision, record["observation"], pose, width, height
        )
    trajectory_end_ns = time.perf_counter_ns()

    capture_ns = record["capture_ns"]
    row = {
        "frame": record["number"],
        "capture_s": f'{record["capture_s"]:.6f}',
        "frame_interval_ms": (
            "" if previous_capture_ns is None else
            f'{(capture_ns - previous_capture_ns) / 1e6:.2f}'
        ),
        "capture_request_ms": f'{record["request_ms"]:.2f}',
        "capture_return_age_ms": f'{record["capture_return_age_ms"]:.2f}',
        "capture_to_detect_start_ms": f'{(record["detect_start_ns"] - capture_ns) / 1e6:.2f}',
        "detect_ms": f'{record["detect_ms"]:.2f}',
        "detect_done_age_ms": f'{record["detect_done_age_ms"]:.2f}',
        "detected": record["observation"] is not None,
        "trajectory_ms": f'{(trajectory_end_ns - trajectory_start_ns) / 1e6:.2f}',
        "capture_to_trajectory_ms": (
            "" if trajectory_result is None else
            f'{(time.clock_gettime_ns(vision.video.clock_id) - capture_ns) / 1e6:.2f}'
        ),
        "target_distance_m": "",
        "direct_duration_s": "",
        "direct_safe": "",
        "phase1_duration_s": "",
    }
    if trajectory_result is not None:
        row["target_distance_m"] = f'{trajectory_result["target_distance_m"]:.3f}'
        duration = trajectory_result["direct_duration_s"]
        row["direct_duration_s"] = "" if duration is None else f"{duration:.3f}"
        row["direct_safe"] = trajectory_result["direct_safe"]
        phase1 = trajectory_result["phase1_duration_s"]
        row["phase1_duration_s"] = "" if phase1 is None else f"{phase1:.3f}"
    writer.writerow(row)
    return row, capture_ns


def percentile_summary(rows, key):
    values = np.asarray([
        float(row[key]) for row in rows if row[key] not in ("", None)
    ])
    if not len(values):
        return "n/a"
    return (
        f"p50 {np.percentile(values, 50):.1f} ms, "
        f"p95 {np.percentile(values, 95):.1f} ms, max {values.max():.1f} ms"
    )


async def run_test(args, csv_path, video_path):
    config_path = Path(__file__).resolve().parents[1] / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    camera = Picamera2()
    recording_started = False
    camera_started = False
    vision_task = None
    frame_results = queue.SimpleQueue()
    try:
        width, height = config["camera"]["resolution"]
        main_stream = {
            "size": (width, height),
            "format": "YUV420",
        }
        camera.configure(camera.create_video_configuration(
            main_stream, encode="main"
        ))

        if not args.no_recording:
            encoder = H264Encoder(10_000_000)
            camera.start_recording(encoder, str(video_path))
            recording_started = True
            camera_started = True
        else:
            camera.start()
            camera_started = True

        video = TimedVideo(camera)
        pose = SimpleNamespace(
            down_m=args.start_down_m,
            roll_deg=args.roll_deg,
            pitch_deg=args.pitch_deg,
            yaw_deg=args.heading_deg,
        )
        drone = FixedPoseDrone(
            args.start_down_m, args.roll_deg, args.pitch_deg, args.heading_deg
        )
        vision = TimedVisualOdometry(
            video,
            build_detector(),
            scale_intrinsic_matrix(
                config["camera"]["intrinsic_matrix"],
                config["camera"]["calibration_resolution"],
                config["camera"]["resolution"],
            ),
            np.asarray(config["camera"]["distortion_coefficients"]),
            marker_size=config["aruco_size"],
            marker_id=args.marker_id,
            camera_to_body=np.asarray(config["camera"]["camera_to_body"]),
            camera_position_body=np.asarray(
                config["camera"]["camera_position_body"]
            ),
            frame_results=frame_results,
        )
        vision.enable_frame_initialization()
        vision_task = asyncio.create_task(vision.run(drone))

        print("Camera pipeline running; no real drone is connected or commanded.")
        print(f"Recording: {'enabled' if recording_started else 'disabled'}")
        print(f"CSV: {csv_path}")
        if recording_started:
            print(f"Video: {video_path}")
        print(f"Duration: {args.duration:.1f} s")
        print("Place the calibrated ArUco marker in view; Ctrl+C stops the test.")

        rows = []
        previous_capture_ns = None
        deadline = asyncio.get_running_loop().time() + args.duration
        last_progress = asyncio.get_running_loop().time()
        with csv_path.open("x", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            while asyncio.get_running_loop().time() < deadline:
                if vision_task.done():
                    vision_task.result()
                    raise RuntimeError("Visual-odometry worker exited unexpectedly")

                drained = False
                while True:
                    try:
                        record = frame_results.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    row, previous_capture_ns = write_result(
                        writer, record, previous_capture_ns, vision, pose,
                        width, height,
                    )
                    rows.append(row)

                now = asyncio.get_running_loop().time()
                if now - last_progress >= 5.0:
                    latest_age = (
                        rows[-1]["detect_done_age_ms"] if rows else "n/a"
                    )
                    print(
                        f"{len(rows)} frames, "
                        f"{sum(bool(row['detected']) for row in rows)} marker "
                        f"detections; latest capture-to-detect age "
                        f"{latest_age} ms"
                    )
                    csv_file.flush()
                    last_progress = now
                if not drained:
                    await asyncio.sleep(0.005)
            csv_file.flush()

        print("\nResults")
        print(f"Frames analyzed: {len(rows)}")
        print(
            f"Frames with marker: "
            f"{sum(bool(row['detected']) for row in rows)}"
        )
        for label, key in (
            ("Capture-to-detect-start age", "capture_to_detect_start_ms"),
            ("Detection and pose time", "detect_ms"),
            ("Capture-to-detection-done age", "detect_done_age_ms"),
            ("Trajectory computation", "trajectory_ms"),
            ("Capture-to-trajectory age", "capture_to_trajectory_ms"),
        ):
            print(f"{label}: {percentile_summary(rows, key)}")

        if rows:
            print("\nFive-second capture-to-detection age averages:")
            start_s = float(rows[0]["capture_s"])
            bins = {}
            for row in rows:
                key = int((float(row["capture_s"]) - start_s) // 5)
                bins.setdefault(key, []).append(float(row["detect_done_age_ms"]))
            for key, values in sorted(bins.items()):
                print(
                    f"  {key * 5:3d}-{key * 5 + 5:3d} s: "
                    f"mean {np.mean(values):.1f} ms, max {np.max(values):.1f} ms"
                )
    finally:
        try:
            if vision_task is not None:
                vision_task.cancel()
                try:
                    await vision_task
                except asyncio.CancelledError:
                    pass
        finally:
            try:
                if camera_started:
                    if recording_started:
                        camera.stop_recording()
                    else:
                        camera.stop()
            finally:
                camera.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure the real camera/VO/trajectory pipeline without "
            "connecting to or controlling a drone."
        )
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--no-recording",
        action="store_true",
        help="Skip H.264 recording to compare the camera pipeline without it.",
    )
    parser.add_argument("--marker-id", type=int, default=None)
    parser.add_argument("--start-down-m", type=float, default=0.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--heading-deg", type=float, default=0.0)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and greater than zero")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"vision_pipeline_{datetime.now():%Y%m%d_%H%M%S}"
    csv_path = args.output_dir / f"{prefix}.csv"
    video_path = args.output_dir / f"{prefix}.h264"
    if csv_path.exists() or (not args.no_recording and video_path.exists()):
        raise FileExistsError(f"Diagnostic output already exists: {prefix}")
    asyncio.run(run_test(args, csv_path, video_path))


if __name__ == "__main__":
    main()
