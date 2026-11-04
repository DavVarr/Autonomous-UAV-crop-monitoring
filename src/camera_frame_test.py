"""Isolate camera/encoder/detection gaps without connecting to a flight controller.

    python3 src/camera_frame_test.py record --duration 120 --output-prefix camera_full
    python3 src/camera_frame_test.py record --duration 120 --output-prefix camera_queued --queued-output
    python3 src/camera_frame_test.py record --duration 120 --output-prefix camera_only --no-detection
    python3 src/camera_frame_test.py record --duration 120 --output-prefix camera_detect --no-recording
    python3 src/camera_frame_test.py analyze camera_full

Uses the production YUV420 stream, 10 Mbit/s H.264 encoder, ArUco parameters,
and pose estimator. Logs all completed camera requests, encoder output timing,
and the subset of frames processed by detection. CSVs are written after stopping
the camera so logging itself does not introduce per-frame disk writes.
"""
import argparse
import asyncio
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np


FIELDS = {
    "camera": ("frame", "sequence", "sensor_ns", "callback_ns", "exposure_us", "frame_duration_us"),
    "encoder": ("frame", "pts_us", "output_start_ns", "output_end_ns", "bytes", "keyframe"),
    "detect": ("frame", "sequence", "sensor_ns", "exposure_us", "request_start_ns",
               "request_end_ns", "detect_start_ns", "detect_end_ns", "markers"),
}


def output_path(prefix, suffix):
    return Path(str(prefix) + suffix)


def clock_ns():
    # libcamera SensorTimestamp uses CLOCK_BOOTTIME on the companion.
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def request_info(request):
    metadata = request.get_metadata()
    return (getattr(request.request, "sequence", None), int(metadata["SensorTimestamp"]),
            int(metadata.get("ExposureTime", 0)), int(metadata.get("FrameDuration", 0)))


async def record(args):
    try:
        import cv2.aruco as aruco
        from picamera2 import Picamera2
        from picamera2.encoders import H264Encoder
        from picamera2.outputs import FileOutput
        import camera as camera_module
        from aruco_pose import my_estimatePoseSingleMarkers, scale_intrinsic_matrix
    except ImportError as error:
        raise SystemExit(f"Recording requires Picamera2 and OpenCV on the Pi: {error}")

    suffixes = [f"_{name}.csv" for name in FIELDS]
    if not args.no_recording:
        suffixes += [".h264", ".pts"]
    for suffix in suffixes:
        path = output_path(args.output_prefix, suffix)
        if path.exists():
            raise FileExistsError(f"Choose a new --output-prefix; {path} already exists")
    if not args.output_prefix.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {args.output_prefix.parent}")

    with (Path(__file__).resolve().parents[1] / "config.json").open() as file:
        config = json.load(file)
    rows = {name: [] for name in FIELDS}

    class TimedOutput(FileOutput):
        def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
            start = clock_ns()
            super().outputframe(frame, keyframe, timestamp, packet, audio)
            rows["encoder"].append((len(rows["encoder"]), timestamp, start, clock_ns(),
                                    len(frame), int(keyframe)))

    def camera_frame(request):
        callback = clock_ns()
        sequence, sensor, exposure, duration = request_info(request)
        rows["camera"].append((len(rows["camera"]), sequence, sensor, callback, exposure, duration))

    if not args.no_detection:
        parameters = aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeMin = 15
        parameters.adaptiveThreshWinSizeMax = 45
        parameters.adaptiveThreshWinSizeStep = 15
        parameters.minMarkerPerimeterRate = 0.1
        detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_1000), parameters)
        mtx = scale_intrinsic_matrix(config["camera"]["intrinsic_matrix"],
                                     config["camera"]["calibration_resolution"],
                                     config["camera"]["resolution"])
        distortion = np.asarray(config["camera"]["distortion_coefficients"])

        def detect(frame):
            start = clock_ns()
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is not None:
                my_estimatePoseSingleMarkers(corners, config["aruco_size"], mtx, distortion)
            return start, clock_ns(), 0 if ids is None else len(ids)

    camera = Picamera2()
    output = None
    started = False
    try:
        width, height = config["camera"]["resolution"]
        camera.configure(camera.create_video_configuration(
            {"size": (width, height), "format": "YUV420"}, encode="main"))
        camera.pre_callback = camera_frame
        print(f"Camera configuration: {camera.camera_configuration()}")
        print(f"Recording: {not args.no_recording}; detection: {not args.no_detection}")
        print(f"Output prefix: {args.output_prefix}; no flight-controller connection")
        if args.no_recording:
            camera.start()
        else:
            output = TimedOutput(str(output_path(args.output_prefix, ".h264")),
                                 pts=str(output_path(args.output_prefix, ".pts")))
            if args.queued_output:
                output = camera_module.QueuedOutput(output)
                print("Queued output: encoder CSV measures file writes in the writer thread")
            camera.start_recording(H264Encoder(10_000_000), output)
        started = True
        deadline = time.perf_counter() + args.duration
        progress = time.perf_counter()
        while time.perf_counter() < deadline:
            if args.no_detection:
                await asyncio.sleep(0.05)
            else:
                start = clock_ns()
                request = camera.capture_request()
                try:
                    sequence, sensor, exposure, _ = request_info(request)
                    frame = camera_module._main_frame(camera, request)
                finally:
                    request.release()
                end = clock_ns()
                detect_start, detect_end, markers = await asyncio.to_thread(detect, frame)
                rows["detect"].append((len(rows["detect"]), sequence, sensor, exposure,
                                       start, end, detect_start, detect_end, markers))
            if time.perf_counter() - progress >= 5:
                print(f"Camera requests: {len(rows['camera'])}; encoder outputs: "
                      f"{len(rows['encoder'])}; detection frames: {len(rows['detect'])}")
                if args.queued_output:
                    print(f"Video queue: {output.queue_depth} frames; error: {output.error}")
                progress = time.perf_counter()
    finally:
        try:
            try:
                if started:
                    if args.no_recording:
                        camera.stop()
                    else:
                        camera.stop_recording()
            finally:
                camera.close()
        finally:
            if args.queued_output and output is not None:
                print(f"Video queue peak: {output.max_queue_depth} frames; error: {output.error}")
            for name, fields in FIELDS.items():
                with output_path(args.output_prefix, f"_{name}.csv").open("x", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(fields)
                    writer.writerows(rows[name])
            print("Timing CSVs saved.")


def values(rows, key):
    return np.asarray([float(row[key]) for row in rows if row[key] != ""])


def distribution(label, data):
    if len(data):
        print(f"  {label}: p50 {np.percentile(data, 50):.2f}, "
              f"p95 {np.percentile(data, 95):.2f}, max {np.max(data):.2f} ms")


def intervals(label, timestamps_ms, gap_ms, origin_ms):
    if len(timestamps_ms) < 2:
        return
    dt = np.diff(timestamps_ms)
    distribution(label, dt)
    gaps = np.flatnonzero(dt > gap_ms)
    print(f"  Gaps > {gap_ms:g} ms: {len(gaps)}; non-increasing timestamps: {np.sum(dt <= 0)}")
    for i in gaps[:10]:
        print(f"    {(timestamps_ms[i] - origin_ms) / 1000:.3f}s -> "
              f"{(timestamps_ms[i + 1] - origin_ms) / 1000:.3f}s: {dt[i]:.1f} ms")


def analyze(prefix, gap_ms=100):
    data = {}
    for name in FIELDS:
        with output_path(prefix, f"_{name}.csv").open() as file:
            data[name] = list(csv.DictReader(file))
    camera, encoder, detection = (data[name] for name in FIELDS)
    sensors = values(camera, "sensor_ns") / 1e6
    origin = sensors[0] if len(sensors) else 0
    print(f"\nCamera: {len(camera)} completed requests (all frames, not just detection)")
    intervals("Sensor intervals", sensors, gap_ms, origin)
    intervals("Callback arrival intervals", values(camera, "callback_ns") / 1e6, gap_ms, origin)
    if camera:
        distribution("Sensor-to-callback age", (values(camera, "callback_ns") - values(camera, "sensor_ns")) / 1e6)
        distribution("Reported frame duration", values(camera, "frame_duration_us") / 1000)
        distribution("Exposure", values(camera, "exposure_us") / 1000)
        sequence = values(camera, "sequence")
        if len(sequence) == len(camera):
            steps = np.diff(sequence)
            print(f"  Camera sequence: {np.sum(steps > 1)} jumps, "
                  f"{int(np.maximum(steps - 1, 0).sum())} missing sequence numbers, "
                  f"{np.sum(steps <= 0)} repeats/backward steps")
        else:
            print("  Camera sequence unavailable on this libcamera version")

    print(f"\nEncoder: {len(encoder)} output callbacks")
    pts = values(encoder, "pts_us") / 1000
    # Encoder PTS can have a different zero than SensorTimestamp: compare intervals.
    intervals("Encoded PTS intervals (encoder-relative time)", pts, gap_ms, pts[0] if len(pts) else 0)
    intervals("Output arrival intervals (camera-relative time)", values(encoder, "output_start_ns") / 1e6, gap_ms, origin)
    distribution("Output write/callback time", (values(encoder, "output_end_ns") - values(encoder, "output_start_ns")) / 1e6)
    if encoder and len(pts) != len(encoder):
        print("  Some encoder outputs have no PTS")

    print(f"\nDetection: {len(detection)} processed frames")
    intervals("Selected sensor intervals", values(detection, "sensor_ns") / 1e6, gap_ms, origin)
    if detection:
        distribution("Capture request/copy", (values(detection, "request_end_ns") - values(detection, "request_start_ns")) / 1e6)
        distribution("Detection + pose", (values(detection, "detect_end_ns") - values(detection, "detect_start_ns")) / 1e6)
        midpoint = values(detection, "sensor_ns") + values(detection, "exposure_us") * 500
        distribution("Exposure midpoint-to-detection age", (values(detection, "detect_end_ns") - midpoint) / 1e6)
        print(f"  Frames with a marker: {np.sum(values(detection, 'markers') > 0)}")
    print("\nDetection can legitimately select fewer frames than the camera produces.")
    print("Sensor gaps implicate camera/source delivery; callback-only gaps suggest request processing stalls.")
    print("Stable camera timing but gapped encoded PTS implicates encoding/output; long writes show output backpressure.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    recorder = commands.add_parser("record", help="Camera-only bench test; no MAVSDK connection")
    recorder.add_argument("--duration", type=float, default=120)
    recorder.add_argument("--output-prefix", type=Path,
                          default=Path(f"camera_frames_{datetime.now():%Y%m%d_%H%M%S_%f}"))
    recorder.add_argument("--no-recording", action="store_true")
    recorder.add_argument("--no-detection", action="store_true")
    recorder.add_argument("--queued-output", action="store_true",
                          help="Write video in a worker; encoder CSV still measures actual file-write time")
    analyzer = commands.add_parser("analyze", help="Analyze timing CSVs on the Pi or laptop")
    analyzer.add_argument("prefix", type=Path)
    analyzer.add_argument("--gap-ms", type=float, default=100)
    args = parser.parse_args()
    if args.command == "record":
        if not np.isfinite(args.duration) or args.duration <= 0:
            parser.error("--duration must be finite and positive")
        if args.queued_output and args.no_recording:
            parser.error("--queued-output requires recording")
        try:
            asyncio.run(record(args))
        except KeyboardInterrupt:
            print("Stopped by user.")
        analyze(args.output_prefix)
    else:
        if not np.isfinite(args.gap_ms) or args.gap_ms <= 0:
            parser.error("--gap-ms must be finite and positive")
        analyze(args.prefix, args.gap_ms)


if __name__ == "__main__":
    main()
