# ev_delay_test.py
"""
Estimate EKF2_EV_DELAY for the ArUco visual odometry.

Recording (on the Pi, props removed, disarmed):
    python3 src/ev_delay_test.py record --duration 60 --setpoint-hz 50 --output-prefix trial01
Analysis (Pi or laptop; the ulog part needs `pip install pyulog`):
    python3 src/ev_delay_test.py csv trial01.csv
    python3 src/ev_delay_test.py ulog trial01.ulg --csv trial01.csv
"""
import argparse
import asyncio
import csv
from datetime import datetime
import json
from pathlib import Path
import threading
import time

import numpy as np

# ---------------------------------------------------------------- recording
try:
    import camera as camera_module
    from simulated.VisualOdometry import VisualOdometry, _euler321
    from mavsdk_grpc.mocap import AngleBody, VisionPositionEstimate
    HAVE_FLIGHT = True
except ImportError as error:  # analysis on a laptop: recording classes unavailable
    HAVE_FLIGHT = False
    FLIGHT_IMPORT_ERROR = str(error)

if HAVE_FLIGHT:

    class TimedVideo(camera_module.Video):
        """Video that also remembers the sensor timestamp of the last frame."""

        def __init__(self, camera):
            super().__init__(camera)
            self.clock_id = None       # clock matching SensorTimestamp
            self.capture_ns = None     # exposure time of the last frame
            self.exposure_us = 0
            self.capture_unix_us = None
            self.frame_number = -1
            self.request_ms = self.return_age_ms = 0.0

        def _pick_clock(self, sensor_ns):
            # libcamera SensorTimestamp uses CLOCK_BOOTTIME; prefer it when valid.
            for cid in (time.CLOCK_BOOTTIME, time.CLOCK_MONOTONIC):
                age = time.clock_gettime_ns(cid) - sensor_ns
                if 0 <= age < 60_000_000_000:
                    return cid
            raise RuntimeError("SensorTimestamp matches neither CLOCK_BOOTTIME "
                               "nor CLOCK_MONOTONIC; cannot measure capture latency")

        def frame(self):
            start = time.perf_counter_ns()
            request = self.camera.capture_request()
            try:
                meta = request.get_metadata()
                frame = camera_module._main_frame(self.camera, request)
            finally:
                request.release()
            self.request_ms = (time.perf_counter_ns() - start) / 1e6
            ts = meta["SensorTimestamp"]
            if self.clock_id is None:
                self.clock_id = self._pick_clock(ts)
            self.capture_ns = ts
            self.exposure_us = meta.get("ExposureTime", 0)
            # MAVSDK's nonzero time_usec expects UNIX epoch, not camera uptime.
            before = time.time_ns()
            now = time.clock_gettime_ns(self.clock_id)
            after = time.time_ns()
            midpoint = ts + int(self.exposure_us * 500)
            self.capture_unix_us = ((before + after) // 2 - (now - midpoint)) // 1000
            self.return_age_ms = (now - midpoint) / 1e6
            self.frame_number += 1
            return frame

    class DelayProbeVO(VisualOdometry):
        """VisualOdometry that logs per-message latency to a CSV.

        send_capture_usec=True supplies UNIX capture time to MAVSDK. This is
        only useful after validating end-to-end clock synchronization; the flag
        does not enable it. Default False matches the production timestamp mode.
        """

        def __init__(self, *args, probe_csv="probe.csv",
                     send_capture_usec=False, **kwargs):
            super().__init__(*args, **kwargs)
            self._send_usec = send_capture_usec
            self._f = open(probe_csv, "x", newline="")
            self._lock = threading.Lock()
            self._w = csv.DictWriter(self._f, fieldnames=[
                "frame", "t_cap_s", "queue_ms", "detect_ms", "post_ms", "total_ms",
                "event", "status", "detected", "clock_id", "capture_unix_us",
                "capture_request_ms", "capture_return_age_ms", "t_prepare_s",
                "attitude_age_ms", "attitude_timestamp_us",
                "north_m", "east_m", "down_m", "roll_rad", "pitch_rad", "yaw_rad", "time_usec",
            ])
            self._w.writeheader()
            self._flush_after = time.perf_counter() + 1.0
            self._cur = None
            self.sent_count = 0       # Prepared send attempts, not RPC completions.
            self.last_sent = None

        # _detect runs in a worker thread, but run() awaits it before grabbing
        # the next frame, so video.capture_ns still belongs to this frame.
        def _detect(self, frame, timestamp):
            v = self.video
            t_cap = v.capture_ns + int(v.exposure_us * 500)   # mid-exposure
            t0 = time.clock_gettime_ns(v.clock_id)
            obs = super()._detect(frame, timestamp)
            t1 = time.clock_gettime_ns(v.clock_id)
            self._cur = dict(frame=v.frame_number, t_cap=t_cap, t0=t0, t1=t1,
                             detected=obs is not None, capture_unix_us=v.capture_unix_us,
                             request_ms=v.request_ms, return_age_ms=v.return_age_ms)
            self._record("detect", "detected" if obs is not None else "not_detected")
            return obs

        def _vpe(self, position, R):
            msg = super()._vpe(position, R)
            # Keep marker-derived attitude for the yaw-delay measurement.
            msg.angle_body = AngleBody(*map(float, _euler321(R)))
            if self._cur is not None and self._send_usec:
                msg = VisionPositionEstimate(
                    self._cur["capture_unix_us"], msg.position_body, msg.angle_body,
                    msg.pose_covariance, msg.reset_counter)
            if self._cur is not None:
                start = time.clock_gettime_ns(self.video.clock_id)
                age = self._telemetry_cache.age("attitude_euler") * 1000
                stamp = getattr(self.get_attitude(), "timestamp_us", None)
                self._record("send", "requested", msg, start, age, stamp)
            self.sent_count += 1
            self.last_sent = msg
            return msg

        def _record(self, event, status, msg=None, start=None, age=None, stamp=None):
            c = self._cur
            row = dict(frame=c["frame"], t_cap_s=f'{c["t_cap"] / 1e9:.9f}',
                       queue_ms=(c["t0"] - c["t_cap"]) / 1e6,
                       detect_ms=(c["t1"] - c["t0"]) / 1e6,
                       total_ms=((start if start is not None else c["t1"]) - c["t_cap"]) / 1e6,
                       event=event, status=status, detected=c["detected"],
                       clock_id=self.video.clock_id, capture_unix_us=c["capture_unix_us"],
                       capture_request_ms=c["request_ms"], capture_return_age_ms=c["return_age_ms"])
            if start is not None:
                row.update(post_ms=(start - c["t1"]) / 1e6, t_prepare_s=start / 1e9,
                           attitude_age_ms=age, attitude_timestamp_us=stamp,
                           north_m=msg.position_body.x_m, east_m=msg.position_body.y_m,
                           down_m=msg.position_body.z_m, roll_rad=msg.angle_body.roll_rad,
                           pitch_rad=msg.angle_body.pitch_rad, yaw_rad=msg.angle_body.yaw_rad,
                           time_usec=msg.time_usec)
            with self._lock:
                if self._f.closed:
                    return
                self._w.writerow(row)
                if time.perf_counter() >= self._flush_after:
                    self._f.flush()
                    self._flush_after = time.perf_counter() + 1.0

        def close_probe(self):
            with self._lock:
                self._f.close()


async def record_test(args):
    """Record the camera/VO bench test; never arm or engage offboard."""
    if not HAVE_FLIGHT:
        raise SystemExit(f"Recording requires Picamera2 and the flight SDK on the companion: {FLIGHT_IMPORT_ERROR}")
    import cv2.aruco as aruco
    from mavsdk_grpc import System
    from mavsdk_grpc.offboard import PositionNedYaw, VelocityNedYaw, AccelerationNed
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from aruco_pose import scale_intrinsic_matrix
    import simulated.fc_simulated as fc

    def build_detector():
        params = aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 15
        params.adaptiveThreshWinSizeMax = 45
        params.adaptiveThreshWinSizeStep = 15
        params.minMarkerPerimeterRate = 0.1
        return aruco.ArucoDetector(
            aruco.getPredefinedDictionary(aruco.DICT_4X4_1000), params)

    async def watch_armed():
        async for armed in drone.telemetry.armed():
            if armed:
                raise RuntimeError("Aircraft is armed; stopping bench test")
        raise RuntimeError("Armed-state telemetry ended")

    async def setpoint_traffic():
        pv = await fc.get_drone_ned_position_velocity(drone)
        p, attitude = pv.position, vision.get_attitude()
        if attitude is None:
            raise RuntimeError("No fresh cached attitude")
        while True:
            if armed_task.done():
                armed_task.result()
            await drone.offboard.set_position_velocity_acceleration_ned(
                PositionNedYaw(p.north_m, p.east_m, p.down_m, attitude.yaw_deg),
                VelocityNedYaw(0., 0., 0., attitude.yaw_deg), AccelerationNed(0., 0., 0.),
            )
            await asyncio.sleep(1.0 / args.setpoint_hz)

    with (Path(__file__).resolve().parents[1] / "config.json").open() as file:
        config = json.load(file)
    drone = System()
    print("Connecting...")
    await drone.connect(system_address=config["system_address"])
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected")
            break
    async for armed in drone.telemetry.armed():
        if armed:
            raise RuntimeError("Bench test requires a disarmed aircraft")
        break
    print("-- PX4 parameters:")
    for name in ("EKF2_EV_CTRL", "EKF2_GPS_CTRL", "EKF2_HGT_REF"):
        try:
            print(f"   {name} = {await drone.param.get_param_int(name)}")
        except Exception as error:
            print(f"   {name}: could not read ({error})")
    print("   (EKF2_EV_CTRL: 0 is valid for timing-only recording; keep yaw-fusion bit 8 clear)")

    origin = await asyncio.wait_for(drone.telemetry.get_gps_global_origin(), 5.0)
    if not np.isfinite(origin.altitude_m):
        raise ValueError("EV origin altitude must be finite")
    w, h = config["camera"]["resolution"]
    csv_path = Path(str(args.output_prefix) + ".csv")
    video_path = Path(str(args.output_prefix) + ".h264")
    if csv_path.exists() or (not args.no_recording and video_path.exists()):
        raise FileExistsError("Choose a new --output-prefix; diagnostic files already exist")
    camera = Picamera2()
    vision = None
    vision_task = traffic_task = None
    camera_started = False
    armed_task = asyncio.create_task(watch_armed())
    try:
        camera.configure(camera.create_video_configuration(
            {"size": (w, h), "format": "YUV420"}, encode="main"))
        if args.no_recording:
            camera.start()
        else:
            camera.start_recording(H264Encoder(10_000_000), str(video_path))
        camera_started = True
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
            origin_altitude_m=origin.altitude_m,
            probe_csv=csv_path,
        )
        vision_task = asyncio.create_task(vision.run(drone))
        print(f"-- CSV: {csv_path}; recording: {not args.no_recording}; no clock/timestamp-mode changes")
        print("-- Hold the drone still with the marker in view...")
        await asyncio.wait_for(vision.wait_ready(), 15.0)
        print("-- Marker detected. Anchoring the marker frame to PX4's local NED...")
        vision.enable_frame_initialization()
        await asyncio.wait_for(vision.wait_frame_ready(), 15.0)
        vision.enable_fusion()
        if args.setpoint_hz:
            traffic_task = asyncio.create_task(setpoint_traffic())
        print("-- Sending vision estimates. Now move the drone by hand. Ctrl+C to stop.\n")

        t0 = time.perf_counter()
        last_count, last_t = 0, t0
        while time.perf_counter() - t0 < args.duration:
            await asyncio.sleep(1.0)
            for task in (armed_task, vision_task, traffic_task):
                if task is not None and task.done():
                    task.result()
                    raise RuntimeError("Bench-test worker stopped unexpectedly")
            now = time.perf_counter()
            rate = (vision.sent_count - last_count) / (now - last_t)
            last_count, last_t = vision.sent_count, now
            sent = vision.last_sent
            obs = vision.get_observation(max_age=0.5)
            if sent is None or obs is None:
                print(f"[{now - t0:6.1f}s] no marker / nothing sent  (rate {rate:.1f} Hz)")
                continue
            pv = await fc.get_drone_ned_position_velocity(drone)
            att = vision.get_attitude()
            if att is None:
                print("No fresh cached attitude")
                continue
            vp = np.array([sent.position_body.x_m, sent.position_body.y_m, sent.position_body.z_m])
            vp[2] += origin.altitude_m  # Convert wire height back to local NED for comparison.
            px = np.array([pv.position.north_m, pv.position.east_m, pv.position.down_m])
            dyaw = (np.degrees(sent.angle_body.yaw_rad) - att.yaw_deg + 180) % 360 - 180
            print(f"[{now - t0:6.1f}s] rate {rate:5.1f} Hz | "
                  f"dist to marker {np.linalg.norm(obs.tvec):5.2f} m | "
                  f"VISION NED {np.round(vp, 2)} | PX4 NED {np.round(px, 2)} | "
                  f"|diff| {np.linalg.norm(vp - px):.2f} m | yaw diff {dyaw:+.1f} deg")
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if vision is not None:
            vision.disable_fusion()
        tasks = [task for task in (vision_task, traffic_task, armed_task) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            if vision is not None:
                vision.close_probe()
        finally:
            try:
                if camera_started:
                    if args.no_recording:
                        camera.stop()
                    else:
                        camera.stop_recording()
            finally:
                camera.close()
        print("\nStopped.")


# ----------------------------------------------------------------- analysis
def report_csv(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    detections = [r for r in rows if r.get("event") == "detect"]
    if detections:
        print(f"{len(detections)} frames, {sum(r['detected'] == 'True' for r in detections)} detections")
    rows = [r for r in rows if "event" not in r or (r["event"] == "send" and r["status"] in ("requested", "rpc_ok"))]
    if not rows:
        print("No prepared vision messages in this CSV; check detection and telemetry.")
        return
    capture = np.asarray([float(r["t_cap_s"]) for r in rows])
    print(f"{len(rows)} messages prepared, {capture[-1] - capture[0]:.1f} s")
    for name in ("capture_request_ms", "queue_ms", "detect_ms", "post_ms", "total_ms",
                 "rpc_ms", "rpc_done_age_ms", "attitude_age_ms"):
        x = np.asarray([float(r[name]) for r in rows if r.get(name) not in (None, "")])
        if not len(x):
            continue
        print(f"  {name:10s} mean {x.mean():6.1f}  p50 {np.percentile(x, 50):6.1f}  "
              f"p95 {np.percentile(x, 95):6.1f}  max {x.max():6.1f}")
    gaps = np.diff(capture)
    if len(gaps):
        print(f"  period     mean {1000 * gaps.mean():6.1f} ms, max gap {gaps.max():.2f} s")
    print("total_ms ends at message preparation, before the vision RPC; completion/delivery is not measured.")
    print("Do not derive link latency by subtracting an unrelated CSV.")


def _axis_angle(q, axis):
    """Signed Euler angle from normalized w,x,y,z quaternions; unwrap yaw/roll."""
    w, x, y, z = q.T
    if axis == "yaw":
        angle = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    elif axis == "roll":
        angle = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    else:
        return np.degrees(np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)))
    return np.degrees(np.unwrap(angle))


def _fit(t_v, a_v, t_a, a_a, lags):
    """Fit delay with one constant angle offset removed from every residual."""
    cost = np.array([np.var(a_v - np.interp(t_v - tau, t_a, a_a)) for tau in lags])
    k = int(np.argmin(cost))
    tau = lags[k]
    if 0 < k < len(lags) - 1:
        c0, c1, c2 = cost[k - 1:k + 2]
        den = c0 - 2 * c1 + c2
        if den > 0:
            tau += 0.5 * (c0 - c2) / den * (lags[1] - lags[0])
    return tau, cost, k


def _window_fit(t_v, a_v, t_a, a_a, lags, minimum_samples=15):
    if (len(t_v) < minimum_samples or np.ptp(a_v) < 15
            or np.ptp(np.interp(t_v, t_a, a_a)) < 15):
        return None
    tau, cost, k = _fit(t_v, a_v, t_a, a_a, lags)
    issues = []
    if k in (0, len(lags) - 1):
        issues.append("minimum is at the search boundary")
    if np.max(np.diff(t_v)) > 0.3:
        issues.append("vision gap exceeds 300 ms")
    near = lags[cost <= cost[k] + max(0.25**2, 0.1 * cost[k])]
    width = near[-1] - near[0]
    if width > 0.15:
        issues.append("cost curve is too flat/ambiguous")
    if np.sqrt(cost[k]) > max(2.0, 0.15 * np.ptp(a_v)):
        issues.append("orientation mismatch is too large")
    if tau < -0.01:
        issues.append("negative delay: check timestamps/reference")
    return dict(tau=tau, cost=cost, k=k, width=width, issues=issues)


def analyze_ulog(path, start=None, end=None, axis="yaw", cost_csv=None):
    from pyulog import ULog
    ul = ULog(path, message_name_filter_list=["vehicle_visual_odometry",
                                              "vehicle_attitude"])

    def load(name):
        try:
            d = ul.get_dataset(name).data
        except (KeyError, IndexError, ValueError):
            raise SystemExit(f"'{name}' not in the log. Log it at full rate "
                             f"(SDLOG_PROFILE) and set SDLOG_MODE=1.")
        q = np.column_stack([d[f"q[{i}]"] for i in range(4)]).astype(float)
        ts = d["timestamp"] * 1e-6
        tsample = d.get("timestamp_sample")
        t = ts if tsample is None else np.where(tsample > 0, tsample * 1e-6, ts)
        ok = np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > 0.5)
        if not np.any(ok):
            raise SystemExit(f"No valid quaternion samples in {name}")
        t, ts, q = t[ok], ts[ok], q[ok] / np.linalg.norm(q[ok], axis=1, keepdims=True)
        order = np.argsort(t)
        t, ts, q = t[order], ts[order], q[order]
        unique = np.r_[True, np.diff(t) > 0]
        return t[unique], ts[unique], q[unique]

    t_v, ts_v, q_v = load("vehicle_visual_odometry")
    t_a, _, q_a = load("vehicle_attitude")
    lags = np.arange(-0.05, 0.80, 0.005)
    origin = ul.start_timestamp * 1e-6
    use = ((t_v - lags[-1] >= t_a[0]) & (t_v - lags[0] <= t_a[-1])
           & (t_v >= origin + start if start is not None else True)
           & (t_v <= origin + end if end is not None else True))
    tv, av = t_v[use], _axis_angle(q_v, axis)[use]
    a_a = _axis_angle(q_a, axis)
    if len(tv) < 2:
        raise SystemExit("Too few samples in the selected interval with reference coverage")
    age_ms = (ts_v[use] - tv) * 1000
    print(f"{axis}: {len(tv)} vision samples, log time {tv[0]-origin:.2f}-{tv[-1]-origin:.2f} s")
    print(f"vision publication - sample time p50/p95: {np.percentile(age_ms, [50,95]).round(1)} ms")
    print("Near-zero age does not prove capture-time synchronization. Fit uses sample timestamps;")
    print("with capture-time stamping this estimates residual error, not total transport latency.")
    fit = _window_fit(tv, av, t_a, a_a, lags)
    if fit is None:
        print("NO RECOMMENDATION: need at least 15 samples and 15 degrees of motion in both signals.")
        return
    tau, cost, k = fit["tau"], fit["cost"], fit["k"]
    print(f"\nBest fitted lag: {tau*1000:.0f} ms; near-minimum span: {fit['width']*1000:.0f} ms (not a confidence interval)")
    print(f"Residual RMS {np.sqrt(cost[k]):.2f} deg; zero-lag RMS {np.sqrt(cost[np.argmin(abs(lags))]):.2f} deg")
    configured = ul.initial_parameters.get("EKF2_EV_DELAY")
    if configured is not None and lags[0] <= configured / 1000 <= lags[-1]:
        print(f"At initial EKF2_EV_DELAY={configured:g} ms: RMS {np.sqrt(cost[np.argmin(abs(lags-configured/1000))]):.2f} deg")
    issues = fit["issues"].copy()
    halves = []
    mid = (tv[0] + tv[-1]) / 2
    for label, mask in (("first half", tv <= mid), ("second half", tv > mid)):
        part = _window_fit(tv[mask], av[mask], t_a, a_a, lags, minimum_samples=6)
        halves.append(part)
        if part is None:
            print(f"{label}: insufficient motion/samples")
            issues.append(f"{label} is not observable; include repeated rotations in both halves")
        else:
            print(f"{label}: {part['tau']*1000:.0f} ms")
            issues.extend(f"{label}: {issue}" for issue in part["issues"])
    if all(part is not None for part in halves) and abs(halves[0]["tau"] - halves[1]["tau"]) > 0.05:
        issues.append("half-window estimates disagree by more than 50 ms")
    ctrl = ul.initial_parameters.get("EKF2_EV_CTRL", 0)
    for stamp, name, value in ul.changed_parameters:
        if name == "EKF2_EV_CTRL" and stamp * 1e-6 < tv[0]:
            ctrl = value
    if axis == "yaw" and (ctrl & 8 or any(name == "EKF2_EV_CTRL" and value & 8
            and tv[0] <= stamp * 1e-6 <= tv[-1] for stamp, name, value in ul.changed_parameters)):
        issues.append("EV yaw fusion was enabled; PX4 yaw is not an independent reference")
    if np.any(age_ms < -50):
        issues.append("vision sample timestamps are more than 50 ms in the future")
    if cost_csv:
        with open(cost_csv, "x", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["delay_ms", "residual_rms_deg"])
            writer.writerows(zip(lags * 1000, np.sqrt(cost)))
        print(f"Cost curve: {cost_csv}")
    if issues:
        print("\nNO RECOMMENDATION:")
        for issue in dict.fromkeys(issues):
            print(f"  - {issue}")
    else:
        print(f"\nCandidate EKF2_EV_DELAY: {max(0, tau*1000):.0f} ms (timing only); confirm with a second recording and EKF innovations.")
    print("The fitted value is not an increment to add to the existing setting. No parameters were changed.")


def main():
    ap = argparse.ArgumentParser(description="Record and analyze the disarmed ArUco delay test")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="Record on the companion; props removed, aircraft disarmed")
    r.add_argument("--duration", type=float, default=60.)
    r.add_argument("--output-prefix", default=f"probe_{datetime.now():%Y%m%d_%H%M%S}")
    r.add_argument("--no-recording", action="store_true")
    r.add_argument("--setpoint-hz", type=int, choices=(0, 20, 50), default=0,
                   help="Optional stationary setpoint traffic; never engages offboard")
    sub.add_parser("csv").add_argument("path")
    u = sub.add_parser("ulog")
    u.add_argument("path")
    u.add_argument("--csv")
    u.add_argument("--start", type=float, help="Start seconds relative to ULog start")
    u.add_argument("--end", type=float, help="End seconds relative to ULog start")
    u.add_argument("--axis", choices=("yaw", "roll", "pitch"), default="yaw")
    u.add_argument("--cost-csv", help="Write the delay/RMS curve to a new CSV")
    a = ap.parse_args()
    if a.cmd == "record":
        if not np.isfinite(a.duration) or a.duration <= 0:
            ap.error("--duration must be finite and positive")
        asyncio.run(record_test(a))
    elif a.cmd == "csv":
        report_csv(a.path)
    else:
        if ((a.start is not None and not np.isfinite(a.start))
                or (a.end is not None and not np.isfinite(a.end))):
            ap.error("Window bounds must be finite")
        if a.start is not None and a.end is not None and a.start >= a.end:
            ap.error("--start must precede --end")
        if a.csv:
            report_csv(a.csv)
        print()
        analyze_ulog(a.path, a.start, a.end, a.axis, a.cost_csv)


if __name__ == "__main__":
    main()
