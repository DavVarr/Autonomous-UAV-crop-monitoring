# ev_delay_test.py
"""
Estimate EKF2_EV_DELAY for the ArUco visual odometry.

Recording (on the Pi, inside your real code):
    from ev_delay_test import TimedVideo, DelayProbeVO
Analysis (Pi or laptop; the ulog part needs `pip install pyulog`):
    python3 ev_delay_test.py csv probe.csv
    python3 ev_delay_test.py ulog flight.ulg --csv probe.csv
"""
import argparse
import csv
import time

import numpy as np

# ---------------------------------------------------------------- recording
try:
    import camera as camera_module
    from simulated.VisualOdometry import VisualOdometry
    from mavsdk_grpc.mocap import VisionPositionEstimate
    HAVE_FLIGHT = True
except ImportError:          # analysis on a laptop: recording classes unavailable
    HAVE_FLIGHT = False

if HAVE_FLIGHT:

    class TimedVideo(camera_module.Video):
        """Video that also remembers the sensor timestamp of the last frame."""

        def __init__(self, camera):
            super().__init__(camera)
            self.clock_id = None       # clock matching SensorTimestamp
            self.capture_ns = None     # exposure time of the last frame
            self.exposure_us = 0

        def _pick_clock(self, sensor_ns):
            for cid in (time.CLOCK_BOOTTIME, time.CLOCK_MONOTONIC):
                age = time.clock_gettime_ns(cid) - sensor_ns
                if 0 <= age < 2_000_000_000:
                    return cid
            raise RuntimeError("SensorTimestamp matches neither CLOCK_BOOTTIME "
                               "nor CLOCK_MONOTONIC; cannot measure capture latency")

        def frame(self):
            request = self.camera.capture_request()
            meta = request.get_metadata()
            frame = camera_module._main_frame(self.camera, request)
            request.release()
            ts = meta["SensorTimestamp"]
            if self.clock_id is None:
                self.clock_id = self._pick_clock(ts)
            self.capture_ns = ts
            self.exposure_us = meta.get("ExposureTime", 0)
            return frame

    class DelayProbeVO(VisualOdometry):
        """VisualOdometry that logs per-message latency to a CSV.

        send_capture_usec=True puts the capture time in the MAVLink time_usec
        field (see "time sync" test). Default False = same as your flight code.
        """

        def __init__(self, *args, probe_csv="probe.csv",
                     send_capture_usec=False, **kwargs):
            super().__init__(*args, **kwargs)
            self._send_usec = send_capture_usec
            self._f = open(probe_csv, "w", newline="")
            self._w = csv.writer(self._f)
            self._w.writerow(["frame", "t_cap_s", "queue_ms", "detect_ms",
                              "post_ms", "total_ms"])
            self._cur = None
            self._n = 0
            self.sent_count = 0
            self.last_sent = None

        # _detect runs in a worker thread, but run() awaits it before grabbing
        # the next frame, so video.capture_ns still belongs to this frame.
        def _detect(self, frame, timestamp):
            v = self.video
            t_cap = v.capture_ns + int(v.exposure_us * 500)   # mid-exposure
            t0 = time.clock_gettime_ns(v.clock_id)
            obs = super()._detect(frame, timestamp)
            t1 = time.clock_gettime_ns(v.clock_id)
            self._cur = (t_cap, t0, t1) if obs is not None else None
            return obs

        def _vpe(self, position, R):
            msg = super()._vpe(position, R)
            if self._cur is not None:
                t_cap, t0, t1 = self._cur
                t_send = time.clock_gettime_ns(self.video.clock_id)
                self._w.writerow([self._n, f"{t_cap / 1e9:.6f}",
                                  f"{(t0 - t_cap) / 1e6:.2f}",    # capture -> detect start
                                  f"{(t1 - t0) / 1e6:.2f}",       # detection (+pose)
                                  f"{(t_send - t1) / 1e6:.2f}",   # attitude query etc. -> send
                                  f"{(t_send - t_cap) / 1e6:.2f}"])
                self._n += 1
                if self._n % 10 == 0:
                    self._f.flush()
                if self._send_usec:
                    msg = VisionPositionEstimate(
                        t_cap // 1000, msg.position_body, msg.angle_body,
                        msg.pose_covariance, msg.reset_counter)
            self.sent_count += 1
            self.last_sent = msg
            return msg

        def close_probe(self):
            self._f.close()


# ----------------------------------------------------------------- analysis
def report_csv(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    print(f"{len(d)} messages, {(d['t_cap_s'][-1] - d['t_cap_s'][0]):.1f} s")
    for name in ("queue_ms", "detect_ms", "post_ms", "total_ms"):
        x = d[name]
        print(f"  {name:10s} mean {x.mean():6.1f}  p50 {np.percentile(x, 50):6.1f}  "
              f"p95 {np.percentile(x, 95):6.1f}  max {x.max():6.1f}")
    gaps = np.diff(d["t_cap_s"])
    print(f"  period     mean {1000 * gaps.mean():6.1f} ms, max gap {gaps.max():.2f} s")
    print(f"=> companion-side latency (exposure -> send): {d['total_ms'].mean():.0f} ms "
          f"(jitter p95-p50: {np.percentile(d['total_ms'], 95) - np.percentile(d['total_ms'], 50):.0f} ms)")
    return d["total_ms"].mean()


def _angle_from_ref(t, q, t0, t1):
    """Rotation angle (deg) of each quaternion relative to the mean over [t0, t1].
    Body-frame relative rotation, so it is independent of the world frame."""
    m = (t >= t0) & (t <= t1)
    if m.sum() < 2:
        raise SystemExit("Not enough samples in the reference window")
    ref = q[m] * np.sign(q[m] @ q[m][0])[:, None]
    qr = ref.mean(axis=0)
    qr /= np.linalg.norm(qr)
    return 2 * np.degrees(np.arccos(np.clip(np.abs(q @ qr), 0, 1)))


def _fit(t_v, a_v, t_a, a_a, lags):
    """Find tau so that vision angle(t_i) ~ attitude angle(t_i - tau)."""
    cost = np.array([np.var(a_v - np.interp(t_v - tau, t_a, a_a)) for tau in lags])
    k = int(np.argmin(cost))
    tau = lags[k]
    if 0 < k < len(lags) - 1:
        c0, c1, c2 = cost[k - 1:k + 2]
        den = c0 - 2 * c1 + c2
        if den > 0:
            tau += 0.5 * (c0 - c2) / den * (lags[1] - lags[0])
    return tau, cost, k


def analyze_ulog(path, csv_mean_ms=None):
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
        t = ts if tsample is None or not tsample.any() else tsample * 1e-6
        ok = np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > 0.5)
        return t[ok], ts[ok], q[ok] / np.linalg.norm(q[ok], axis=1, keepdims=True)

    t_v, ts_v, q_v = load("vehicle_visual_odometry")
    t_a, _, q_a = load("vehicle_attitude")
    print(f"vision samples: {len(t_v)} ({len(t_v) / (t_v[-1] - t_v[0]):.1f} Hz in the log)")
    print(f"vision timestamp - timestamp_sample: mean {1000 * np.mean(ts_v - t_v):.1f} ms "
          f"(about 0 => PX4 stamps on arrival; large => PX4 uses your time_usec)")

    ref0, ref1 = t_v[0], t_v[0] + 2.0          # drone still during first 2 s
    a_v = _angle_from_ref(t_v, q_v, ref0, ref1)
    a_a = _angle_from_ref(t_a, q_a, ref0, ref1)
    use = t_v > ref1
    tv, av = t_v[use], a_v[use]
    print(f"motion amplitude (std of vision angle): {av.std():.1f} deg")
    if av.std() < 5:
        print("WARNING: too little motion, the estimate will be unreliable")

    lags = np.arange(-0.05, 0.80, 0.005)
    tau, cost, k = _fit(tv, av, t_a, a_a, lags)
    mid = len(tv) // 2
    tau1 = _fit(tv[:mid], av[:mid], t_a, a_a, lags)[0]
    tau2 = _fit(tv[mid:], av[mid:], t_a, a_a, lags)[0]

    print(f"\nbest lag: {1000 * tau:.0f} ms   (first half {1000 * tau1:.0f}, second half {1000 * tau2:.0f})")
    print(f"residual RMS at best lag {np.sqrt(cost[k]):.2f} deg vs at zero lag "
          f"{np.sqrt(cost[np.argmin(np.abs(lags))]):.2f} deg")
    if k in (0, len(lags) - 1):
        print("WARNING: minimum at the edge of the search range")
    if abs(tau1 - tau2) > 0.03:
        print("WARNING: the two halves disagree by more than 30 ms; repeat with smoother, larger motion")
    print(f"=> set EKF2_EV_DELAY ~ {1000 * tau:.0f} ms")
    if csv_mean_ms is not None:
        print(f"   of which companion side {csv_mean_ms:.0f} ms, link + PX4 receive ~ {1000 * tau - csv_mean_ms:.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("csv").add_argument("path")
    u = sub.add_parser("ulog")
    u.add_argument("path")
    u.add_argument("--csv")
    a = ap.parse_args()
    if a.cmd == "csv":
        report_csv(a.path)
    else:
        mean = report_csv(a.csv) if a.csv else None
        print()
        analyze_ulog(a.path, mean)


if __name__ == "__main__":
    main()