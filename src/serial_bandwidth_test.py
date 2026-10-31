"""Disarmed MAVSDK link-load test; no arming, mode changes, or parameter writes.

Use the same serial connection as flight, with props removed and PX4 logging
while disarmed (SDLOG_MODE=1, configured separately). Keep the aircraft still:
the vision traffic is a fixed copy of PX4's initial pose, not camera odometry.

    python3 src/serial_bandwidth_test.py record
    python3 src/serial_bandwidth_test.py analyze flight.ulg --csv serial_load_*.csv

Vision covariance tags let the analyzer match local generation times to ULog
receipt times without synchronized clock origins. PX4 does not publish received
trajectory_setpoint outside offboard, so those tags are only a fallback. These
are diagnostic setpoints, not a flyable trajectory. The unpaced phase is limited
to 300 calls by default.
"""
import argparse
import asyncio
import csv
from datetime import datetime
import importlib
import inspect
import json
import math
from pathlib import Path
import struct
import time


FIELDS = ("kind", "phase", "sequence", "start_s", "end_s", "rpc_ms", "north_m", "tag")


def sdk_modules():
    try:
        sdk = importlib.import_module("mavsdk_grpc")
    except ModuleNotFoundError as error:
        if error.name != "mavsdk_grpc":
            raise
        sdk = importlib.import_module("mavsdk")
    return (
        sdk,
        importlib.import_module(f"{sdk.__name__}.offboard"),
        importlib.import_module(f"{sdk.__name__}.mocap"),
    )


async def first(stream):
    try:
        return await asyncio.wait_for(anext(stream), 15.0)
    finally:
        await stream.aclose()


async def require_disarmed(drone, ready):
    async for armed in drone.telemetry.armed():
        if armed:
            raise RuntimeError("Aircraft is armed; stopping the diagnostic")
        ready.set()
    raise RuntimeError("Armed-state telemetry ended")


async def pace(deadline, rate):
    if rate == 0:
        return deadline
    period = 1.0 / rate
    deadline += period
    now = time.monotonic()
    if deadline <= now:
        # Skip missed slots; never generate a catch-up burst.
        deadline = now + period
    await asyncio.sleep(deadline - now)
    return deadline


async def exercise(drone, offboard, mocap, args, writer):
    pv = await first(drone.telemetry.position_velocity_ned())
    attitude = await first(drone.telemetry.attitude_euler())
    p = pv.position
    if not all(math.isfinite(value) for value in (
        p.north_m, p.east_m, p.down_m,
        attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg,
    )):
        raise RuntimeError("Finite local position and attitude telemetry are required")
    origin = time.monotonic()
    phase = "baseline"
    flush_at = origin
    summary = []

    def log(kind, sequence, start, end, north, tag, source_phase):
        nonlocal flush_at
        writer.writerow({
            "kind": kind, "phase": source_phase, "sequence": sequence,
            "start_s": f"{start - origin:.9f}",
            "end_s": f"{end - origin:.9f}",
            "rpc_ms": f"{(end - start) * 1000:.3f}",
            "north_m": repr(north),
            "tag": repr(tag),
        })
        if end >= flush_at:
            args.csv_file.flush()
            flush_at = end + 0.5

    async def vision_traffic():
        reset = (
            {"reset_counter": 1}
            if "reset_counter" in inspect.signature(mocap.VisionPositionEstimate).parameters
            else {}
        )
        covariance = [0.0] * 21
        for i in (0, 6, 11):
            covariance[i] = 0.05 ** 2
        for i in (15, 18, 20):
            covariance[i] = math.radians(2.0) ** 2
        sequence = 0
        deadline = time.monotonic()
        while True:
            sequence += 1
            tag = struct.unpack("<f", struct.pack("<f", 0.05 ** 2 + sequence * 1e-7))[0]
            covariance[0] = tag
            estimate = mocap.VisionPositionEstimate(
                0,
                mocap.PositionBody(p.north_m, p.east_m, p.down_m),
                mocap.AngleBody(*map(math.radians, (
                    attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg
                ))),
                mocap.Covariance(covariance.copy()), **reset,
            )
            source_phase = phase
            start = time.monotonic()
            await asyncio.wait_for(
                drone.mocap.set_vision_position_estimate(estimate), 10.0
            )
            end = time.monotonic()
            log("vision", sequence, start, end, p.north_m, tag, source_phase)
            deadline = await pace(deadline, args.vision_hz)

    async def setpoint(north):
        await asyncio.wait_for(
            drone.offboard.set_position_velocity_acceleration_ned(
                offboard.PositionNedYaw(north, p.east_m, p.down_m, attitude.yaw_deg),
                offboard.VelocityNedYaw(0.0, 0.0, 0.0, attitude.yaw_deg),
                offboard.AccelerationNed(0.0, 0.0, 0.0),
            ), 10.0,
        )

    vision_task = asyncio.create_task(vision_traffic()) if args.vision_hz else None
    sequence = 0
    try:
        await asyncio.sleep(2.0)
        for rate in args.rates:
            if vision_task is not None and vision_task.done():
                vision_task.result()
            phase = "unpaced" if rate == 0 else f"{rate:g}Hz"
            print(f"Phase {phase}: up to {args.phase_seconds:g} s", flush=True)
            start_phase = time.monotonic()
            deadline = start_phase
            count = 0
            while time.monotonic() - start_phase < args.phase_seconds:
                if rate == 0 and count >= args.unpaced_limit:
                    break
                if vision_task is not None and vision_task.done():
                    vision_task.result()
                # A unique millimetre-scale tag, represented exactly as sent (float32).
                sequence += 1
                north = struct.unpack("<f", struct.pack("<f", p.north_m + sequence * 0.001))[0]
                start = time.monotonic()
                await setpoint(north)
                end = time.monotonic()
                log("setpoint", sequence, start, end, north, north, phase)
                count += 1
                deadline = await pace(deadline, rate)
            elapsed = time.monotonic() - start_phase
            summary.append((phase, count, elapsed))
            print(f"  {count} calls / {elapsed:.2f} s = {count / elapsed:.1f} Hz", flush=True)
            phase = f"drain_after_{phase}"
            await asyncio.sleep(args.drain_seconds)
        # Return the stored setpoint to the actual stationary position; no mode change.
        await setpoint(p.north_m)
        phase = "final_drain"
        await asyncio.sleep(args.drain_seconds)
    finally:
        if vision_task is not None:
            vision_task.cancel()
            try:
                await vision_task
            except asyncio.CancelledError:
                pass
        args.csv_file.flush()
    print("\nCompleted while disarmed. Local RPC completion is not wire delivery.")
    for phase, count, elapsed in summary:
        print(f"  {phase}: {count / elapsed:.1f} application calls/s")
    print("Analyze this CSV together with the disarmed PX4 ULog.")


async def record(args):
    sdk, offboard, mocap = sdk_modules()
    address = args.system_address
    if address is None:
        with (Path(__file__).resolve().parents[1] / "config.json").open() as stream:
            address = json.load(stream)["system_address"]
    print(f"Connecting through {address} using {sdk.__name__}")
    print("Props removed; aircraft stationary and disarmed. No modes or parameters are changed.")
    drone = sdk.System()
    await drone.connect(system_address=address)

    async def connected():
        async for state in drone.core.connection_state():
            if state.is_connected:
                return
    await asyncio.wait_for(connected(), 30.0)
    ready = asyncio.Event()
    monitor = asyncio.create_task(require_disarmed(drone, ready))
    ready_wait = asyncio.create_task(ready.wait())
    load = None
    try:
        done, _ = await asyncio.wait(
            (monitor, ready_wait),
            timeout=15.0, return_when=asyncio.FIRST_COMPLETED,
        )
        if monitor in done:
            monitor.result()
        if not ready.is_set():
            raise TimeoutError("No armed-state telemetry")
        with args.output.open("x", newline="", encoding="utf-8") as stream:
            args.csv_file = stream
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            print(f"CSV: {args.output}")
            load = asyncio.create_task(exercise(drone, offboard, mocap, args, writer))
            try:
                done, _ = await asyncio.wait((monitor, load), return_when=asyncio.FIRST_COMPLETED)
                if monitor in done:
                    monitor.result()
                await load
            finally:
                load.cancel()
                await asyncio.gather(load, return_exceptions=True)
    finally:
        monitor.cancel()
        ready_wait.cancel()
        await asyncio.gather(monitor, ready_wait, return_exceptions=True)


def analyze(args):
    import numpy as np
    from pyulog import ULog

    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    kind = "vision" if any(row["kind"] == "vision" for row in rows) else "setpoint"
    sent = [row for row in rows if row["kind"] == kind]
    if not sent:
        raise ValueError("No messages in the CSV")
    by_tag = {}
    for row in sent:
        key = struct.pack("<f", float(row["tag"]))
        if key in by_tag:
            raise ValueError("Message tags are not unique at float32 precision")
        by_tag[key] = row
    topic_name, field = (
        ("vehicle_visual_odometry", "position_variance[0]") if kind == "vision"
        else ("trajectory_setpoint", "position[0]")
    )
    ulog = ULog(str(args.ulog), message_name_filter_list=[topic_name, "telemetry_status"])
    data = next((topic.data for topic in ulog.data_list if topic.name == topic_name), None)
    seen = set()
    matches = []
    if data is not None:
        for timestamp, tag in zip(data["timestamp"], data[field]):
            key = struct.pack("<f", tag)
            if key in by_tag and key not in seen:
                row = by_tag[key]
                seen.add(key)
                matches.append((row, float(timestamp) / 1e6 - float(row["start_s"])))
    baseline_phase = sent[0]["phase"]
    baseline = [delay for row, delay in matches if row["phase"] == baseline_phase]
    offset = float(np.median(baseline)) if len(baseline) >= 3 else None
    print(f"Arrival probe: {kind} tags in {topic_name}")
    if offset is None:
        print("Insufficient baseline matches: delay cannot be estimated. Check the CSV/ULog")
        print("pair and disarmed topic logging. PX4 normally suppresses trajectory_setpoint")
        print("outside offboard; keep --vision-hz nonzero for disarmed arrival measurements.")
    else:
        print("Added arrival delay relative to first-phase median (not absolute latency):")
    for phase in dict.fromkeys(row["phase"] for row in sent):
        phase_sent = [row for row in sent if row["phase"] == phase]
        phase_matches = [(row, (delay - offset) * 1000) for row, delay in matches if row["phase"] == phase] if offset is not None else []
        delays = np.asarray([delay for _, delay in phase_matches])
        rpc = np.asarray([float(row["rpc_ms"]) for row in phase_sent])
        print(f"\n  {phase}: {len(phase_matches)}/{len(phase_sent)} tags observed; RPC p50/p95 {np.percentile(rpc, [50,95]).round(1)} ms")
        sp_rows = [row for row in rows if row["kind"] == "setpoint" and row["phase"] == phase]
        if sp_rows:
            elapsed = float(sp_rows[-1]["end_s"]) - float(sp_rows[0]["start_s"])
            print(f"    setpoint calls {len(sp_rows) / max(elapsed, 1e-9):.1f}/s")
        if len(delays):
            print(f"    added delay p50/p95/max {np.percentile(delays, [50,95,100]).round(1)} ms")
            times = np.asarray([float(row["start_s"]) for row, _ in phase_matches])
            if len(delays) >= 3 and np.ptp(times) > 0:
                print(f"    delay trend {np.polyfit(times, delays, 1)[0]:.1f} ms per generated second")
    capacity = args.baud / 10
    print(f"\nNominal 8N1 capacity: {capacity:.0f} bytes/s per direction")
    for topic in ulog.data_list:
        if topic.name != "telemetry_status":
            continue
        rx = topic.data["rx_rate_avg"]
        print(f"  telemetry_status[{topic.multi_id}] RX p95/max {np.percentile(rx, [95,100]).round(0)} bytes/s")
    print("\nLow ULog sampling can miss tags and adds sampling uncertainty. Use a faster")
    print(f"{topic_name} logging rate if only a few tags are observed. Also inspect")
    print("vehicle_visual_odometry, estimator aid status, and RX rate versus time in the ULog.")


def rate(value):
    result = 0.0 if value == "unpaced" else float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("Rate must be non-negative or 'unpaced'")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    recording = commands.add_parser("record", help="Send diagnostic traffic while disarmed")
    recording.add_argument("--system-address", help="Defaults to config.json; use the same UART as flight")
    recording.add_argument("--rates", nargs="+", type=rate, default=[10.0, 20.0, 50.0, 0.0])
    recording.add_argument("--phase-seconds", type=float, default=5.0)
    recording.add_argument("--drain-seconds", type=float, default=12.0)
    recording.add_argument("--vision-hz", type=float, default=5.0, help="Stationary pose messages/s; 0 disables")
    recording.add_argument("--unpaced-limit", type=int, default=300)
    recording.add_argument("--output", type=Path, default=Path(f"serial_load_{datetime.now():%Y%m%d_%H%M%S}.csv"))
    analysis = commands.add_parser("analyze", help="Match CSV tags to their ULog arrival times")
    analysis.add_argument("ulog", type=Path)
    analysis.add_argument("--csv", type=Path, required=True)
    analysis.add_argument("--baud", type=int, default=57600)
    args = parser.parse_args()
    if args.command == "record":
        if not all(math.isfinite(v) and v > 0 for v in (args.phase_seconds, args.drain_seconds)):
            parser.error("Phase and drain durations must be finite and positive")
        if not math.isfinite(args.vision_hz) or args.vision_hz < 0 or args.unpaced_limit < 1:
            parser.error("Vision rate must be finite/non-negative and unpaced limit positive")
        if args.rates[0] == 0:
            parser.error("Start with a bounded rate for baseline delay measurement")
        asyncio.run(record(args))
    else:
        if args.baud <= 0:
            parser.error("Baud must be positive")
        analyze(args)


if __name__ == "__main__":
    main()
