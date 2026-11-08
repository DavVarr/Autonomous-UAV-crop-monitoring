"""Capture GPS waypoints interactively for config.json.

    python3 src/capture_waypoints.py --altitude 3

Latitude/longitude come from raw GPS. Altitude is the requested cruise height
above the launch reference, not the GPS altitude above mean sea level.
Quitting saves the captured waypoints to config.json, replacing existing ones.
"""
import argparse
import asyncio
import json
from math import isfinite
from pathlib import Path
import sys


async def read_command():
    # Keep MAVSDK's event loop running while waiting at the Linux/Pi terminal.
    print("Enter: capture waypoint | q: finish > ", end="", flush=True)
    loop = asyncio.get_running_loop()
    ready = loop.create_future()
    fd = sys.stdin.fileno()

    def receive():
        loop.remove_reader(fd)
        if not ready.done():
            ready.set_result(sys.stdin.readline())

    loop.add_reader(fd, receive)
    try:
        return await ready
    finally:
        loop.remove_reader(fd)


async def capture_waypoint(drone, altitude):
    async for info in drone.telemetry.gps_info():
        if info.fix_type.value >= 3:  # 3-D fix, DGPS, or RTK.
            break
    else:
        raise RuntimeError("GPS-info stream ended")

    async for gps in drone.telemetry.raw_gps():
        lat, lon = gps.latitude_deg, gps.longitude_deg
        if isfinite(lat) and isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
            return {"latitude": round(lat, 7), "longitude": round(lon, 7), "altitude": altitude}
    raise RuntimeError("Raw-GPS stream ended")


async def run(altitude):
    from mavsdk_grpc import System

    config_path = Path(__file__).resolve().parents[1] / "config.json"
    with config_path.open() as file:
        config = json.load(file)
    drone = System()
    print(f"Connecting to {config['system_address']}...")
    await drone.connect(system_address=config["system_address"])
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    else:
        raise RuntimeError("Connection-state stream ended")
    print(f"Connected. Waypoint cruise altitude: {altitude:g} m above launch reference.")
    waypoints = []
    try:
        while True:
            command = await read_command()
            if command == "" or command.strip().lower() == "q":
                break
            if command.strip():
                continue
            print("Reading GPS...")
            try:
                waypoint = await asyncio.wait_for(capture_waypoint(drone, altitude), timeout=10.0)
            except asyncio.TimeoutError:
                print("No valid GPS sample within 10 seconds; try again.")
                continue
            waypoints.append(waypoint)
            print(f"Waypoint {len(waypoints)}:")
            print(json.dumps(waypoint, indent=2))
    finally:
        config["waypoints"] = waypoints
        with config_path.open("w") as file:
            json.dump(config, file, indent=2)
            file.write("\n")
        print(f"\nSaved {len(waypoints)} waypoints to {config_path}:")
        print(json.dumps({"waypoints": waypoints}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--altitude", type=float, default=3.0,
                        help="Waypoint cruise height in metres above launch reference (default: 3)")
    args = parser.parse_args()
    if not isfinite(args.altitude) or args.altitude <= 0:
        parser.error("--altitude must be finite and positive")
    try:
        asyncio.run(run(args.altitude))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
