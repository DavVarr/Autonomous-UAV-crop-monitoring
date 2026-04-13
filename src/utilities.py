from mavsdk.telemetry import PositionNed, Position
from mavsdk.offboard import PositionNedYaw
from camera import checkArucoPresenceFromImage
from math import fabs, degrees, radians, pi, cos, sin, sqrt, atan2, asin
import numpy as np
import fc
from mavsdk import System
import cv2.aruco as aruco
from picamera2 import Picamera2

def sign(num):
    if num == 0: return 0
    return -1 if num < 0 else 1

def look_for_aruco_image(camera: Picamera2, detector, mtx, dist):
    for i in range(3):
        result = checkArucoPresenceFromImage(camera, detector, mtx, dist)
        if not result is None:
            return result
    return None

def get_aruco_distances_and_yaw_image(camera: Picamera2, detector, mtx, dist) -> list[float]:
    aruco_result = checkArucoPresenceFromImage(camera, detector, mtx, dist)
    if aruco_result is None:
        return None
    else:
        tvecs = aruco_result[1]
        rvecs = aruco_result[0]

        forward_distance = - tvecs[0][1][0]
        right_distance = tvecs[0][0][0]
        vertical_distance = tvecs[0][2][0]

        # Camera mount compensation
        yaw_difference = normalize_heading(degrees(rvecs[0][2][0]) - 90)

        return [forward_distance, right_distance, vertical_distance, yaw_difference]

def get_circle_coordinates(starting_position: PositionNed, heading: float, radius: float) -> list[PositionNed]:
    coordinates = []
    for i in range(0, 10):
        theta = (2 * pi * i) / 10 + radians(heading)
        new_north = starting_position.north_m + radius * cos(theta)
        new_east = starting_position.east_m + radius * sin(theta)
        coordinates.append(PositionNed(new_north, new_east, starting_position.down_m))
    return coordinates

async def get_position_corrections(
        current_forward_distance: float,
        current_right_distance: float,
        current_forward_speed: float,
        current_right_speed: float,
        old_forward_correction: float,
        old_right_correction: float,
        old_forward_wind_speed: float,
        old_right_wind_speed: float,
        alfa:float,
        beta:float
    ):

    forward_wind_speed = old_forward_correction - current_forward_speed
    forward_wind_speed = forward_wind_speed * beta + (1 - beta) * old_forward_wind_speed

    right_wind_speed = old_right_correction - current_right_speed
    right_wind_speed = right_wind_speed * beta + (1 - beta) * old_right_wind_speed

    current_forward_correction = -forward_wind_speed + (min(max(fabs(alfa * current_forward_distance), 0.1), 0.4) * sign(current_forward_distance))
    current_right_correction = -right_wind_speed + (min(max(fabs(alfa * current_right_distance), 0.1), 0.4) * sign(current_right_distance))

    #print(f"Forward speed: {current_forward_speed}")
    #print(f"Forward correction: {current_forward_correction}")
    #print(f"Old Forward correction: {old_forward_correction}")
    #print(f"Right correction: {current_right_correction}")
    #print(f"Forward wind: {forward_wind_speed}")
    #print(f"Right wind: {right_wind_speed}")
    #print("")

    return [current_forward_correction, current_right_correction, forward_wind_speed, right_wind_speed]

async def calculate_target_ned_position(drone: System, aruco_distances: list[float]) -> PositionNedYaw:
    current_ned_position: PositionNed = await fc.get_drone_ned_position(drone)
    current_heading = await fc.get_drone_heading(drone)

    # 270 degrees added to compensate for camera placement
    ned_vector = rotate_vector(aruco_distances[0], aruco_distances[1], 270)

    target_position = PositionNedYaw(
        ned_vector[0] + current_ned_position.north_m,
        ned_vector[1] + current_ned_position.east_m,
        current_ned_position.down_m + aruco_distances[2],
        current_heading)
    
    return target_position

async def calculate_target_global_position(drone: System, aruco_distances: list[float]) -> Position:
    current_global_position = await fc.get_drone_global_position(drone)
    current_heading = await fc.get_drone_heading(drone)

    # 270 degrees added to compensate for camera placement
    rotated_vector = rotate_vector(aruco_distances[0], aruco_distances[1], 270)

    distance = sqrt(pow(rotated_vector[0], 2) + pow(rotated_vector[1], 2))
    angle = degrees(atan2(rotated_vector[1], rotated_vector[0])) % 360
    direction = (current_heading + angle) % 360

    target_position = calculate_new_coordinates(current_global_position, distance, direction)
    
    return target_position

def rotate_vector(x, y, theta):
    theta_rad = np.radians(theta)

    v = np.array([x, y])
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])

    v_rotated = np.dot(R, v)
    return v_rotated

def calculate_new_coordinates(initial_position: Position, distance: float, direction: float):
    lat = radians(initial_position.latitude_deg)
    lon = radians(initial_position.longitude_deg)
    direction = radians(direction)
    
    earth_radius = 6378137.0
    
    new_lat = asin(sin(lat) * cos(distance / earth_radius) +
                          cos(lat) * sin(distance / earth_radius) * cos(direction))
    
    new_lon = lon + atan2(sin(direction) * sin(distance / earth_radius) * cos(lat),
                                 cos(distance / earth_radius) - sin(lat) * sin(new_lat))
    
    new_lat = degrees(new_lat)
    new_lon = degrees(new_lon)
    
    return Position(new_lat, new_lon, initial_position.absolute_altitude_m, initial_position.relative_altitude_m)

def normalize_heading(heading: float):
    if heading < -180: return heading + 360
    elif heading > 180: return heading - 360
    else: return heading

if __name__ == "__main__":
    # Preparing camera
    camera = Picamera2()
    camera_config = camera.create_still_configuration({"size": (1640, 1232)})
    camera.configure(camera_config)
    camera.start()

    # intrinsic matrix of the Raspberry Pi camera, set with a resolution of 1640 × 1232
    mtx = np.array([[1607, 0, 820.0],
                    [0, 1607, 616.0],
                    [0, 0, 1]])
    dist = np.array([0.0, 0.0, 0.0, 0.0])

    #Preparing Aruco detector
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    arucoParams = aruco.DetectorParameters()

    # Detector parameters tuning
    arucoParams.adaptiveThreshWinSizeMax = 45
    arucoParams.adaptiveThreshWinSizeMin = 15
    arucoParams.adaptiveThreshWinSizeStep = 15
    # The marker side is 50 pixels at 8 meters of distance, so 170 pixels (safety margin) divided by 1640
    arucoParams.minMarkerPerimeterRate = 0.1
    
    detector = aruco.ArucoDetector(arucoDict, arucoParams)

    while True:
        res = get_aruco_distances_and_yaw_image(camera, detector, mtx, dist)
        if res is not None:
            rotated_vector = rotate_vector(res[0], res[1], 270)

            #print(f"Forward: {round(rotated_vector[0], 2)}")
            #print(f"Right: {round(rotated_vector[1], 2)}")
            print(f"Vertical: {round(res[2],2)}")
            #print(f"Yaw: {round(res[3], 2)}")
            print()
        else:
            print("Not Found")