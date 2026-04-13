import cv2.aruco as aruco
import cv2
import asyncio
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from time import perf_counter, sleep

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_center_points = np.array([
        [-marker_size / 2, -marker_size / 2, 0],              # Top left corner
        [marker_size / 2, -marker_size / 2, 0],           # Top right corner
        [marker_size / 2, marker_size / 2, 0],       # Bottom right corner
        [-marker_size / 2, marker_size / 2, 0]           # Bottom left corner
    ], dtype=np.float32)

    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_center_points, c, mtx, distortion, True, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)

    return rvecs, tvecs, trash

def checkArucoPresenceFromImage(camera: Picamera2, detector, mtx, dist):
    request = camera.capture_request()
    image = request.make_array("main")
    request.release()

    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)
    if not markerIds is None:
        #return my_estimatePoseSingleMarkers(markerCorners, 0.603, mtx, dist)
        return my_estimatePoseSingleMarkers(markerCorners, 0.25, mtx, dist)
    else:
        return None

def cameraTest():
    ARUCO_DETECTOR_THRESHOLD_WINDOW_MIN = 15
    ARUCO_DETECTOR_THRESHOLD_WINDOW_MAX = 45
    ARUCO_DETECTOR_THRESHOLD_WINDOW_STEP = 15
    ARUCO_DETECTOR_MIN_MARKER_PERIMETER_RATE = 0.1

    mtx = np.array([[1607, 0, 820.0],
                    [0, 1607, 616.0],
                    [0, 0, 1]])
    dist = np.array([0.0, 0.0, 0.0, 0.0])

    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    arucoParams = aruco.DetectorParameters()

    # Detector parameters tuning
    arucoParams.adaptiveThreshWinSizeMax = ARUCO_DETECTOR_THRESHOLD_WINDOW_MAX
    arucoParams.adaptiveThreshWinSizeMin = ARUCO_DETECTOR_THRESHOLD_WINDOW_MIN
    arucoParams.adaptiveThreshWinSizeStep = ARUCO_DETECTOR_THRESHOLD_WINDOW_STEP
    # The marker side is 50 pixels at 8 meters of distance, so 170 pixels (safety margin) divided by 1640
    arucoParams.minMarkerPerimeterRate = ARUCO_DETECTOR_MIN_MARKER_PERIMETER_RATE
    
    detector = aruco.ArucoDetector(arucoDict, arucoParams)

    picam2 = Picamera2()
    main_stream = {"size": (1640, 1232)}
    lores_stream = {"size": (1640, 1232)}
    video_config = picam2.create_video_configuration(main_stream, lores_stream, encode="lores")
    picam2.configure(video_config)
    encoder = H264Encoder(10000000)

    picam2.start_recording(encoder, 'test.h264')
    sleep(2)

    # It's better to capture the still in this thread, not in the one driving the camera.
    image_start = perf_counter()
    request = picam2.capture_request()
    request.save("main", "test.png")
    array = request.make_array("main")
    request.release()
    image_end = perf_counter()
    print(f"capturing image took {image_end - image_start} seconds")

    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(array)
    if not markerIds is None:
        #return my_estimatePoseSingleMarkers(markerCorners, 0.603, mtx, dist)
        print(my_estimatePoseSingleMarkers(markerCorners, 0.25, mtx, dist))
        aruco_end = perf_counter()
        print(f"Finding Aruco took {aruco_end - image_start} seconds")
    else:
        print("Aruco not found")

    sleep(2)
    picam2.stop_recording()

if __name__ == "__main__":
    cameraTest()
