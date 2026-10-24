import cv2.aruco as aruco
import cv2
import asyncio
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from time import perf_counter, sleep

from aruco_pose import my_estimatePoseSingleMarkers


class Video():
    def __init__(self, camera: Picamera2):
        self.camera = camera

    def frame(self):
        request = self.camera.capture_request()
        frame = request.make_array("main")
        request.release()
        return frame

    def frame_available(self):
        return True

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
