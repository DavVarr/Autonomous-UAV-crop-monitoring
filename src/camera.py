import cv2.aruco as aruco
import cv2
import asyncio
import logging
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import Output
from queue import Queue, Empty, Full
from threading import Event, Thread
from time import perf_counter, sleep

from aruco_pose import my_estimatePoseSingleMarkers


class QueuedOutput(Output):
    """Write encoded bytes to a FileOutput without holding up the encoder.

    The 120-frame queue absorbs about four seconds at 30 fps. Overflow stops
    recording at the queued prefix, rather than blocking capture or resuming
    an H.264 stream with missing reference frames. Inspect error after stop().
    """

    def __init__(self, output, queue_size=120):
        super().__init__()
        if queue_size < 1:
            raise ValueError("Video queue size must be positive")
        self.output = output
        self._queue = Queue(maxsize=queue_size)
        self._stopping = Event()
        self._worker = None
        self.error = None
        self.max_queue_depth = 0

    @property
    def queue_depth(self):
        return self._queue.qsize()

    def start(self):
        if self._worker is not None:
            raise RuntimeError("Create a new QueuedOutput for each recording")
        self.output.start()
        super().start()
        self._worker = Thread(target=self._write, name="video-writer", daemon=True)
        self._worker.start()

    def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
        if not self.recording or self._stopping.is_set():
            return
        try:
            # Copy bytes only: retain no camera request or encoder packet buffer.
            self._queue.put_nowait((bytes(frame), keyframe, timestamp, None, audio))
            self.max_queue_depth = max(self.max_queue_depth, self.queue_depth)
        except Full:
            self._fail(BufferError("Video queue full; recording stopped"))

    def _fail(self, error):
        if self.error is None:
            self.error = error
        self._stopping.set()

    def _write(self):
        try:
            while not (self._stopping.is_set() and self._queue.empty()):
                try:
                    frame = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                self.output.outputframe(*frame)
                if getattr(self.output, "dead", False):
                    raise OSError("Video file output failed")
        except Exception as error:
            self._fail(error)
        finally:
            try:
                self.output.stop()
            except Exception as error:
                self._fail(error)
            while not self._queue.empty():
                self._queue.get_nowait()
            if self.error is not None:
                logging.error("Video recording failed: %s", self.error)

    def stop(self):
        super().stop()
        self._stopping.set()
        if self._worker is not None:
            self._worker.join()
        else:
            self.output.stop()


class Video():
    def __init__(self, camera: Picamera2):
        self.camera = camera

    def frame(self):
        request = self.camera.capture_request()
        frame = _main_frame(self.camera, request)
        request.release()
        return frame

    def frame_available(self):
        return True

def checkArucoPresenceFromImage(camera: Picamera2, detector, mtx, dist):
    request = camera.capture_request()
    image = _main_frame(camera, request)
    request.release()

    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)
    if not markerIds is None:
        #return my_estimatePoseSingleMarkers(markerCorners, 0.603, mtx, dist)
        return my_estimatePoseSingleMarkers(markerCorners, 0.25, mtx, dist)
    else:
        return None


def _main_frame(camera: Picamera2, request):
    frame = request.make_array("main")
    main_config = camera.camera_configuration()["main"]
    if main_config["format"] == "YUV420":
        return frame[:main_config["size"][1], :]
    return frame

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
