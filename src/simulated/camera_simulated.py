import cv2.aruco as aruco
import cv2
import gi
import numpy as np
from PIL import Image
from pathlib import Path

gi.require_version('Gst', '1.0')
from gi.repository import Gst


class Video():
    """BlueRov video capture class constructor

    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
    """

    def __init__(self, port=5600):
        """Summary

        Args:
            port (int, optional): UDP port
        """

        Gst.init(None)

        self.port = port
        self._frame = None

        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:5600)
        self.video_source = 'udpsrc port={}'.format(self.port)
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        # Create a sink to get data
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]

        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array

        Args:
            sample (TYPE): Description

        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        """ Get Frame

        Returns:
            iterable: bool and image frame, cap.read() output
        """
        return self._frame

    def frame_available(self):
        """Check if frame is available

        Returns:
            bool: true if frame is available
        """
        return type(self._frame) != type(None)

    def run(self):
        """ Get frame to update _frame
        """

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame

        return Gst.FlowReturn.OK
    

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

    # The points origin is centered in the top left corner, and must be defined accordingly
    marker_corner_points = np.array([
        [0, 0, 0],                # Top left corner
        [marker_size, 0, 0],            # Top right corner
        [marker_size, marker_size, 0],       # Bottom right corner
        [0, marker_size, 0]            # Bottom left corner
    ], dtype=np.float32)

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
        
        #Rt = cv2.Rodrigues(R)
        #print(f"R: {R}")
        #print(f"Rt: {Rt}")

        """roll = degrees(atan2(-Rt[2][1], Rt[2][2]))
        pitch = degrees(asin(Rt[2][0]))
        yaw = degrees(atan2(-Rt[1][0], Rt[0][0]))
        print(f"Roll Calculated: {roll}")
        print(f"Pitch Calculated: {pitch}")
        print(f"Yaw Calculated: {yaw}")

    forward_distance = - tvecs[0][1][0]
    right_distance = tvecs[0][0][0]
    vertical_distance = tvecs[0][2][0]

    print(f"forward_distance: {forward_distance}, right_distance: {right_distance}, vertical_distance: {vertical_distance}")"""
    return rvecs, tvecs, trash

def checkArucoPresence(video: Video, detector, mtx, dist):
    while True:
        if not video.frame_available():
            continue

        original_frame = video.frame()
        frame = original_frame.copy()
        frame.setflags(True)

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
        if not markerIds is None:
            return my_estimatePoseSingleMarkers(markerCorners, 0.5, mtx, dist)
        else:
            return None

    
# Function to test a real aruco pad
def aruco_pad_test():
    ARUCO_DETECTOR_THRESHOLD_WINDOW_MIN = 15
    ARUCO_DETECTOR_THRESHOLD_WINDOW_MAX = 45
    ARUCO_DETECTOR_THRESHOLD_WINDOW_STEP = 15
    ARUCO_DETECTOR_MIN_MARKER_PERIMETER_RATE = 0.1

    # intrinsic matrix of the Raspberry Pi camera, set with a resolution of 1640 × 1232
    mtx = np.array([[1607, 0, 820.0],
                    [0, 1607, 616.0],
                    [0, 0, 1]])
    dist = np.array([0.0, 0.0, 0.0, 0.0])

    while True:
        path = Path('test.png') 
        img = Image.open(path)
        original_frame = np.asarray(img)
        frame = original_frame.copy()
        frame.setflags(True)

        arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        arucoParams = aruco.DetectorParameters()

        # Detector parameters tuning
        arucoParams.adaptiveThreshWinSizeMax = ARUCO_DETECTOR_THRESHOLD_WINDOW_MAX
        arucoParams.adaptiveThreshWinSizeMin = ARUCO_DETECTOR_THRESHOLD_WINDOW_MIN
        arucoParams.adaptiveThreshWinSizeStep = ARUCO_DETECTOR_THRESHOLD_WINDOW_STEP
        # The marker side is 50 pixels at 8 meters of distance, so 170 pixels (safety margin) divided by 1640
        arucoParams.minMarkerPerimeterRate = ARUCO_DETECTOR_MIN_MARKER_PERIMETER_RATE

        arucoDetector = aruco.ArucoDetector(arucoDict, arucoParams)

        markerCorners, markerIds, rejectedCandidates = arucoDetector.detectMarkers(frame)
        if not markerIds is None:
            print("Marker detected")
            rvecs, tvecs, trash = my_estimatePoseSingleMarkers(markerCorners, 0.25, mtx, dist)

            tvecs2 = np.array([tvecs[0][0][0], tvecs[0][1][0], tvecs[0][2][0]])
            rvecs2 = np.array([rvecs[0][0][0], rvecs[0][1][0], rvecs[0][2][0]])
            #print([cv2.Rodrigues(tvecs2), cv2.Rodrigues(rvecs2)])

            for idx in range(len(markerIds)):
                cv2.drawFrameAxes(frame,mtx,dist,rvecs[idx],tvecs[idx],5)
                #print('marker id:%d, pos_x = %f,pos_y = %f, pos_z = %f' % (markerIds[idx],tvecs[idx][0],tvecs[idx][1],tvecs[idx][2]))

            aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            #print(f"Rvecs: {degrees(rvecs[0][0][0])}, {degrees(rvecs[0][1][0])}, {degrees(rvecs[0][2][0])}")

        cv2.imshow('frame', frame)

        # waiting to let the main script regain control
        #await asyncio.sleep(0.2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def acquire_images():
    # Create the video object
    # Add port= if is necessary to use a different one
    video = Video()

    # intrinsic matrix of the simulated gimbal camera, which streams a video 640 x 360
    mtx = np.array([[1000.0, 0, 320.0],
                [0, 1000.0, 180.0],
                [0, 0, 1]])
    dist = np.array([0.0, 0.0, 0.0, 0.0])

    while True:
        if not video.frame_available():
            continue

        original_frame = video.frame()
        frame = original_frame.copy()
        frame.setflags(True)

        arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        arucoParams = aruco.DetectorParameters()
        arucoDetector = aruco.ArucoDetector(arucoDict, arucoParams)

        markerCorners, markerIds, rejectedCandidates = arucoDetector.detectMarkers(frame)
        if not markerIds is None:
            rvecs, tvecs, trash = my_estimatePoseSingleMarkers(markerCorners, 5, mtx, dist)

            tvecs2 = np.array([tvecs[0][0][0], tvecs[0][1][0], tvecs[0][2][0]])
            rvecs2 = np.array([rvecs[0][0][0], rvecs[0][1][0], rvecs[0][2][0]])
            print([cv2.Rodrigues(tvecs2), cv2.Rodrigues(rvecs2)])

            for idx in range(len(markerIds)):
                cv2.drawFrameAxes(frame,mtx,dist,rvecs[idx],tvecs[idx],5)
                #print('marker id:%d, pos_x = %f,pos_y = %f, pos_z = %f' % (markerIds[idx],tvecs[idx][0],tvecs[idx][1],tvecs[idx][2]))

            aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            #print(f"Rvecs: {degrees(rvecs[0][0][0])}, {degrees(rvecs[0][1][0])}, {degrees(rvecs[0][2][0])}")

        cv2.imshow('frame', frame)

        # waiting to let the main script regain control
        #await asyncio.sleep(0.2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    aruco_pad_test()