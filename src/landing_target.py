import math
import cv2
import cv2.aruco as aruco
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
import os
import pickle
from pyzbar.pyzbar import decode

class LandingTarget():
    def __init__(self,
                 calibration_file='camera_calibration.pkl', 
                 known_qr_size_cm=28.5,
                 camera_resolution=(1280, 720),
                 ) -> None:
        self.qr_content_saved = False
        self.picam2 = Picamera2()
        self.qr_detector = cv2.QRCodeDetector()
        # Set epsilon parameters for detection sensitivity
        self.qr_detector.setEpsX(0.9)  # Horizontal sensitivity (default is 0.2)
        self.qr_detector.setEpsY(0.9)  # Vertical sensitivity (default is 0.1)
        # Configure the camera
        config = self.picam2.create_still_configuration(
            main={"size": camera_resolution, "format": "RGB888"}
        )
        self.picam2.configure(config)
        # Set camera controls (optional: adjust based on conditions)
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        # Start the camera
        self.picam2.start()
        print("Camera initialized")
        self.qr_content = None
        self.save_directory = "qr_content"
        # Ensure the save directory exists
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        # Load camera calibration data
        try:
            with open(calibration_file, 'rb') as f:
                calibration_data = pickle.load(f)
            self.camera_matrix = calibration_data['camera_matrix']
            self.dist_coeffs = calibration_data['dist_coeffs']
            self.img_size = calibration_data['img_size']
            # Get focal length from camera matrix
            self.focal_length_x = self.camera_matrix[0, 0]
            self.focal_length_y = self.camera_matrix[1, 1]
            # Get principal point (image center in pixel coordinates)
            self.principal_point_x = self.camera_matrix[0, 2]
            self.principal_point_y = self.camera_matrix[1, 2]
            print("Loaded camera calibration data:")
            print(f"Focal length: ({self.focal_length_x}, {self.focal_length_y})")
            print(f"Principal point: ({self.principal_point_x}, {self.principal_point_y})")
        except (FileNotFoundError, KeyError):
            print("Warning: Calibration file not found or invalid. Using estimated parameters.")
            raise NotImplementedError("Calibration file not found or invalid. Please calibrate the camera first.")

        # Known size of QR code in cm
        self.known_qr_size_cm = known_qr_size_cm
        
        # Convert to meters for MAVLink
        self.known_qr_size_m = known_qr_size_cm / 100.0

        # Define 3D coordinates of QR code corners (assuming flat and square)
        half_size = self.known_qr_size_m / 2
        self.object_points = np.array([
            [-half_size, -half_size, 0],  # Top-left
            [half_size, -half_size, 0],   # Top-right
            [half_size, half_size, 0],    # Bottom-right
            [-half_size, half_size, 0]    # Bottom-left
        ], dtype=np.float32)
    
    def _get_frame(self):
        """Capture a frame from picamera2."""
        return self.picam2.capture_array()


    def _get_qr_corners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        qr_codes = decode(gray)

        if len(qr_codes) > 0:
            qr_code = qr_codes[0]
            print("qr code fully decoded!")
        else:
            qr_code = None

        if qr_code is not None: return np.array([(p.x, p.y) for p in qr_code.polygon], dtype=np.float32), qr_code.data.decode('utf-8')
        print("proceeding to find finder patterns")

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        found, points = self.qr_detector.detect(frame_bgr)
        if found and points is not None and len(points) > 0:
            points = points.astype(np.int32)
            if not np.isnan(points).any() and not np.isinf(points).any():
                # we've found a qr code and the points are not fucking insane
                print("we've found a qr code and the points are not fucking insane")
                return points[0], None
        
        print("No qr codes pa")
        return None, None

    def order_points(self, pts):
        """Order points in [top-left, top-right, bottom-right, bottom-left] order."""
        # Convert to numpy array if it's not already
        pts = np.array(pts)
        # Sort by y-coordinate (top-to-bottom)
        sorted_by_y = pts[np.argsort(pts[:, 1])]
        # Get top and bottom points
        top_points = sorted_by_y[:2]
        bottom_points = sorted_by_y[2:]
        # Sort top points by x-coordinate (left-to-right)
        top_left, top_right = top_points[np.argsort(top_points[:, 0])]
        # Sort bottom points by x-coordinate (left-to-right)
        bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
        
        # Return ordered points
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    def calculate_landing_target_params(self, corners):
        """Calculate MAVLink landing target parameters from corners using PnP."""
        if len(corners) != 4:
            # If corners are not exactly 4, estimate to make a square
            # Find the center
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])
            # Estimate average side length
            dists = []
            for i in range(len(corners)):
                next_idx = (i + 1) % len(corners)
                dist = np.linalg.norm(corners[i] - corners[next_idx])
                dists.append(dist)
            avg_side = np.mean(dists)
            half_side = avg_side / 2
            # Create a square around the center
            square_corners = np.array([
                [center_x - half_side, center_y - half_side],  # Top-left
                [center_x + half_side, center_y - half_side],  # Top-right
                [center_x + half_side, center_y + half_side],  # Bottom-right
                [center_x - half_side, center_y + half_side]   # Bottom-left
            ], dtype=np.float32)
            corners = square_corners
        else:
            # Ensure corners are in the correct order
            corners = self.order_points(corners)
        # Solve PnP to get rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(
            self.object_points, corners, self.camera_matrix, self.dist_coeffs
        )
        if not success:
            return None
        # Extract x, y, z coordinates from translation vector
        tx, ty, tz = tvec.flatten()
        # Distance is the Z component of the translation vector (in meters)
        distance = tz
        # Calculate the center of the QR code in pixel coordinates
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        # Calculate angular offset from image center
        dx_pixels = center_x - self.principal_point_x
        dy_pixels = center_y - self.principal_point_y
        # Convert to radians
        angle_x = math.atan2(dx_pixels, self.focal_length_x)
        angle_y = math.atan2(dy_pixels, self.focal_length_y)
        # Calculate the width and height of the QR code in pixels
        width = np.linalg.norm(corners[0] - corners[1])
        height = np.linalg.norm(corners[0] - corners[3])
        # Calculate angular size
        size_x = math.atan2(width, self.focal_length_x)
        size_y = math.atan2(height, self.focal_length_y)
        # Return MAVLink parameters
        return {
            'angle_x': angle_x,
            'angle_y': angle_y,
            'distance': distance,
            'size_x': size_x,
            'size_y': size_y,
            'rvec': rvec,
            'tvec': tvec,
            'corners': corners
        }


    def get_landing_target_vals(self):
        frame = self._get_frame()
        qr_corners, qr_data = self._get_qr_corners(frame)
        if qr_corners is None: return None
        if qr_data is not None:
            # save it to a file
            if self.qr_content_saved != True:
                text_file = open("qr_content.txt", "w")
                text_file.write(qr_data)
                text_file.close()
                self.qr_content_saved = True

        landing_target_vals = self.calculate_landing_target_params(qr_corners)
        # returns this shit {
        #     'angle_x': angle_x,
        #     'angle_y': angle_y,
        #     'distance': distance,
        #     'size_x': size_x,
        #     'size_y': size_y,
        #     'rvec': rvec,
        #     'tvec': tvec,
        #     'corners': corners
        # }
        if landing_target_vals is None: return None
        return landing_target_vals['angle_x'], landing_target_vals['angle_y']

    def detect_aruco_marker(self):
        """Detect ArUco marker and calculate landing target parameters."""
        frame = self._get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Define ArUco dictionary and parameters
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        aruco_params = aruco.DetectorParameters()
        
        # Detect markers in the frame
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        if ids is not None and len(corners) > 0:
            print(f"Detected ArUco marker with ID: {ids.flatten()[0]}")
            ordered_corners = self.order_points(corners[0][0])
            landing_target_vals = self.calculate_landing_target_params(ordered_corners)
            
            if landing_target_vals:
                return landing_target_vals['angle_x'], landing_target_vals['angle_y']
        
        print("No ArUco marker detected.")
        return None
