#!/usr/bin/env python3
import numpy as np
import cv2
import math
import pickle
import time
import os
import argparse
# from picamera2 import Picamera2
# from libcamera import controls

class ArucoLandingTargetTracker:
    def __init__(self, 
                 calibration_file='camera_calibration.pkl', 
                 known_marker_size_cm=10.0,
                 camera_resolution=(1280, 720),
                 dictionary_id=cv2.aruco.DICT_4X4_50,
                 marker_id=0):
        
        # Initialize picamera2
        self.picam2 = Picamera2()
        
        # Configure the camera with fast shutter for drone vibration
        config = self.picam2.create_still_configuration(
            main={"size": camera_resolution, "format": "RGB888"}
        )
        self.picam2.configure(config)
        
        # Use optimized exposure controls for drone with fixed focus camera
        self.picam2.set_controls({
            "AfMode": controls.AfModeEnum.Manual,  # Fixed focus
            "AeEnable": False,                     # Disable auto exposure
            "ExposureTime": 2000,                  # Fast exposure (2ms) to minimize motion blur
            "AnalogueGain": 2.0                    # Moderate gain
        })
        
        # Start the camera
        self.picam2.start()
        print("Camera initialized with settings optimized for drone-mounted camera")
        
        # Give camera time to warm up
        time.sleep(1)
        
        # Initialize ArUco detector
        self.dictionary_id = dictionary_id
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Optimize parameters for drone vibration and fixed focus
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE  # Faster processing
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        self.aruco_params.minCornerDistanceRate = 0.05
        
        # Create detector with enhanced parameters
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # The specific marker ID we're looking for
        self.target_marker_id = marker_id
        
        # Marker detection storage
        self.marker_id = None
        self.marker_info_saved = False
        self.save_directory = "marker_detection"
        
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
            # Get image size from camera
            frame = self.capture_frame()
            h, w = frame.shape[:2]
            self.img_size = (w, h)
                
            # Estimate camera matrix (approximate)
            self.focal_length_x = self.focal_length_y = w
            self.principal_point_x = w / 2
            self.principal_point_y = h / 2
                
            self.camera_matrix = np.array([
                [self.focal_length_x, 0, self.principal_point_x],
                [0, self.focal_length_y, self.principal_point_y],
                [0, 0, 1]
            ])
                
            self.dist_coeffs = np.zeros((5, 1))
        
        # Known size of marker in cm
        self.known_marker_size_cm = known_marker_size_cm
        
        # Convert to meters for MAVLink
        self.known_marker_size_m = known_marker_size_cm / 100.0
        
        # Define 3D coordinates of marker corners (assuming flat and square)
        half_size = self.known_marker_size_m / 2
        self.object_points = np.array([
            [-half_size, -half_size, 0],  # Top-left
            [half_size, -half_size, 0],   # Top-right
            [half_size, half_size, 0],    # Bottom-right
            [-half_size, half_size, 0]    # Bottom-left
        ], dtype=np.float32)
        
        # Initialize Kalman filter for smoother tracking
        self.kalman = cv2.KalmanFilter(8, 4)  # 8 state variables, 4 measurement variables
        
        # Adjust state transition matrix for faster response to drone movement
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
            [0, 0, 0, 0, 0.7, 0, 0, 0],  # dx decays (drone movement changes quickly)
            [0, 0, 0, 0, 0, 0.7, 0, 0],  # dy decays
            [0, 0, 0, 0, 0, 0, 0.7, 0],  # dw decays
            [0, 0, 0, 0, 0, 0, 0, 0.7]   # dh decays
        ], dtype=np.float32)

        # Measurement matrix
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # Measure x
            [0, 1, 0, 0, 0, 0, 0, 0],  # Measure y
            [0, 0, 1, 0, 0, 0, 0, 0],  # Measure w
            [0, 0, 0, 1, 0, 0, 0, 0]   # Measure h
        ], dtype=np.float32)
        
        # Set process noise covariance higher for drone vibration
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.1
        
        # Set measurement noise covariance
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        
        # Initialize Kalman filter
        self.kalman.statePre = np.zeros((8, 1), dtype=np.float32)
        self.kalman.statePost = np.zeros((8, 1), dtype=np.float32)
        
        # Flag for Kalman filter initialization
        self.kalman_initialized = False
        
        # Variables for tracking moving average of corners
        self.corner_history = []
        self.max_history = 5
        
        # Flag for image enhancement
        self.enhance_image = True

    def capture_frame(self):
        """Capture a frame from picamera2."""
        return self.picam2.capture_array()
    
    def preprocess_image(self, frame):
        """Apply image enhancement for better marker detection."""
        if not self.enhance_image:
            if len(frame.shape) == 3:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), frame
            return frame, frame
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.copy()
        
        # Apply light blur to reduce noise
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Create a copy for visualization
        if len(frame.shape) == 3:
            enhanced_color = frame.copy()
        else:
            enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced, enhanced_color
    
    def detect_markers(self, frame):
        """
        Detect ArUco markers in the frame.
        Returns corners, ids, and rejected candidates
        """
        # Preprocess the image
        enhanced, enhanced_color = self.preprocess_image(frame)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(enhanced)
        
        # Save marker info if detected and not already saved
        if ids is not None and len(ids) > 0 and not self.marker_info_saved:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.target_marker_id or self.target_marker_id == -1:
                    self.marker_id = int(marker_id)
                    print(f"ArUco marker detected: ID {self.marker_id}")
                    
                    # Save ID to a file
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{self.save_directory}/aruco_marker_id_{timestamp}.txt"
                    with open(filename, 'w') as f:
                        dict_name = self.get_dictionary_name(self.dictionary_id)
                        f.write(f"ArUco Dictionary: {dict_name}\nMarker ID: {self.marker_id}")
                    print(f"Marker info saved to {filename}")
                    self.marker_info_saved = True
                    break
        
        return corners, ids, rejected, enhanced_color
    
    def get_dictionary_name(self, dict_id):
        """Get the name of the ArUco dictionary from its ID."""
        dict_names = {
            cv2.aruco.DICT_4X4_50: "DICT_4X4_50",
            cv2.aruco.DICT_4X4_100: "DICT_4X4_100",
            cv2.aruco.DICT_4X4_250: "DICT_4X4_250",
            cv2.aruco.DICT_4X4_1000: "DICT_4X4_1000",
            cv2.aruco.DICT_5X5_50: "DICT_5X5_50",
            cv2.aruco.DICT_5X5_100: "DICT_5X5_100",
            cv2.aruco.DICT_5X5_250: "DICT_5X5_250",
            cv2.aruco.DICT_5X5_1000: "DICT_5X5_1000",
            cv2.aruco.DICT_6X6_50: "DICT_6X6_50",
            cv2.aruco.DICT_6X6_100: "DICT_6X6_100",
            cv2.aruco.DICT_6X6_250: "DICT_6X6_250",
            cv2.aruco.DICT_6X6_1000: "DICT_6X6_1000",
            cv2.aruco.DICT_7X7_50: "DICT_7X7_50",
            cv2.aruco.DICT_7X7_100: "DICT_7X7_100",
            cv2.aruco.DICT_7X7_250: "DICT_7X7_250",
            cv2.aruco.DICT_7X7_1000: "DICT_7X7_1000",
            cv2.aruco.DICT_ARUCO_ORIGINAL: "DICT_ARUCO_ORIGINAL"
        }
        return dict_names.get(dict_id, f"Unknown Dictionary ID: {dict_id}")
    
    def update_kalman(self, corners):
        """Update Kalman filter with detected corners."""
        if corners is None or len(corners) == 0:
            return None
        
        # Extract the corners of the first detected marker
        marker_corners = corners[0][0]
        
        # Calculate center point and dimensions
        center_x = np.mean(marker_corners[:, 0])
        center_y = np.mean(marker_corners[:, 1])
        width = np.linalg.norm(marker_corners[0] - marker_corners[1])
        height = np.linalg.norm(marker_corners[0] - marker_corners[3])
        
        # Create measurement
        measurement = np.array([[center_x], [center_y], [width], [height]], dtype=np.float32)
        
        # If this is the first detection, initialize the Kalman filter state
        if not self.kalman_initialized:
            self.kalman.statePre[0] = center_x
            self.kalman.statePre[1] = center_y
            self.kalman.statePre[2] = width
            self.kalman.statePre[3] = height
            self.kalman.statePost = self.kalman.statePre.copy()
            self.kalman_initialized = True
            return marker_corners
        
        # Predict next state
        prediction = self.kalman.predict()
        
        # Update with measurement
        corrected = self.kalman.correct(measurement)
        
        # Get corrected values
        corrected_center_x = corrected[0, 0]
        corrected_center_y = corrected[1, 0]
        corrected_width = corrected[2, 0]
        corrected_height = corrected[3, 0]
        
        # Calculate corrected corners
        half_width = corrected_width / 2
        half_height = corrected_height / 2
        
        # Apply corrected values to smooth corner positions
        corrected_corners = np.array([
            [corrected_center_x - half_width, corrected_center_y - half_height],
            [corrected_center_x + half_width, corrected_center_y - half_height],
            [corrected_center_x + half_width, corrected_center_y + half_height],
            [corrected_center_x - half_width, corrected_center_y + half_height]
        ], dtype=np.float32)
        
        # Update corner history
        self.corner_history.append(corrected_corners)
        if len(self.corner_history) > self.max_history:
            self.corner_history.pop(0)
        
        # Calculate moving average of corners for even smoother tracking
        smooth_corners = np.mean(self.corner_history, axis=0)
        return smooth_corners
    
    def calculate_landing_target_params(self, corners):
        """Calculate MAVLink landing target parameters from marker corners."""
        if corners is None or len(corners) == 0:
            return None
            
        # Use Kalman filtered corners for smoother tracking
        smooth_corners = self.update_kalman(corners)
        
        if smooth_corners is None:
            return None
        
        # Reshape corners for solvePnP if needed
        if isinstance(smooth_corners, np.ndarray) and smooth_corners.shape == (4, 2):
            # We already have smoothed corners
            corners_array = smooth_corners
        else:
            # Extract corners from the first detected marker
            corners_array = smooth_corners[0].reshape(4, 2)
        
        # Solve PnP to get rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(
            self.object_points, corners_array, self.camera_matrix, self.dist_coeffs
        )
        
        if not success:
            return None
        
        # Extract x, y, z coordinates from translation vector
        tx, ty, tz = tvec.flatten()
        
        # Distance is the Z component of the translation vector (in meters)
        distance = tz
        
        # Calculate the center of the marker in pixel coordinates
        center_x = np.mean(corners_array[:, 0])
        center_y = np.mean(corners_array[:, 1])
        
        # Calculate angular offset from image center
        dx_pixels = center_x - self.principal_point_x
        dy_pixels = center_y - self.principal_point_y
        
        # Convert to radians
        angle_x = math.atan2(dx_pixels, self.focal_length_x)
        angle_y = math.atan2(dy_pixels, self.focal_length_y)
        
        # Calculate the width and height of the marker in pixels
        width = np.linalg.norm(corners_array[0] - corners_array[1])
        height = np.linalg.norm(corners_array[0] - corners_array[3])
        
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
            'corners': corners_array,
            'center': (int(center_x), int(center_y)),
            'marker_id': self.marker_id if self.marker_id is not None else -1
        }
    
    def run(self):
        """Run the ArUco marker landing target tracker."""
        print("Starting ArUco Marker Landing Target Tracker...")
        print("Press 'q' to quit")
        print("Press 'e' to toggle image enhancement")
        
        frame_count = 0
        fps_start = time.time()
        fps = 0
        
        while True:
            # Capture frame
            frame = self.capture_frame()
            
            if frame is None:
                print("Error: Could not capture frame.")
                break
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                now = time.time()
                fps = frame_count / (now - fps_start)
                fps_start = now
                frame_count = 0
            
            # Detect ArUco markers
            corners, ids, rejected, display_img = self.detect_markers(frame)
            
            # Draw crosshair at image center
            cv2.drawMarker(display_img, 
                          (int(self.principal_point_x), int(self.principal_point_y)), 
                          (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            
            # Display mode and FPS
            cv2.putText(display_img, f"Enhancement: {'ON' if self.enhance_image else 'OFF'}", 
                       (10, display_img.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display_img, f"FPS: {fps:.1f}", 
                       (10, display_img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if ids is not None and len(ids) > 0:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
                
                # Find matching marker ID
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == self.target_marker_id or self.target_marker_id == -1:
                        # Calculate landing target parameters
                        params = self.calculate_landing_target_params([corners[i]])
                        
                        if params:
                            # Display target information
                            self.display_target_info(display_img, params)
                        break
            
            # Display the frame
            cv2.imshow('ArUco Marker Landing Target Tracker', display_img)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                # Toggle image enhancement
                self.enhance_image = not self.enhance_image
                print(f"Image enhancement: {'ON' if self.enhance_image else 'OFF'}")
        
        # Release resources
        self.close()
    
    def display_target_info(self, display_img, params):
        """Display landing target information on the image."""
        # Draw coordinate axes
        try:
            # Draw coordinate axes
            axes_length = 0.05  # Length of axes in meters
            imgpts, _ = cv2.projectPoints(
                np.array([
                    [0, 0, 0],
                    [axes_length, 0, 0],
                    [0, axes_length, 0],
                    [0, 0, axes_length]
                ], dtype=np.float32),
                params['rvec'], params['tvec'], self.camera_matrix, self.dist_coeffs
            )
            # Check for valid points before drawing
            if not np.isnan(imgpts).any() and not np.isinf(imgpts).any():
                imgpts = imgpts.astype(int)
                origin = tuple(imgpts[0].ravel())
                # Additional safety check for valid coordinates
                if (0 <= origin[0] < display_img.shape[1] and 
                    0 <= origin[1] < display_img.shape[0]):
                    # Only draw if coordinates are within image bounds
                    display_img = cv2.line(display_img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 2)  # X-axis (red)
                    display_img = cv2.line(display_img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 2)  # Y-axis (green)
                    display_img = cv2.line(display_img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 2)  # Z-axis (blue)
        except Exception as e:
            # Silently handle projection errors
            pass
        
        # Format MAVLink parameters for display
        text_lines = [
            f"Marker ID: {params['marker_id']}",
            f"angle_x: {params['angle_x']:.4f} rad",
            f"angle_y: {params['angle_y']:.4f} rad",
            f"distance: {params['distance']:.4f} m",
            f"size_x: {params['size_x']:.4f} rad",
            f"size_y: {params['size_y']:.4f} rad"
        ]
        
        # Display parameters
        y0 = 30
        for i, line in enumerate(text_lines):
            y = y0 + i * 30
            cv2.putText(display_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw direction vector to target
        center = params['center']
        cv2.line(display_img, 
                (int(self.principal_point_x), int(self.principal_point_y)), 
                center, (0, 255, 255), 2)
        
        # Calculate distance in pixels from center
        dx = center[0] - self.principal_point_x
        dy = center[1] - self.principal_point_y
        pixel_distance = math.sqrt(dx*dx + dy*dy)
        
        # Display pixel distance
        cv2.putText(display_img, f"Pixel offset: {pixel_distance:.1f}", 
                   (center[0] + 10, center[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Print MAVLink parameters to console (for integration with MAVLink)
        print("MAVLink Landing Target Parameters:")
        for line in text_lines:
            print(f"  {line}")
        print("---")
    
    def close(self):
        """Release resources."""
        self.picam2.stop()
        cv2.destroyAllWindows()
        print("Resources released")

class ArucoMarkerGenerator:
    """Utility class to generate ArUco marker images."""
    
    @staticmethod
    def generate_marker(dictionary_id=cv2.aruco.DICT_4X4_50, marker_id=0, size_pixels=300, border_bits=1, save=True, filename=None):
        """Generate and save an ArUco marker image."""
        # Get dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        
        # Create marker image
        try:
            # Try with positional argument (works in OpenCV 4.11.0)
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels, border_bits)
        except Exception:
            try:
                # Fallback without border_bits parameter
                marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)
                print("Warning: border_bits parameter not supported, using default border.")
            except Exception as e:
                print(f"Error generating marker: {e}")
                print("Trying alternative method...")
                # Fallback to older OpenCV versions API if needed
                marker_img = np.zeros((size_pixels, size_pixels), dtype=np.uint8)
                cv2.aruco.drawMarker(aruco_dict, marker_id, size_pixels, marker_img, border_bits)
                print("Used legacy drawMarker method instead.")
        
        # Convert to color for better visualization
        marker_img_color = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        
        # Add a white border for printing
        border_size = size_pixels // 10
        marker_with_border = cv2.copyMakeBorder(
            marker_img_color,
            border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        # Add text with marker info
        dict_name = {
            cv2.aruco.DICT_4X4_50: "4x4_50",
            cv2.aruco.DICT_5X5_50: "5x5_50",
            cv2.aruco.DICT_6X6_50: "6x6_50",
            cv2.aruco.DICT_7X7_50: "7x7_50",
            cv2.aruco.DICT_4X4_100: "4x4_100",
            cv2.aruco.DICT_5X5_100: "5x5_100",
            cv2.aruco.DICT_6X6_100: "6x6_100",
            cv2.aruco.DICT_7X7_100: "7x7_100",
            cv2.aruco.DICT_4X4_250: "4x4_250",
            cv2.aruco.DICT_5X5_250: "5x5_250",
            cv2.aruco.DICT_6X6_250: "6x6_250",
            cv2.aruco.DICT_7X7_250: "7x7_250",
            cv2.aruco.DICT_4X4_1000: "4x4_1000",
            cv2.aruco.DICT_5X5_1000: "5x5_1000",
            cv2.aruco.DICT_6X6_1000: "6x6_1000",
            cv2.aruco.DICT_7X7_1000: "7x7_1000",
            cv2.aruco.DICT_ARUCO_ORIGINAL: "ORIGINAL"
        }.get(dictionary_id, f"Dict_{dictionary_id}")
        
        cv2.putText(
            marker_with_border,
            f"ArUco {dict_name} - ID: {marker_id}",
            (border_size, marker_with_border.shape[0] - border_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        
        # Save the marker if requested
        if save:
            if filename is None:
                # Create directory if it doesn't exist
                if not os.path.exists("aruco_markers"):
                    os.makedirs("aruco_markers")
                filename = f"aruco_markers/aruco_{dict_name}_id{marker_id}.png"
            
            cv2.imwrite(filename, marker_with_border)
            print(f"ArUco marker saved to {filename}")
        
        return marker_with_border
    
    @staticmethod
    def generate_marker_board(dictionary_id=cv2.aruco.DICT_6X6_250, marker_ids=None, grid_size=(2, 2), 
                              marker_size_cm=10.0, separation_cm=1.0, size_pixels=2000, save=True, filename=None):
        """Generate and save an ArUco marker board image with specified grid."""
        if marker_ids is None:
            # Use sequential IDs starting from 0 if not specified
            marker_ids = list(range(grid_size[0] * grid_size[1]))
        
        if len(marker_ids) != grid_size[0] * grid_size[1]:
            raise ValueError(f"Number of marker IDs ({len(marker_ids)}) doesn't match grid size ({grid_size[0]}x{grid_size[1]})")
        
        # Get dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        
        # Calculate pixels per cm
        total_width_cm = grid_size[0] * marker_size_cm + (grid_size[0] + 1) * separation_cm
        total_height_cm = grid_size[1] * marker_size_cm + (grid_size[1] + 1) * separation_cm
        
        # Calculate aspect ratio
        aspect_ratio = total_width_cm / total_height_cm
        
        # Determine final image dimensions while maintaining aspect ratio
        if aspect_ratio >= 1.0:
            img_width = size_pixels
            img_height = int(size_pixels / aspect_ratio)
        else:
            img_height = size_pixels
            img_width = int(size_pixels * aspect_ratio)
        
        # Calculate marker and separation size in pixels
        px_per_cm = img_width / total_width_cm
        marker_size_px = int(marker_size_cm * px_per_cm)
        separation_px = int(separation_cm * px_per_cm)
        
        # Create a white background
        board_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # Function to compute the position of a marker in the grid
        def marker_position(row, col):
            x = separation_px + col * (marker_size_px + separation_px)
            y = separation_px + row * (marker_size_px + separation_px)
            return x, y
        
        # Draw each marker
        marker_idx = 0
        for row in range(grid_size[1]):
            for col in range(grid_size[0]):
                if marker_idx < len(marker_ids):
                    # Generate marker
                    marker_id = marker_ids[marker_idx]
                    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
                    
                    # Convert to BGR
                    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
                    
                    # Place marker on board
                    x, y = marker_position(row, col)
                    board_img[y:y+marker_size_px, x:x+marker_size_px] = marker_bgr
                    
                    # Add marker ID label
                    label_x = x + marker_size_px // 2 - 20
                    label_y = y + marker_size_px + 20
                    if label_y < img_height - 10:  # Ensure label is within image bounds
                        cv2.putText(
                            board_img,
                            f"ID: {marker_id}",
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA
                        )
                    
                    marker_idx += 1
        
        # Add board information
        dict_name = {
            cv2.aruco.DICT_4X4_50: "4x4_50",
            cv2.aruco.DICT_5X5_50: "5x5_50",
            cv2.aruco.DICT_6X6_50: "6x6_50",
            cv2.aruco.DICT_7X7_50: "7x7_50",
            cv2.aruco.DICT_4X4_100: "4x4_100",
            cv2.aruco.DICT_5X5_100: "5x5_100",
            cv2.aruco.DICT_6X6_100: "6x6_100",
            cv2.aruco.DICT_7X7_100: "7x7_100",
            cv2.aruco.DICT_4X4_250: "4x4_250",
            cv2.aruco.DICT_5X5_250: "5x5_250",
            cv2.aruco.DICT_6X6_250: "6x6_250",
            cv2.aruco.DICT_7X7_250: "7x7_250",
            cv2.aruco.DICT_4X4_1000: "4x4_1000",
            cv2.aruco.DICT_5X5_1000: "5x5_1000",
            cv2.aruco.DICT_6X6_1000: "6x6_1000",
            cv2.aruco.DICT_7X7_1000: "7x7_1000",
            cv2.aruco.DICT_ARUCO_ORIGINAL: "ORIGINAL"
        }.get(dictionary_id, f"Dict_{dictionary_id}")
        
        info_text = f"ArUco Board - Dictionary: {dict_name} - Size: {marker_size_cm}cm - Separation: {separation_cm}cm"
        cv2.putText(
            board_img,
            info_text,
            (20, img_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        
        # Save the board if requested
        if save:
            if filename is None:
                # Create directory if it doesn't exist
                if not os.path.exists("aruco_markers"):
                    os.makedirs("aruco_markers")
                filename = f"aruco_markers/aruco_board_{dict_name}_{grid_size[0]}x{grid_size[1]}.png"
            
            cv2.imwrite(filename, board_img)
            print(f"ArUco marker board saved to {filename}")
        
        return board_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArUco Marker Landing Target Tracker')
    parser.add_argument('--calibration', type=str, default='camera_calibration.pkl', 
                        help='Path to camera calibration file (default: camera_calibration.pkl)')
    parser.add_argument('--marker-size', type=float, default=10.0, 
                        help='Known size of marker in cm (default: 10.0)')
    parser.add_argument('--resolution', type=str, default='1280,720',
                        help='Camera resolution in format "width,height" (default: 1280,720)')
    parser.add_argument('--dictionary', type=int, default=cv2.aruco.DICT_6X6_250,
                        help='ArUco dictionary ID (default: DICT_6X6_250)')
    parser.add_argument('--marker-id', type=int, default=0,
                        help='Specific marker ID to track (-1 for any marker) (default: 0)')
    parser.add_argument('--generate-marker', action='store_true',
                        help='Generate an ArUco marker and exit')
    parser.add_argument('--generate-board', action='store_true',
                        help='Generate an ArUco marker board and exit')
    parser.add_argument('--marker-pixels', type=int, default=1000,
                        help='Size of generated marker in pixels (default: 1000)')
    parser.add_argument('--grid-size', type=str, default='2,2',
                        help='Grid size for marker board in format "cols,rows" (default: 2,2)')
    
    args = parser.parse_args()
    
    # Parse grid size for board generation
    if ',' in args.grid_size:
        grid_cols, grid_rows = map(int, args.grid_size.replace(' ', '').split(','))
        grid_size = (grid_cols, grid_rows)
    else:
        grid_size = (2, 2)
    
    # If generate marker flag is set, generate marker and exit
    if args.generate_marker:
        ArucoMarkerGenerator.generate_marker(
            dictionary_id=args.dictionary,
            marker_id=args.marker_id,
            size_pixels=args.marker_pixels
        )
        exit(0)
    
    # If generate board flag is set, generate marker board and exit
    if args.generate_board:
        ArucoMarkerGenerator.generate_marker_board(
            dictionary_id=args.dictionary,
            grid_size=grid_size,
            marker_size_cm=args.marker_size,
            size_pixels=args.marker_pixels
        )
        exit(0)
    
    # Parse resolution
    width, height = map(int, args.resolution.replace(' ', '').split(','))
    
    try:
        tracker = ArucoLandingTargetTracker(
            calibration_file=args.calibration,
            known_marker_size_cm=args.marker_size,
            camera_resolution=(width, height),
            dictionary_id=args.dictionary,
            marker_id=args.marker_id
        )
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'tracker' in locals():
            tracker.close()
