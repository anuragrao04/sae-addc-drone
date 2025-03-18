#!/usr/bin/env python3
import numpy as np
import cv2
import math
import time
import os
import argparse
from picamera2 import Picamera2
from libcamera import controls

class SimpleArucoTracker:
    def __init__(self, marker_size_cm=15.0, camera_resolution=(1280, 720), marker_id=0):
        # Initialize camera
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(
            main={"size": camera_resolution, "format": "RGB888"}
        )
        self.picam2.configure(config)
        
        # Enable auto-exposure for better visibility
        self.picam2.set_controls({
            "AfMode": controls.AfModeEnum.Manual,  # Fixed focus
            "AeEnable": True,                      # Auto exposure
            "AnalogueGain": 2.0                    # Moderate gain
        })
        
        self.picam2.start()
        print("Camera initialized")
        time.sleep(1)  # Warm up
        
        # Fixed dictionary for maximum distance (DICT_5X5_50)
        self.dictionary_id = cv2.aruco.DICT_5X5_50
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        
        # Create detector parameters optimized for distance
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Optimize for long-distance detection
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.minMarkerPerimeterRate = 0.01  # Detect smaller (distant) markers
        self.aruco_params.adaptiveThreshConstant = 5     # More sensitive threshold
        
        # Marker ID to track
        self.marker_id = marker_id
        
        # Physical marker size (for distance estimation)
        self.marker_size_cm = marker_size_cm
        self.marker_size_m = marker_size_cm / 100.0
        
        # Camera matrix (estimated if no calibration)
        frame = self.capture_frame()
        h, w = frame.shape[:2]
        
        # Estimate camera matrix
        self.focal_length = w
        self.center_x = w / 2
        self.center_y = h / 2
        
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center_x],
            [0, self.focal_length, self.center_y],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros((5, 1))
        
        # 3D points for distance calculation
        half_size = self.marker_size_m / 2
        self.object_points = np.array([
            [-half_size, -half_size, 0],  # Top-left
            [half_size, -half_size, 0],   # Top-right
            [half_size, half_size, 0],    # Bottom-right
            [-half_size, half_size, 0]    # Bottom-left
        ], dtype=np.float32)
        
        # Image enhancement toggle
        self.enhance_image = True

    def capture_frame(self):
        """Capture a frame from the camera"""
        return self.picam2.capture_array()
    
    def preprocess_image(self, frame):
        """Basic preprocessing to enhance marker detection"""
        if not self.enhance_image:
            if len(frame.shape) == 3:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), frame
            return frame, frame
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.copy()
        
        # Enhance for better detection
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Optional: apply sharpening for better corner detection
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened, frame
    
    def detect_marker(self, frame):
        """Detect ArUco markers in the frame"""
        # Preprocess the image
        enhanced, original = self.preprocess_image(frame)
        
        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            enhanced, self.aruco_dict, parameters=self.aruco_params)
        
        # Filter for specified marker ID
        if ids is not None and len(ids) > 0 and self.marker_id != -1:
            filtered_corners = []
            filtered_ids = []
            
            for i, id in enumerate(ids.flatten()):
                if id == self.marker_id:
                    filtered_corners.append(corners[i])
                    filtered_ids.append([id])
            
            if filtered_corners:
                corners = filtered_corners
                ids = np.array(filtered_ids)
            else:
                corners = []
                ids = None
        
        return corners, ids, original
    
    def calculate_distance(self, corners):
        """Calculate distance to marker using PnP"""
        if not corners:
            return None
            
        # Extract corners from the first detected marker
        marker_corners = corners[0].reshape(4, 2)
        
        # Solve PnP to get rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(
            self.object_points, marker_corners, self.camera_matrix, self.dist_coeffs)
        
        if not success:
            return None
        
        # Extract distance (z-component of translation vector)
        distance = tvec[2][0]
        
        # Calculate center of marker
        center_x = np.mean(marker_corners[:, 0])
        center_y = np.mean(marker_corners[:, 1])
        
        # Calculate angle to marker
        angle_x = math.atan2(center_x - self.center_x, self.focal_length)
        angle_y = math.atan2(center_y - self.center_y, self.focal_length)
        
        return {
            'distance': distance,
            'angle_x': angle_x,
            'angle_y': angle_y,
            'center': (int(center_x), int(center_y)),
            'rvec': rvec,
            'tvec': tvec
        }
    
    def run(self):
        """Run the marker tracking loop"""
        print("Starting ArUco Marker Tracker optimized for maximum distance")
        print("Press 'q' to quit, 'e' to toggle enhancement")
        
        while True:
            # Capture frame
            frame = self.capture_frame()
            if frame is None:
                print("Error capturing frame")
                continue
            
            # Detect markers
            corners, ids, display_img = self.detect_marker(frame)
            
            # Draw crosshair at image center
            cv2.drawMarker(display_img, 
                          (int(self.center_x), int(self.center_y)), 
                          (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            
            # Draw status text
            cv2.putText(display_img, f"Enhancement: {'ON' if self.enhance_image else 'OFF'}", 
                       (10, display_img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if ids is not None and len(ids) > 0:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
                
                # Calculate distance
                params = self.calculate_distance(corners)
                if params:
                    # Draw axes manually (compatible with all OpenCV versions)
                    try:
                        # Project axis points
                        axis_length = self.marker_size_m / 2
                        axis_points = np.array([
                            [0, 0, 0],           # Origin
                            [axis_length, 0, 0], # X-axis
                            [0, axis_length, 0], # Y-axis
                            [0, 0, axis_length]  # Z-axis
                        ], dtype=np.float32)
                        
                        imgpts, _ = cv2.projectPoints(
                            axis_points, params['rvec'], params['tvec'], 
                            self.camera_matrix, self.dist_coeffs
                        )
                        
                        # Draw each axis 
                        origin = tuple(imgpts[0].ravel().astype(int))
                        x_point = tuple(imgpts[1].ravel().astype(int))
                        y_point = tuple(imgpts[2].ravel().astype(int))
                        z_point = tuple(imgpts[3].ravel().astype(int))
                        
                        display_img = cv2.line(display_img, origin, x_point, (0, 0, 255), 3) # X-axis: Red
                        display_img = cv2.line(display_img, origin, y_point, (0, 255, 0), 3) # Y-axis: Green
                        display_img = cv2.line(display_img, origin, z_point, (255, 0, 0), 3) # Z-axis: Blue
                    except Exception as e:
                        print(f"Warning: Could not draw axes - {e}")
                    
                    # Draw distance info
                    distance_text = f"Distance: {params['distance']:.2f}m"
                    cv2.putText(display_img, distance_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Draw angle info
                    angle_text = f"Angle X: {math.degrees(params['angle_x']):.1f}° Y: {math.degrees(params['angle_y']):.1f}°"
                    cv2.putText(display_img, angle_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Draw line from center to marker
                    cv2.line(display_img, 
                            (int(self.center_x), int(self.center_y)), 
                            params['center'], (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('ArUco Marker Tracker', display_img)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                self.enhance_image = not self.enhance_image
        
        # Release resources
        self.close()
    
    def close(self):
        """Release resources"""
        self.picam2.stop()
        cv2.destroyAllWindows()
        print("Tracker stopped")

def generate_marker(marker_id=0, size_pixels=1000):
    """Generate an ArUco marker image"""
    # Fixed dictionary for maximum distance (DICT_5X5_50)
    dictionary_id = cv2.aruco.DICT_5X5_50
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    
    # Create directory if it doesn't exist
    if not os.path.exists("aruco_markers"):
        os.makedirs("aruco_markers")
    
    # Generate marker image
    try:
        # Try with positional argument first
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels, 1)
    except:
        # Fallback for older OpenCV versions
        marker_img = np.zeros((size_pixels, size_pixels), dtype=np.uint8)
        cv2.aruco.drawMarker(aruco_dict, marker_id, size_pixels, marker_img, 1)
    
    # Convert to color and add border
    marker_color = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    border_size = size_pixels // 10
    marker_with_border = cv2.copyMakeBorder(
        marker_color, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    
    # Add text with marker info
    cv2.putText(
        marker_with_border,
        f"ArUco 5x5_50 - ID: {marker_id}",
        (border_size, marker_with_border.shape[0] - border_size // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )
    
    # Save the marker
    filename = f"aruco_markers/aruco_5x5_50_id{marker_id}.png"
    cv2.imwrite(filename, marker_with_border)
    print(f"Marker saved to {filename}")
    print("For optimal detection distance, print this marker as large as possible")
    print("Recommended minimum size: 15-20cm on white matte paper")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple ArUco Marker Tracker')
    parser.add_argument('--marker-size', type=float, default=15.0, 
                        help='Known size of marker in cm (default: 15.0)')
    parser.add_argument('--resolution', type=str, default='1280,720',
                        help='Camera resolution in format "width,height" (default: 1280,720)')
    parser.add_argument('--marker-id', type=int, default=0,
                        help='Specific marker ID to track (-1 for any marker) (default: 0)')
    parser.add_argument('--generate-marker', action='store_true',
                        help='Generate an ArUco marker and exit')
    parser.add_argument('--marker-pixels', type=int, default=1000,
                        help='Size of generated marker in pixels (default: 1000)')
    
    args = parser.parse_args()
    
    # If generate marker flag is set, generate marker and exit
    if args.generate_marker:
        generate_marker(marker_id=args.marker_id, size_pixels=args.marker_pixels)
        exit(0)
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.replace(' ', '').split(','))
    except:
        width, height = 1280, 720
    
    # Start tracker
    try:
        tracker = SimpleArucoTracker(
            marker_size_cm=args.marker_size,
            camera_resolution=(width, height),
            marker_id=args.marker_id
        )
        tracker.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if 'tracker' in locals():
            tracker.close()
