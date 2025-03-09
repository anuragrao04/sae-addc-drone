#!/usr/bin/env python3
import numpy as np
import cv2
import math
import pickle
import time
import os
import argparse
from pyzbar.pyzbar import decode
from picamera2 import Picamera2
from libcamera import controls

class QRLandingPadTracker:
    def __init__(self, 
                 calibration_file='camera_calibration.pkl', 
                 known_qr_size_cm=10.0,
                 camera_resolution=(640, 480),
                 min_distance_for_decode=0.5):  # 50cm threshold for QR decoding
        
        # Initialize picamera2
        self.picam2 = Picamera2()
        
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
        
        # Give camera time to warm up
        time.sleep(2)
        
        # Distance threshold for switching to decode mode
        self.min_distance_for_decode = min_distance_for_decode
        
        # Flag for QR code content
        self.qr_content = None
        self.qr_content_saved = False
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

    def capture_frame(self):
        """Capture a frame from picamera2."""
        return self.picam2.capture_array()
    
    def detect_qr_code_full(self, frame):
        """Try to detect and decode full QR code."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        qr_codes = decode(gray)
        
        if qr_codes and not self.qr_content_saved:
            # Save the QR code content
            self.qr_content = qr_codes[0].data.decode('utf-8')
            print(f"QR Code decoded: {self.qr_content}")
            
            # Save content to a file
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{self.save_directory}/qr_content_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write(self.qr_content)
            print(f"QR content saved to {filename}")
            self.qr_content_saved = True
        
        return qr_codes
    
    def detect_finder_patterns(self, frame):
        """
        Detect the three QR code finder patterns (the squares in three corners).
        Returns the coordinates of the patterns if found, or None otherwise.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # We need to identify contours that form the hierarchical pattern of the QR finder pattern
        # (a square within a square within a square)
        hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        
        # Find potential finder patterns (squares with a specific hierarchy)
        potential_finders = []
        
        # Process hierarchy
        if len(hierarchy) > 0:
            # Get hierarchy array
            hierarchy_array = hierarchy[0]
            
            # Process each contour
            for i, contour in enumerate(contours):
                # Skip small contours
                if cv2.contourArea(contour) < 100:
                    continue
                
                # Check if approximately square shaped
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                
                if len(approx) == 4:  # It's a quadrilateral
                    # Check if it's approximately square
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    
                    if 0.8 <= aspect_ratio <= 1.2:  # Square-ish
                        potential_finders.append((i, contour, approx))
        
        # Group finder patterns that are close to each other
        finder_patterns = []
        processed = set()
        
        for i, contour, approx in potential_finders:
            if i in processed:
                continue
                
            # Get center of this contour
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Check if it forms a corner with two other patterns
            matches = []
            for j, other_contour, other_approx in potential_finders:
                if j in processed or i == j:
                    continue
                
                # Get center of other contour
                M_other = cv2.moments(other_contour)
                if M_other["m00"] == 0:
                    continue
                
                cx_other = int(M_other["m10"] / M_other["m00"])
                cy_other = int(M_other["m01"] / M_other["m00"])
                
                # Check if the centers are within a reasonable distance
                distance = np.sqrt((cx - cx_other)**2 + (cy - cy_other)**2)
                
                # This threshold depends on the expected size of the QR code
                if distance < 300:  # Adjust based on your needs
                    matches.append((j, other_contour, other_approx, (cx_other, cy_other)))
            
            # If we have found at least 2 potential matches, this might be part of a finder pattern
            if len(matches) >= 2:
                # Mark this and matched contours as processed
                processed.add(i)
                corner_points = [approx]
                centers = [(cx, cy)]
                
                for j, other_contour, other_approx, center in matches:
                    processed.add(j)
                    corner_points.append(other_approx)
                    centers.append(center)
                
                # We have found a potential finder pattern
                finder_patterns.append((corner_points, centers))
        
        # Now filter further to find three patterns in a square arrangement
        if len(finder_patterns) >= 3:
            # Calculate centers of all finder patterns
            all_centers = []
            for _, centers in finder_patterns:
                for center in centers:
                    all_centers.append(center)
            
            # If there are at least 3 centers, we can check if they form a right angle
            if len(all_centers) >= 3:
                # Convert to numpy array for easier manipulation
                centers_array = np.array(all_centers)
                
                # Find the corners of the QR code by analyzing the centers
                hull = cv2.convexHull(np.array(centers_array, dtype=np.int32))
                
                # If we have a quadrilateral, we likely have a QR code
                if 3 <= len(hull) <= 4:
                    return hull
        
        # No finder patterns in a valid arrangement found
        return None
    
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
    
    def run(self):
        """Run the QR code landing pad tracker."""
        print("Starting QR Landing Pad Tracker...")
        print("Press 'q' to quit")
        
        while True:
            # Capture frame
            frame = self.capture_frame()
            
            if frame is None:
                print("Error: Could not capture frame.")
                break
            
            # Create a copy for visualization
            display_img = frame.copy()
            
            # Draw crosshair at image center
            cv2.drawMarker(display_img, 
                          (int(self.principal_point_x), int(self.principal_point_y)), 
                          (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            
            # Try to detect full QR code first (if we're close enough)
            qr_codes = self.detect_qr_code_full(frame)
            
            if qr_codes:
                # We detected a full QR code!
                for qr_code in qr_codes:
                    # Draw the QR code boundary
                    points = np.array(qr_code.polygon, dtype=np.int32)
                    cv2.polylines(display_img, [points], True, (0, 255, 0), 2)
                    
                    # Calculate landing target parameters
                    corners = np.array([(p.x, p.y) for p in qr_code.polygon], dtype=np.float32)
                    params = self.calculate_landing_target_params(corners)
                    
                    if params:
                        # We're tracking the QR code
                        self.display_target_info(display_img, params, True)
            else:
                # Try to detect finder patterns (for long-range detection)
                finder_patterns = self.detect_finder_patterns(frame)
                
                if finder_patterns is not None:
                    # Draw the finder patterns
                    cv2.polylines(display_img, [finder_patterns], True, (0, 0, 255), 2)
                    
                    # Calculate landing target parameters
                    params = self.calculate_landing_target_params(finder_patterns)
                    
                    if params:
                        # We're tracking the finder patterns
                        self.display_target_info(display_img, params, False)
            
            # Display the frame
            cv2.imshow('QR Landing Pad Tracker', display_img)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.close()
    
    def display_target_info(self, display_img, params, full_qr_detected):
        """Display landing target information on the image."""
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
        
        imgpts = imgpts.astype(int)
        origin = tuple(imgpts[0].ravel())
        display_img = cv2.line(display_img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 2)  # X-axis (red)
        display_img = cv2.line(display_img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 2)  # Y-axis (green)
        display_img = cv2.line(display_img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 2)  # Z-axis (blue)
        
        # Format MAVLink parameters for display
        text_lines = [
            f"angle_x: {params['angle_x']:.4f} rad",
            f"angle_y: {params['angle_y']:.4f} rad",
            f"distance: {params['distance']:.4f} m",
            f"size_x: {params['size_x']:.4f} rad",
            f"size_y: {params['size_y']:.4f} rad"
        ]
        
        if full_qr_detected:
            text_lines.append("Mode: FULL QR DETECTED")
            
            # If we've decoded the QR content, display it
            if self.qr_content:
                text_lines.append(f"QR Content: {self.qr_content[:20]}...")
        else:
            text_lines.append("Mode: FINDER PATTERNS ONLY")
        
        # Check if we should attempt to decode QR content
        if not self.qr_content_saved and params['distance'] < self.min_distance_for_decode:
            text_lines.append("Distance OK for QR decode")
        
        # Display parameters
        y0 = 30
        for i, line in enumerate(text_lines):
            y = y0 + i * 30
            cv2.putText(display_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QR Code Landing Pad Tracker')
    parser.add_argument('--calibration', type=str, default='camera_calibration.pkl', 
                        help='Path to camera calibration file (default: camera_calibration.pkl)')
    parser.add_argument('--qr-size', type=float, default=10.0, 
                        help='Known size of QR code in cm (default: 10.0)')
    parser.add_argument('--resolution', type=str, default='640,480',
                        help='Camera resolution in format "width,height" (default: 640,480)')
    parser.add_argument('--decode-distance', type=float, default=0.5,
                        help='Minimum distance in meters for QR code decoding (default: 0.5)')
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split(','))
    
    try:
        tracker = QRLandingPadTracker(
            calibration_file=args.calibration,
            known_qr_size_cm=args.qr_size,
            camera_resolution=(width, height),
            min_distance_for_decode=args.decode_distance
        )
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'tracker' in locals():
            tracker.close()
