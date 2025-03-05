# qr_code_tracker.py
import cv2
import numpy as np
import pickle
import math
from pyzbar.pyzbar import decode
import argparse

class QRCodeTracker:
    def __init__(self, camera_id=0, calibration_file='camera_calibration.pkl', known_qr_size_cm=10.0):
        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")
        
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
            ret, frame = self.cap.read()
            if ret:
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
            else:
                raise ValueError("Could not read frame from camera")
        
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
    
    def detect_qr_code(self, frame):
        """Detect QR code in the frame and return the decode results."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        qr_codes = decode(gray)
        return qr_codes
    
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
    
    def calculate_mavlink_params(self, qr_code):
        """Calculate MAVLink landing target parameters from QR code using PnP."""
        # Get the corners of the QR code
        corners = np.array([(p.x, p.y) for p in qr_code.polygon], dtype=np.float32)
        
        if len(corners) != 4:
            # If corners are not exactly 4, this is not a valid square QR code
            return None
        
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
            'tvec': tvec
        }
    
    def run(self):
        """Run the QR code tracker."""
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Undistort the frame using camera calibration
            undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # Detect QR codes
            qr_codes = self.detect_qr_code(undistorted)
            
            # Create a copy for visualization
            display_img = undistorted.copy()
            
            # Draw crosshair at image center
            cv2.drawMarker(display_img, 
                          (int(self.principal_point_x), int(self.principal_point_y)), 
                          (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            
            # If QR codes detected
            for qr_code in qr_codes:
                # Draw the QR code boundary
                points = np.array(qr_code.polygon, dtype=np.int32)
                cv2.polylines(display_img, [points], True, (0, 255, 0), 2)
                
                # Calculate MAVLink parameters
                params = self.calculate_mavlink_params(qr_code)
                
                if params:
                    # Draw coordinate axes on the QR code
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
            
            # Display the frame
            cv2.imshow('QR Code Tracker', display_img)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
    
    def close(self):
        """Release resources."""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QR Code Tracker for MAVLink')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--calibration', type=str, default='camera_calibration.pkl', 
                        help='Path to camera calibration file (default: camera_calibration.pkl)')
    parser.add_argument('--qr-size', type=float, default=10.0, 
                        help='Known size of QR code in cm (default: 10.0)')
    
    args = parser.parse_args()
    
    try:
        tracker = QRCodeTracker(
            camera_id=args.camera,
            calibration_file=args.calibration,
            known_qr_size_cm=args.qr_size
        )
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'tracker' in locals():
            tracker.close()
