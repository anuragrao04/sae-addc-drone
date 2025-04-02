# landing_target.py

import cv2
import cv2.aruco as aruco
import pickle
import socket
import os
import math
import time

# --- Platform Detection ---
IS_RASPBERRY_PI = socket.gethostname() == "raspberrypi"

# --- Conditional Import for PiCamera ---
if IS_RASPBERRY_PI:
    try:
        from picamera2 import Picamera2
        print("Picamera2 library found.")
    except ImportError:
        print("ERROR: Picamera2 library not found on Raspberry Pi. Install it.")
        IS_RASPBERRY_PI = False # Fallback to OpenCV
else:
    print("Not running on Raspberry Pi hostname, using standard OpenCV VideoCapture.")


class LandingTarget:
    # --- Constants ---
    # Markers
    ARUCO_ID_TO_FIND = 129  # Example ID for the larger marker
    # ARUCO_MARKER_SIZE_METERS = 0.40 # Not used in this angle calculation method, but useful if using PnP
    TRANSITION_ALTITUDE = 7.0 # Altitude (meters) below which we switch to QR

    # Camera settings
    CAMERA_RESOLUTION = (1280, 720) # Width, Height - Desired Runtime Resolution

    # Files
    CALIBRATION_FILE = 'camera_calibration.pkl'

    def __init__(self, calibration_file_path=None, camera_resolution=CAMERA_RESOLUTION):
        """
        Initializes the LandingTarget detector.

        Args:
            calibration_file_path (str, optional): Path to the camera_calibration.pkl file.
                                                    Defaults to looking in the same directory.
            camera_resolution (tuple): Runtime camera resolution (width, height).
        """
        print("Initializing LandingTarget...")
        self.is_raspberry_pi = IS_RASPBERRY_PI
        self.camera_resolution = camera_resolution # Store requested runtime resolution
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera = None
        self.picam2 = None # Explicitly for Picamera2 object
        self.horizontal_fov_rad = None # Will be calculated
        self.vertical_fov_rad = None   # Will be calculated
        self.calibrated_image_size = None # Store the size used for calibration
        self._last_detected_center = None # For visualization

        # --- Load Calibration Data ---
        if calibration_file_path is None:
            # Assume calibration file is in the same directory as this script
            calibration_file_path = os.path.join(os.path.dirname(__file__), self.CALIBRATION_FILE)

        self._load_calibration(calibration_file_path)
        if self.camera_matrix is None:
             raise IOError(f"Failed to load calibration file from {calibration_file_path}")

        # --- Calculate FOV from Calibration ---
        if self.camera_matrix is not None and self.calibrated_image_size is not None:
             self._calculate_fov()
             if self.horizontal_fov_rad is None:
                 print("WARNING: Could not calculate FOV from calibration data. Precision landing may be inaccurate.")
                 # Consider adding fallback default FOV values or raising an error here if FOV is critical

        # --- Initialize Camera ---
        self._initialize_camera() # This will try to use self.camera_resolution
        if self.camera is None and self.picam2 is None:
            raise RuntimeError("Failed to initialize camera.")

        # --- Verify Runtime Resolution matches Calibration Resolution (Post-Initialization Check) ---
        # Get actual initialized resolution
        actual_width, actual_height = self._get_actual_camera_resolution()
        if actual_width is not None:
            self.camera_resolution = (actual_width, actual_height) # Update with actual resolution
            print(f"Camera successfully initialized with actual resolution: {self.camera_resolution}")
            if self.calibrated_image_size and self.calibrated_image_size != self.camera_resolution:
                print(f"WARNING: Actual runtime resolution {self.camera_resolution} differs from "
                      f"calibration resolution {self.calibrated_image_size}. "
                      "FOV calculations may be less accurate. Consider recalibrating "
                      "at the runtime resolution or running the camera at the calibration resolution.")
                # Optionally recalculate FOV based on the *actual* runtime resolution
                # print("Attempting to recalculate FOV for actual runtime resolution...")
                # self._calculate_fov(override_size=self.camera_resolution)
        else:
             print("WARNING: Could not verify actual camera resolution after initialization.")


        # --- Initialize ArUco & QR ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = aruco.DetectorParameters()
        self.qr_detector = cv2.QRCodeDetector()

        print("LandingTarget Initialized Successfully.")
        print(f"Platform: {'Raspberry Pi (Picamera2)' if self.is_raspberry_pi else 'Laptop/Other (OpenCV)'}")
        print(f"Using ArUco ID: {self.ARUCO_ID_TO_FIND} above {self.TRANSITION_ALTITUDE}m")
        print(f"Using QR Code below {self.TRANSITION_ALTITUDE}m")

    def _load_calibration(self, file_path):
        """Loads calibration data from a .pkl file."""
        try:
            with open(file_path, 'rb') as f:
                calibration_data = pickle.load(f)
            self.camera_matrix = calibration_data['camera_matrix']
            self.dist_coeffs = calibration_data['dist_coeffs']
            # Ensure 'img_size' exists and is stored
            self.calibrated_image_size = calibration_data.get('img_size')
            if self.calibrated_image_size is None:
                 print("WARNING: 'img_size' not found in calibration file.")
            print(f"Calibration data loaded successfully from {file_path}")
        except FileNotFoundError:
            print(f"ERROR: Calibration file not found at {file_path}")
            self.camera_matrix = None # Ensure state is consistent
        except Exception as e:
            print(f"ERROR: Failed to load calibration file {file_path}: {e}")
            self.camera_matrix = None

    def _calculate_fov(self, override_size=None):
         """ Calculates FOV in radians using calibration data. """
         image_size_to_use = override_size if override_size else self.calibrated_image_size
         if self.camera_matrix is None or image_size_to_use is None:
             print("Cannot calculate FOV: Missing camera matrix or image size.")
             return

         width, height = image_size_to_use
         try:
             # Pass 0 for apertureWidth and apertureHeight if unknown
             fovx_deg, fovy_deg, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(
                 self.camera_matrix,
                 image_size_to_use, # Use tuple (width, height)
                 0,  # apertureWidth (physical sensor size in mm - often unknown)
                 0   # apertureHeight (physical sensor size in mm - often unknown)
             )

             self.horizontal_fov_rad = math.radians(fovx_deg)
             self.vertical_fov_rad = math.radians(fovy_deg)

             print(f"Calculated FOV from calibration ({width}x{height}):")
             print(f"  Horizontal: {fovx_deg:.2f} degrees ({self.horizontal_fov_rad:.4f} radians)")
             print(f"  Vertical:   {fovy_deg:.2f} degrees ({self.vertical_fov_rad:.4f} radians)")

         except Exception as e:
             print(f"Error calculating FOV using cv2.calibrationMatrixValues: {e}")
             self.horizontal_fov_rad = None
             self.vertical_fov_rad = None


    def _initialize_camera(self):
        """Initializes the camera based on the detected platform."""
        if self.is_raspberry_pi:
            try:
                self.picam2 = Picamera2()
                # Use preview config for potentially lower latency, still for consistency? Let's use preview.
                config = self.picam2.create_preview_configuration(
                     main={"size": self.camera_resolution, "format": "RGB888"} # Picamera2 uses RGB
                )
                self.picam2.configure(config)
                # Optional: Add autofocus controls if needed
                # self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "AfSpeed": controls.AfSpeedEnum.Fast})
                self.picam2.start()
                print(f"Picamera2 initialized requesting resolution {self.camera_resolution}")
                time.sleep(2.0) # Allow camera to warm up
            except Exception as e:
                print(f"ERROR: Failed to initialize Picamera2: {e}. Attempting OpenCV fallback.")
                self.picam2 = None # Ensure it's None if init fails
                self.is_raspberry_pi = False # Force fallback
                self._initialize_opencv_camera() # Try OpenCV
        else: # Not on Raspberry Pi or Picamera2 failed
             self._initialize_opencv_camera()

    def _initialize_opencv_camera(self):
        """Initializes camera using OpenCV VideoCapture."""
        try:
            self.camera = cv2.VideoCapture(0) # Use camera index 0
            if not self.camera.isOpened():
                 raise IOError("Cannot open default OpenCV camera.")

            # Set desired resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])

            # Verify the resolution was set (some cameras ignore requests)
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"OpenCV camera initialized. Requested resolution {self.camera_resolution}, Actual resolution: {actual_width}x{actual_height}")

        except Exception as e:
            print(f"ERROR: Failed to initialize OpenCV camera: {e}")
            self.camera = None

    def _get_actual_camera_resolution(self):
         """ Tries to get the currently configured resolution from the active camera object. """
         if self.is_raspberry_pi and self.picam2:
              try:
                   # Picamera2 stores config info; accessing 'size' from the 'main' stream config
                   sensor_modes = self.picam2.sensor_modes
                   current_mode = self.picam2.camera_configuration()['main']
                   return current_mode['size'] # Returns (width, height) tuple
              except Exception as e:
                   print(f"Could not get Picamera2 actual resolution: {e}")
                   return None, None
         elif self.camera and self.camera.isOpened():
              try:
                   width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                   height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                   return width, height
              except Exception as e:
                   print(f"Could not get OpenCV actual resolution: {e}")
                   return None, None
         else:
              return None, None


    def capture_frame(self):
        """Captures a frame from the initialized camera and applies undistortion."""
        frame = None
        success = False
        if self.is_raspberry_pi and self.picam2:
            # Picamera2 capture_array gives RGB, convert to BGR for OpenCV
            frame_rgb = self.picam2.capture_array("main") # Capture from the 'main' stream
            if frame_rgb is not None:
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                success = True
        elif self.camera:
            success, frame = self.camera.read()

        if not success or frame is None:
            # print("ERROR: Failed to capture frame.") # Avoid spamming console
            return None

        # --- Apply Undistortion ---
        if self.camera_matrix is not None and self.dist_coeffs is not None:
             # Using the same camera matrix for newCameraMatrix is common for basic undistortion
             # For optimal results, especially with cropping, compute new matrix once:
             # h, w = frame.shape[:2]
             # new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), alpha=1, newImgSize=(w,h)) # alpha=1 keeps all pixels
             # frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_camera_mtx)
             # x, y, w_roi, h_roi = roi
             # frame = frame[y:y+h_roi, x:x+w_roi] # Crop if using roi
             # --- Simplified undistortion ---
             frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)


        return frame

    def _calculate_angles(self, marker_center_px, frame_resolution):
        """
        Calculates angular offset (radians) from image center to marker center.
        Uses the FOV calculated during initialization.
        """
        # Use self.horizontal_fov_rad and self.vertical_fov_rad calculated in __init__
        if marker_center_px is None or frame_resolution is None or \
           self.horizontal_fov_rad is None or self.vertical_fov_rad is None:
            # print("Cannot calculate angles: Missing inputs or FOV.") # Avoid spam
            return None

        mc_x, mc_y = marker_center_px
        res_w, res_h = frame_resolution
        fov_h = self.horizontal_fov_rad # Use calculated value
        fov_v = self.vertical_fov_rad   # Use calculated value

        # Calculate center of the frame
        frame_center_x = res_w * 0.5
        frame_center_y = res_h * 0.5

        # Calculate angular offset in radians
        # Positive X angle means target is to the right of center
        # Positive Y angle means target is below center (as Y increases downwards in image coords)
        x_angle = (mc_x - frame_center_x) * (fov_h / res_w)
        y_angle = (mc_y - frame_center_y) * (fov_v / res_h)

        return x_angle, y_angle

    def _get_aruco_position(self, frame):
        """
        Detects the predefined ArUco marker and returns its center pixel coordinates.

        Args:
            frame: The input camera frame (BGR).

        Returns:
            tuple: (x_center, y_center) pixel coordinates or None if not found.
        """
        if frame is None: return None

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            image=gray_img,
            dictionary=self.aruco_dict,
            parameters=self.parameters
        )

        marker_center = None
        if ids is not None:
            for i, marker_id in enumerate(ids):
                if marker_id[0] == self.ARUCO_ID_TO_FIND:
                    marker_corners = corners[i][0] # Get the corners for this marker
                    # Calculate the center by averaging corner coordinates
                    x_center = marker_corners[:, 0].mean()
                    y_center = marker_corners[:, 1].mean()
                    marker_center = (x_center, y_center)
                    break # Found the desired marker, exit loop

        return marker_center

    def _get_qr_position(self, frame):
        """
        Detects a QR code and returns its center pixel coordinates. Uses detectMulti for robustness.

        Args:
            frame: The input camera frame (BGR).

        Returns:
            tuple: (x_center, y_center) pixel coordinates or None if not found.
        """
        if frame is None: return None

        marker_center = None
        try:
            # Use detectMulti, but we only care about the first detection's points
            # We don't decode here as it's slower and not needed for positioning
            found, points = self.qr_detector.detectMulti(frame)

            if found and points is not None and len(points) > 0:
                 # Use the first detected QR code's points
                 qr_points = points[0]
                 # Calculate the center by averaging corner coordinates
                 x_center = qr_points[:, 0].mean()
                 y_center = qr_points[:, 1].mean()
                 marker_center = (x_center, y_center)

        except Exception as e:
            # Some OpenCV versions might throw errors on certain inputs
            # print(f"Error during QR code detection: {e}") # Avoid spam
            pass

        return marker_center

    def get_lading_target_vals(self, frame, current_altitude):
        """
        Processes a frame to find the landing target (ArUco or QR) based on altitude
        and returns the calculated angular offsets for the LANDING_TARGET MAVLink message.

        Args:
            frame: The input camera frame (BGR).
            current_altitude (float): The drone's current altitude in meters.

        Returns:
            tuple: (x_angle_rad, y_angle_rad) or None if no target is found.
        """
        self._last_detected_center = None # Reset visualization marker each call
        if frame is None:
            # print("Received None frame.")
            return None
        if current_altitude is None:
             # print("Received None altitude.")
             return None

        marker_center_px = None

        # Decide which marker type to look for
        if current_altitude > self.TRANSITION_ALTITUDE:
            # print(f"Alt {current_altitude:.1f}m > {self.TRANSITION_ALTITUDE:.1f}m: Look ArUco") # Debug print
            marker_center_px = self._get_aruco_position(frame)
        else:
            # print(f"Alt {current_altitude:.1f}m <= {self.TRANSITION_ALTITUDE:.1f}m: Look QR") # Debug print
            marker_center_px = self._get_qr_position(frame)

        # Calculate angles if a marker was found
        if marker_center_px is not None:
            self._last_detected_center = marker_center_px # Store for visualization
            # Pass the actual frame resolution used at runtime
            runtime_resolution = frame.shape[1], frame.shape[0]
            angles_rad = self._calculate_angles(
                marker_center_px,
                runtime_resolution # Use actual resolution of the input frame
            )
            return angles_rad
        else:
            # No target found
            return None

    def visualize_target(self, frame):
        """ Helper to draw detection info onto the frame for display/debugging. """
        if frame is None: return

        # Use the current runtime camera resolution for drawing center lines
        res_w, res_h = self.camera_resolution

        # Draw crosshairs at image center
        center_x = int(res_w / 2)
        center_y = int(res_h / 2)
        cv2.line(frame, (center_x, 0), (center_x, res_h), (0, 0, 255), 1)
        cv2.line(frame, (0, center_y), (res_w, center_y), (0, 0, 255), 1)

        if self._last_detected_center:
            # Draw circle at detected target center
            cv2.circle(frame, (int(self._last_detected_center[0]), int(self._last_detected_center[1])), 7, (0, 255, 0), -1)
            cv2.putText(frame, "Target Found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
             cv2.putText(frame, "Target NOT Found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Landing Target View", frame)
        # Key press handling should happen in the main loop using this script,
        # but we need waitKey(1) to allow the window to refresh.
        cv2.waitKey(1)


    def release_camera(self):
        """Releases camera resources."""
        print("Releasing camera resources...")
        if self.is_raspberry_pi and self.picam2:
            try:
                self.picam2.stop()
                print("Picamera2 stopped.")
            except Exception as e:
                print(f"Error stopping Picamera2: {e}")
        elif self.camera:
            try:
                self.camera.release()
                print("OpenCV camera released.")
            except Exception as e:
                print(f"Error releasing OpenCV camera: {e}")
        cv2.destroyAllWindows() # Close any OpenCV windows opened by this class


# --- Example Usage (for testing this module standalone) ---
if __name__ == '__main__':
    print("Running LandingTarget standalone test...")
    landing_target = None # Initialize to None for finally block
    try:
        # Assumes calibration file is present in the same directory
        landing_target = LandingTarget()

        # Simulate altitudes for testing switching
        test_altitudes = [10.0, 8.0, 6.0, 4.0, 2.0] # Meters
        alt_idx = 0

        print("\nStarting capture loop. Press 'n' to cycle altitude, 'q' to quit.")

        while True:
            current_alt = test_altitudes[alt_idx % len(test_altitudes)]
            print(f"\nSimulating Altitude: {current_alt:.1f}m")

            frame = landing_target.capture_frame()
            if frame is None:
                print("Failed to get frame, trying again...")
                time.sleep(0.5) # Wait a bit before retrying
                continue

            # --- Process frame ---
            landing_vals = landing_target.get_lading_target_vals(frame, current_alt)

            # --- Output results ---
            if landing_vals:
                print(f"  Target Found! Angles (rad): X={landing_vals[0]:.4f}, Y={landing_vals[1]:.4f}")
                print(f"                Angles (deg): X={math.degrees(landing_vals[0]):.2f}, Y={math.degrees(landing_vals[1]):.2f}")
            else:
                print("  Target not found.")

            # --- Visualization ---
            landing_target.visualize_target(frame) # visualize_target now calls waitKey(1)

            # --- Non-blocking key check alternative ---
            # key = cv2.waitKey(1) & 0xFF
            # Need a longer wait to make key presses responsive in standalone test
            key = cv2.waitKey(100) & 0xFF # Wait 100ms

            if key == ord('q'):
                print("'q' pressed, quitting.")
                break
            elif key == ord('n'): # Next altitude
                 alt_idx += 1
                 print("Switching to next simulated altitude.")


    except IOError as e:
         print(f"Initialization Error: {e}. Please ensure calibration file exists.")
    except RuntimeError as e:
         print(f"Initialization Error: {e}. Please ensure camera is connected and accessible.")
    except KeyboardInterrupt:
         print("\nKeyboardInterrupt detected, exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if landing_target: # Check if landing_target was successfully initialized
            landing_target.release_camera()
        print("Standalone test finished.")
