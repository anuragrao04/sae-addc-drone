#!/usr/bin/env python3
import numpy as np
import cv2
import glob
import pickle
import os
import time
import socket # Added for platform detection

# --- Platform Detection ---
# Check hostname to determine if running on Raspberry Pi
try:
    IS_RASPBERRY_PI = socket.gethostname() == "raspberrypi"
except:
    IS_RASPBERRY_PI = False # Assume not Pi if hostname check fails

# --- Conditional Import for PiCamera ---
if IS_RASPBERRY_PI:
    try:
        from picamera2 import Picamera2
        from libcamera import controls
        print("Platform: Raspberry Pi (Picamera2 available)")
    except ImportError:
        print("Platform: Raspberry Pi (Picamera2 library not found, install it for full functionality)")
        IS_RASPBERRY_PI = False # Treat as non-Pi if import fails
else:
    print("Platform: Non-Raspberry Pi (using OpenCV VideoCapture)")
    # Define dummy classes if not on Pi and import failed, so functions don't crash
    class Picamera2: pass
    class controls: AfModeEnum = type('Enum', (), {'Continuous': 0})()

# ==================================
# CAMERA HANDLING FUNCTIONS
# ==================================

def initialize_camera(resolution=(1280, 720)):
    """
    Initializes the appropriate camera based on the platform.

    Args:
        resolution (tuple): Desired camera resolution (width, height).

    Returns:
        tuple: (camera_object, is_pi) or (None, False) if initialization fails.
               camera_object is either a Picamera2 instance or cv2.VideoCapture instance.
    """
    if IS_RASPBERRY_PI:
        try:
            print(f"Initializing Picamera2 with resolution {resolution}...")
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                 main={"size": resolution, "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            print("Picamera2 started. Allowing time to settle...")
            time.sleep(2.0)
            return picam2, True
        except Exception as e:
            print(f"ERROR: Failed to initialize Picamera2: {e}. Trying OpenCV fallback...")
            # Fallback to OpenCV attempt below
            pass # Let it fall through to the OpenCV section if PiCam fails

    # If not Pi OR if Picamera2 failed
    try:
        print(f"Initializing OpenCV VideoCapture(0) requesting resolution {resolution}...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open default OpenCV camera.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"OpenCV camera opened. Actual resolution: {actual_width}x{actual_height}")
        time.sleep(0.5)
        return cap, False # Return False for is_pi when using OpenCV
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenCV camera: {e}")
        return None, False

def capture_single_frame(camera_obj, is_pi):
    """
    Captures a single frame from the initialized camera object.

    Args:
        camera_obj: The initialized camera object (Picamera2 or VideoCapture).
        is_pi (bool): Flag indicating if the camera object is Picamera2.

    Returns:
        numpy.ndarray: The captured frame in BGR format, or None if capture fails.
    """
    frame = None
    success = False
    if camera_obj is None:
         print("Error: Camera object is None.")
         return None

    if is_pi:
        try:
            frame_rgb = camera_obj.capture_array("main")
            if frame_rgb is not None:
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                success = True
        except Exception as e:
             print(f"Error capturing frame with Picamera2: {e}")
             success = False
    else: # OpenCV VideoCapture
        try:
            success, frame = camera_obj.read()
        except Exception as e:
             print(f"Error capturing frame with OpenCV VideoCapture: {e}")
             success = False

    if not success or frame is None:
        # print("ERROR: Failed to capture frame.") # Avoid spamming
        return None

    return frame

def release_camera(camera_obj, is_pi):
    """Releases the camera resources."""
    if camera_obj is None:
        return
    print("Releasing camera resources...")
    if is_pi:
        try:
            camera_obj.stop()
            print("Picamera2 stopped.")
        except Exception as e:
            print(f"Error stopping Picamera2: {e}")
    else:
        try:
            camera_obj.release()
            print("OpenCV camera released.")
        except Exception as e:
            print(f"Error releasing OpenCV camera: {e}")

# ==================================
# CALIBRATION FUNCTIONS (capture_calibration_images is MODIFIED)
# ==================================

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

# --- MODIFIED FUNCTION ---
def capture_calibration_images(num_images=20, resolution=(1280, 720), save_dir='calibration_images'):
    """
    Capture images of a chessboard pattern for camera calibration using
    the appropriate camera for the platform (Picamera2 or OpenCV).

    Args:
        num_images (int): Number of calibration images to capture
        resolution (tuple): Camera resolution (width, height)
        save_dir (str): Directory to save calibration images
    """
    # Create directory for calibration images
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory {save_dir}")

    # Initialize the appropriate camera for the platform
    camera_obj, is_pi = initialize_camera(resolution)
    if camera_obj is None:
        print("Failed to initialize camera for calibration capture.")
        input("\nPress Enter to return to menu...")
        return False

    print(f"Using {'Picamera2' if is_pi else 'OpenCV VideoCapture'} for image capture.")

    img_count = 0
    print("\nChessboard Calibration Image Capture")
    print("====================================")
    print("Instructions:")
    print("1. Hold a chessboard pattern in front of the camera")
    print("2. Move it to different positions and orientations")
    print("3. Press 'c' to capture an image when the board is steady")
    print("4. Press 'q' to quit after capturing enough images")
    print(f"Goal: Capture {num_images} clear images of the chessboard\n")

    window_name = 'Calibration Capture'
    cv2.namedWindow(window_name)

    while img_count < num_images:
        # Capture frame using the generalized function
        frame_bgr = capture_single_frame(camera_obj, is_pi)

        if frame_bgr is None:
             print("Error getting frame during calibration capture. Retrying...")
             time.sleep(0.1) # Brief pause before retry
             continue

        # Create a copy for display
        display = frame_bgr.copy()

        # Display instruction on frame
        cv2.putText(display, f"Captured: {img_count}/{num_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, "Press 'c' to capture", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, "Press 'q' to quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display frame
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF # waitKey is essential for imshow to work
        if key == ord('q'):
            print("Quit key pressed.")
            break
        elif key == ord('c'):
            # Save image
            img_path = f'{save_dir}/calib_{img_count:02d}.jpg'
            try:
                cv2.imwrite(img_path, frame_bgr)
                print(f"Saved {img_path}")
                img_count += 1
            except Exception as e:
                 print(f"Error saving image {img_path}: {e}")
            # Wait a moment to avoid duplicate captures
            time.sleep(0.5)

    # Clean up
    release_camera(camera_obj, is_pi)
    cv2.destroyAllWindows()
    print(f"\nCaptured {img_count} images for calibration")

    # Wait for user to press enter before returning to menu
    input("\nPress Enter to return to menu...")

    return img_count > 0
# --- END OF MODIFIED FUNCTION ---


def calibrate_camera(chessboard_size=(9, 6), square_size=2.5, save_dir='calibration_images', output_file='camera_calibration.pkl'):
    """
    Calibrate camera using chessboard images. (No camera interaction needed here)

    Args:
        chessboard_size (tuple): Size of chessboard in inner corners (width, height)
        square_size (float): Size of each chessboard square in cm or other unit
        save_dir (str): Directory containing calibration images
        output_file (str): Where to save the calibration data

    Returns:
        dict: Camera calibration parameters, or None if calibration failed
    """
    print("\nStarting Camera Calibration Calculation")
    print("====================================")

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    objpoints = []
    imgpoints = []
    images = glob.glob(f'{save_dir}/*.jpg')

    if not images:
        print(f"❌ No calibration images found in '{save_dir}' directory.")
        input("\nPress Enter to return to menu...")
        return None

    print(f"Found {len(images)} calibration images to process.")
    img_size = None
    successful_detections = 0

    for i, fname in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}: {os.path.basename(fname)}... ", end='')
        img = cv2.imread(fname)
        if img is None:
            print(f"✗ Failed to read image {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            successful_detections += 1
            print("✓ Chessboard detected")
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
        else:
            print("✗ No chessboard detected")

    if not objpoints:
        print("❌ No chessboard patterns were found in any of the images.")
        input("\nPress Enter to return to menu...")
        return None

    print(f"\nSuccessfully detected chessboard in {successful_detections} out of {len(images)} images.")

    if img_size is None:
        print("❌ Could not determine image size from processed images.")
        input("\nPress Enter to return to menu...")
        return None

    print("\nCalculating camera parameters... ", end='')
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    print("done!")

    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'img_size': img_size
    }
    with open(output_file, 'wb') as f:
        pickle.dump(calibration_data, f)

    print(f"\n✅ Calibration complete! Data saved to {output_file}")
    print("\nCamera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)
    print(f"\nImage Size used: {img_size}")

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    reprojection_error = mean_error / len(objpoints)
    print(f"\nReprojection error: {reprojection_error:.6f} pixels")

    if reprojection_error < 0.5: print("Excellent calibration! (Error < 0.5 pixels)")
    elif reprojection_error < 1.0: print("Good calibration. (Error < 1.0 pixels)")
    else: print("Calibration could be improved.")

    if len(images) > 0:
        test_img = cv2.imread(images[0])
        if test_img is not None:
            h, w = test_img.shape[:2]
            cal_w, cal_h = img_size
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (cal_w, cal_h), 1, (cal_w, cal_h))
            dst = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, newcameramtx)
            x, y, roi_w, roi_h = roi
            dst_cropped = dst[y:y+roi_h, x:x+roi_w]
            undistorted_path = 'calibration_test_undistorted.jpg'
            cv2.imwrite(undistorted_path, dst_cropped)
            print(f"\nSaved undistorted test image (cropped to ROI) to {undistorted_path}")
            original_resized = cv2.resize(test_img, (roi_w, roi_h))
            comparison = np.hstack((original_resized, dst_cropped))
            comparison_path = 'calibration_comparison.jpg'
            cv2.imwrite(comparison_path, comparison)
            print(f"Saved side-by-side comparison to {comparison_path}")

    input("\nPress Enter to return to menu...")
    return calibration_data


def test_calibration(calibration_file='camera_calibration.pkl', resolution=(1280, 720)):
    """
    Test camera calibration by displaying a live feed with distortion correction.
    Uses the appropriate camera for the platform.
    """
    try:
        with open(calibration_file, 'rb') as f:
            calibration_data = pickle.load(f)
        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs = calibration_data['dist_coeffs']
        calib_img_size = calibration_data.get('img_size')
        print(f"Loaded calibration data from {calibration_file}")
        if calib_img_size: print("Calibration Image Size:", calib_img_size)
    except Exception as e:
        print(f"Error loading calibration file {calibration_file}: {e}")
        input("\nPress Enter to return to menu...")
        return

    camera_obj, is_pi = initialize_camera(resolution)
    if camera_obj is None:
        print("Failed to initialize camera for testing.")
        input("\nPress Enter to return to menu...")
        return

    actual_width, actual_height = resolution
    if is_pi:
        try: actual_width, actual_height = camera_obj.camera_configuration()['main']['size']
        except: pass
    else:
        try:
             actual_width = int(camera_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
             actual_height = int(camera_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except: pass
    actual_resolution = (actual_width, actual_height)
    print(f"Testing calibration with camera running at {actual_resolution}")

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, actual_resolution, 1, actual_resolution)
    x_roi, y_roi, w_roi, h_roi = roi

    print("Camera started. Press 'q' to quit.")
    window_name = 'Calibration Test'
    cv2.namedWindow(window_name)

    while True:
        frame_bgr = capture_single_frame(camera_obj, is_pi)
        if frame_bgr is None:
            time.sleep(0.1)
            continue

        undistorted = cv2.undistort(frame_bgr, camera_matrix, dist_coeffs, None, newcameramtx)
        undistorted_cropped = undistorted[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        original_resized = cv2.resize(frame_bgr, (w_roi, h_roi))
        comparison = np.hstack((original_resized, undistorted_cropped))
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(comparison, "Undistorted (Cropped)", (w_roi + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow(window_name, comparison)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_camera(camera_obj, is_pi)
    cv2.destroyAllWindows()
    input("\nPress Enter to return to menu...")


# ==================================
# MENU FUNCTIONS
# ==================================

def run_all_steps():
    """Run all steps: capture, calibrate, and test in sequence."""
    clear_screen()
    print("Camera Calibration Complete Pipeline")
    print("===================================")
    # Parameter gathering...
    resolution_choice = input("Enter camera resolution (width,height) [1280,720]: ").strip() or "1280,720"
    try: resolution = tuple(map(int, resolution_choice.split(',')))
    except: resolution = (1280, 720); print("Using default resolution.")

    num_images_str = input("Enter number of images to capture [20]: ").strip() or "20"
    try: num_images = int(num_images_str)
    except: num_images = 20; print("Using default number of images.")

    chessboard_choice = input("Enter chessboard size (width,height inner corners) [9,6]: ").strip() or "9,6"
    try: chessboard_size = tuple(map(int, chessboard_choice.split(',')))
    except: chessboard_size = (9, 6); print("Using default chessboard size.")

    square_size_str = input("Enter chessboard square size in cm (or units) [2.5]: ").strip() or "2.5"
    try: square_size = float(square_size_str)
    except: square_size = 2.5; print("Using default square size.")

    save_dir = input("Enter directory for calibration images [calibration_images]: ").strip() or "calibration_images"
    output_file = input("Enter output file for calibration data [camera_calibration.pkl]: ").strip() or "camera_calibration.pkl"

    # Step 1: Capture images (Works on Pi and non-Pi now)
    print("\nStep 1: Capturing calibration images")
    captured_ok = capture_calibration_images(num_images, resolution, save_dir)

    if captured_ok:
        # Step 2: Calibrate camera
        print("\nStep 2: Calibrating camera")
        calibration_data = calibrate_camera(chessboard_size, square_size, save_dir, output_file)

        if calibration_data is not None:
            # Step 3: Test calibration
            print("\nStep 3: Testing calibration")
            test_calibration(output_file, resolution)

    print("\nComplete pipeline finished!")
    input("\nPress Enter to return to menu...")


def get_menu_choice():
    """Display the main menu and get the user's choice."""
    clear_screen()
    print("=================================")
    print("  Camera Calibration Tool")
    print(f"  Platform: {'Raspberry Pi' if IS_RASPBERRY_PI else 'Other/Laptop'}")
    print("=================================")
    print("1. Capture Calibration Images") # Now works on both
    print("2. Calibrate Camera (Uses saved images)")
    print("3. Test Calibration (Live feed)")
    print("4. Run Complete Pipeline (1->2->3)")
    print("5. Test Single Frame Capture")
    print("6. Exit")
    print("=================================")
    while True:
        choice = input("Enter your choice (1-6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']: return int(choice)
        else: print("Invalid choice.")

def capture_menu():
    """Menu for capturing calibration images."""
    clear_screen()
    print("Capture Calibration Images")
    print("==========================")
    # Parameter gathering...
    resolution_choice = input("Enter camera resolution (width,height) [1280,720]: ").strip() or "1280,720"
    try: resolution = tuple(map(int, resolution_choice.split(',')))
    except: resolution = (1280, 720); print("Using default resolution.")
    num_images_str = input("Enter number of images to capture [20]: ").strip() or "20"
    try: num_images = int(num_images_str)
    except: num_images = 20; print("Using default number of images.")
    save_dir = input("Enter directory for calibration images [calibration_images]: ").strip() or "calibration_images"
    capture_calibration_images(num_images, resolution, save_dir)

def calibrate_menu():
    """Menu for calibrating the camera."""
    clear_screen()
    print("Calibrate Camera")
    print("================")
    # Parameter gathering...
    chessboard_choice = input("Enter chessboard size (width,height inner corners) [9,6]: ").strip() or "9,6"
    try: chessboard_size = tuple(map(int, chessboard_choice.split(',')))
    except: chessboard_size = (9, 6); print("Using default chessboard size.")
    square_size_str = input("Enter chessboard square size in cm (or units) [2.5]: ").strip() or "2.5"
    try: square_size = float(square_size_str)
    except: square_size = 2.5; print("Using default square size.")
    save_dir = input("Enter directory with calibration images [calibration_images]: ").strip() or "calibration_images"
    output_file = input("Enter output file for calibration data [camera_calibration.pkl]: ").strip() or "camera_calibration.pkl"
    calibrate_camera(chessboard_size, square_size, save_dir, output_file)


def test_menu():
    """Menu for testing calibration."""
    clear_screen()
    print("Test Calibration")
    print("===============")
    # Parameter gathering...
    calibration_file = input("Enter path to calibration file [camera_calibration.pkl]: ").strip() or "camera_calibration.pkl"
    resolution_choice = input("Enter camera resolution (width,height) [1280,720]: ").strip() or "1280,720"
    try: resolution = tuple(map(int, resolution_choice.split(',')))
    except: resolution = (1280, 720); print("Using default resolution.")
    test_calibration(calibration_file, resolution)

def capture_single_frame_test_menu():
     """Menu option to test capturing a single frame."""
     clear_screen()
     print("Test Single Frame Capture")
     print("=========================")
     resolution_choice = input("Enter camera resolution (width,height) [1280,720]: ").strip() or "1280,720"
     try: resolution = tuple(map(int, resolution_choice.split(',')))
     except: resolution = (1280, 720); print("Using default resolution.")

     camera_obj, is_pi = initialize_camera(resolution)
     if camera_obj is None:
          print("Failed to initialize camera.")
     else:
          print("Attempting to capture a single frame...")
          frame = capture_single_frame(camera_obj, is_pi)
          release_camera(camera_obj, is_pi) # Release camera after capture
          if frame is not None:
               print(f"Frame captured successfully! Shape: {frame.shape}")
               save_path = "single_frame_capture_test.jpg"
               try:
                    cv2.imwrite(save_path, frame)
                    print(f"Frame saved to {save_path}")
                    cv2.imshow("Single Frame Capture Test", frame)
                    print("Displaying captured frame. Press any key to close.")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
               except Exception as e:
                    print(f"Error saving or displaying frame: {e}")
          else:
               print("Failed to capture frame.")
     input("\nPress Enter to return to menu...")


def main():
    """Main function for the camera calibration tool."""
    while True:
        choice = get_menu_choice()
        if choice == 1: capture_menu()
        elif choice == 2: calibrate_menu()
        elif choice == 3: test_menu()
        elif choice == 4: run_all_steps()
        elif choice == 5: capture_single_frame_test_menu()
        elif choice == 6: print("Exiting camera calibration tool. Goodbye!"); break

if __name__ == "__main__":
    main()
