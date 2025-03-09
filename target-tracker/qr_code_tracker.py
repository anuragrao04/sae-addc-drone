#!/usr/bin/env python3
import numpy as np
import cv2
import glob
import pickle
import os
import time
from picamera2 import Picamera2
from libcamera import controls

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def capture_calibration_images(num_images=20, resolution=(1280, 720), save_dir='calibration_images'):
    """
    Capture images of a chessboard pattern for camera calibration using Picamera2.
    
    Args:
        num_images (int): Number of calibration images to capture
        resolution (tuple): Camera resolution (width, height)
        save_dir (str): Directory to save calibration images
    """
    # Create directory for calibration images
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory {save_dir}")
    
    # Initialize Picamera2
    picam2 = Picamera2()
    
    # Configure camera
    config = picam2.create_still_configuration(
        main={"size": resolution, "format": "RGB888"}
    )
    picam2.configure(config)
    
    # Set autofocus mode to continuous
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    
    # Start camera
    picam2.start()
    print("Camera initialized. Allow a moment for it to adjust...")
    time.sleep(2)  # Give camera time to initialize
    
    img_count = 0
    print("\nChessboard Calibration Image Capture")
    print("====================================")
    print("Instructions:")
    print("1. Hold a chessboard pattern in front of the camera")
    print("2. Move it to different positions and orientations")
    print("3. Press 'c' to capture an image when the board is steady")
    print("4. Press 'q' to quit after capturing enough images")
    print(f"Goal: Capture {num_images} clear images of the chessboard\n")
    
    while img_count < num_images:
        # Capture frame
        frame = picam2.capture_array()
        
        # Convert from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
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
        cv2.imshow('Calibration', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save image
            img_path = f'{save_dir}/calib_{img_count:02d}.jpg'
            cv2.imwrite(img_path, frame_bgr)
            print(f"Saved {img_path}")
            img_count += 1
            # Wait a moment to avoid duplicate captures
            time.sleep(0.5)
    
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
    print(f"\nCaptured {img_count} images for calibration")
    
    # Wait for user to press enter before returning to menu
    input("\nPress Enter to return to menu...")
    
    return img_count > 0

def calibrate_camera(chessboard_size=(9, 6), square_size=2.5, save_dir='calibration_images', output_file='camera_calibration.pkl'):
    """
    Calibrate camera using chessboard images.
    
    Args:
        chessboard_size (tuple): Size of chessboard in inner corners (width, height)
        square_size (float): Size of each chessboard square in cm
        save_dir (str): Directory containing calibration images
        output_file (str): Where to save the calibration data
        
    Returns:
        dict: Camera calibration parameters, or None if calibration failed
    """
    print("\nStarting Camera Calibration")
    print("=========================")
    
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Path to calibration images
    images = glob.glob(f'{save_dir}/*.jpg')
    
    if not images:
        print(f"❌ No calibration images found in '{save_dir}' directory.")
        input("\nPress Enter to return to menu...")
        return None
    
    print(f"Found {len(images)} calibration images to process.")
    
    # Size for the first image
    img_size = None
    
    # Counter for successful chessboard detections
    successful_detections = 0
    
    for i, fname in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}: {os.path.basename(fname)}... ", end='')
        
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # If found, add object points and image points
        if ret:
            successful_detections += 1
            print("✓ Chessboard detected")
            objpoints.append(objp)
            
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(200)  # Show each image briefly
        else:
            print("✗ No chessboard detected")
    
    cv2.destroyAllWindows()
    
    if not objpoints:
        print("❌ No chessboard patterns were found in any of the images.")
        print("Try again with different images or lighting conditions.")
        input("\nPress Enter to return to menu...")
        return None
    
    print(f"\nSuccessfully detected chessboard in {successful_detections} out of {len(images)} images.")
    
    # Calibrate camera
    print("\nCalculating camera parameters... ", end='')
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    print("done!")
    
    # Save calibration results
    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'img_size': img_size
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(calibration_data, f)
    
    print(f"\n✅ Calibration complete! Data saved to {output_file}")
    print("\nCamera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    reprojection_error = mean_error/len(objpoints)
    print(f"\nReprojection error: {reprojection_error:.6f} pixels")
    
    if reprojection_error < 0.5:
        print("Excellent calibration! (Error < 0.5 pixels)")
    elif reprojection_error < 1.0:
        print("Good calibration. (Error < 1.0 pixels)")
    else:
        print("Calibration could be improved. Consider taking more images or using a better chessboard pattern.")
    
    # Let's also generate an undistorted test image to show the correction
    if len(images) > 0:
        test_img = cv2.imread(images[0])
        h, w = test_img.shape[:2]
        
        # Get optimal new camera matrix for undistortion
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        
        # Undistort
        dst = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, newcameramtx)
        
        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # Save the undistorted image
        undistorted_path = 'calibration_test_undistorted.jpg'
        cv2.imwrite(undistorted_path, dst)
        print(f"\nSaved undistorted test image to {undistorted_path}")
        
        # Save side-by-side comparison
        original = cv2.imread(images[0])
        original = cv2.resize(original, (w, h))
        comparison = np.hstack((original, dst))
        comparison_path = 'calibration_comparison.jpg'
        cv2.imwrite(comparison_path, comparison)
        print(f"Saved side-by-side comparison to {comparison_path}")
    
    # Wait for user to press enter before returning to menu
    input("\nPress Enter to return to menu...")
    
    return calibration_data

def test_calibration(calibration_file='camera_calibration.pkl', resolution=(1280, 720)):
    """
    Test camera calibration by displaying a live feed with distortion correction.
    
    Args:
        calibration_file (str): Path to the calibration data file
        resolution (tuple): Camera resolution (width, height)
    """
    try:
        with open(calibration_file, 'rb') as f:
            calibration_data = pickle.load(f)
        
        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs = calibration_data['dist_coeffs']
        
        print(f"Loaded calibration data from {calibration_file}")
        print("Camera Matrix:")
        print(camera_matrix)
        print("Distortion Coefficients:")
        print(dist_coeffs)
    except FileNotFoundError:
        print(f"Calibration file {calibration_file} not found.")
        input("\nPress Enter to return to menu...")
        return
    
    # Initialize Picamera2
    picam2 = Picamera2()
    
    # Configure camera
    config = picam2.create_still_configuration(
        main={"size": resolution, "format": "RGB888"}
    )
    picam2.configure(config)
    
    # Start camera
    picam2.start()
    print("Camera started. Press 'q' to quit.")
    time.sleep(2)  # Give camera time to initialize
    
    w, h = resolution
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    while True:
        # Capture frame
        frame = picam2.capture_array()
        
        # Convert from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Undistort
        undistorted = cv2.undistort(frame_bgr, camera_matrix, dist_coeffs, None, newcameramtx)
        
        # Crop the image
        x, y, w, h = roi
        undistorted_cropped = undistorted[y:y+h, x:x+w]
        
        # Resize original to match cropped dimensions
        original_resized = cv2.resize(frame_bgr, (w, h))
        
        # Create side-by-side comparison
        comparison = np.hstack((original_resized, undistorted_cropped))
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(comparison, "Undistorted", (w + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Display the result
        cv2.imshow('Calibration Test', comparison)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
    
    # Wait for user to press enter before returning to menu
    input("\nPress Enter to return to menu...")

def run_all_steps():
    """Run all steps: capture, calibrate, and test in sequence."""
    clear_screen()
    print("Camera Calibration Complete Pipeline")
    print("===================================")
    
    # Get camera resolution
    print("\nCamera Resolution:")
    resolution_choice = input("Enter camera resolution (width,height) [1280,720]: ").strip()
    if resolution_choice:
        try:
            width, height = map(int, resolution_choice.split(','))
            resolution = (width, height)
        except ValueError:
            print("Invalid resolution format. Using default 1280x720.")
            resolution = (1280, 720)
    else:
        resolution = (1280, 720)
    
    # Get number of images
    num_images = input("Enter number of images to capture [20]: ").strip()
    if num_images:
        try:
            num_images = int(num_images)
        except ValueError:
            print("Invalid number. Using default 20 images.")
            num_images = 20
    else:
        num_images = 20
    
    # Get chessboard size
    chessboard_choice = input("Enter chessboard size (width,height inner corners) [9,6]: ").strip()
    if chessboard_choice:
        try:
            width, height = map(int, chessboard_choice.split(','))
            chessboard_size = (width, height)
        except ValueError:
            print("Invalid format. Using default 9x6.")
            chessboard_size = (9, 6)
    else:
        chessboard_size = (9, 6)
    
    # Get square size
    square_size = input("Enter chessboard square size in cm [2.5]: ").strip()
    if square_size:
        try:
            square_size = float(square_size)
        except ValueError:
            print("Invalid number. Using default 2.5cm.")
            square_size = 2.5
    else:
        square_size = 2.5
    
    # Get save directory
    save_dir = input("Enter directory for calibration images [calibration_images]: ").strip()
    if not save_dir:
        save_dir = 'calibration_images'
    
    # Get output file
    output_file = input("Enter output file for calibration data [camera_calibration.pkl]: ").strip()
    if not output_file:
        output_file = 'camera_calibration.pkl'
    
    # Step 1: Capture images
    print("\nStep 1: Capturing calibration images")
    success = capture_calibration_images(
        num_images=num_images,
        resolution=resolution,
        save_dir=save_dir
    )
    
    if success:
        # Step 2: Calibrate camera
        print("\nStep 2: Calibrating camera")
        calibration_data = calibrate_camera(
            chessboard_size=chessboard_size,
            square_size=square_size,
            save_dir=save_dir,
            output_file=output_file
        )
        
        if calibration_data is not None:
            # Step 3: Test calibration
            print("\nStep 3: Testing calibration")
            test_calibration(
                calibration_file=output_file,
                resolution=resolution
            )
    
    print("\nComplete pipeline finished!")

def get_menu_choice():
    """Display the main menu and get the user's choice."""
    clear_screen()
    print("=================================")
    print("  Camera Calibration Tool")
    print("=================================")
    print("1. Capture Calibration Images")
    print("2. Calibrate Camera")
    print("3. Test Calibration")
    print("4. Run Complete Pipeline")
    print("5. Exit")
    print("=================================")
    
    while True:
        choice = input("Enter your choice (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return int(choice)
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

def capture_menu():
    """Menu for capturing calibration images."""
    clear_screen()
    print("Capture Calibration Images")
    print("==========================")
    
    # Get camera resolution
    resolution_choice = input("Enter camera resolution (width,height) [1280,720]: ").strip()
    if resolution_choice:
        try:
            width, height = map(int, resolution_choice.split(','))
            resolution = (width, height)
        except ValueError:
            print("Invalid resolution format. Using default 1280x720.")
            resolution = (1280, 720)
    else:
        resolution = (1280, 720)
    
    # Get number of images
    num_images = input("Enter number of images to capture [20]: ").strip()
    if num_images:
        try:
            num_images = int(num_images)
        except ValueError:
            print("Invalid number. Using default 20 images.")
            num_images = 20
    else:
        num_images = 20
    
    # Get save directory
    save_dir = input("Enter directory for calibration images [calibration_images]: ").strip()
    if not save_dir:
        save_dir = 'calibration_images'
    
    # Capture images
    capture_calibration_images(
        num_images=num_images,
        resolution=resolution,
        save_dir=save_dir
    )

def calibrate_menu():
    """Menu for calibrating the camera."""
    clear_screen()
    print("Calibrate Camera")
    print("================")
    
    # Get chessboard size
    chessboard_choice = input("Enter chessboard size (width,height inner corners) [9,6]: ").strip()
    if chessboard_choice:
        try:
            width, height = map(int, chessboard_choice.split(','))
            chessboard_size = (width, height)
        except ValueError:
            print("Invalid format. Using default 9x6.")
            chessboard_size = (9, 6)
    else:
        chessboard_size = (9, 6)
    
    # Get square size
    square_size = input("Enter chessboard square size in cm [2.5]: ").strip()
    if square_size:
        try:
            square_size = float(square_size)
        except ValueError:
            print("Invalid number. Using default 2.5cm.")
            square_size = 2.5
    else:
        square_size = 2.5
    
    # Get images directory
    save_dir = input("Enter directory with calibration images [calibration_images]: ").strip()
    if not save_dir:
        save_dir = 'calibration_images'
    
    # Get output file
    output_file = input("Enter output file for calibration data [camera_calibration.pkl]: ").strip()
    if not output_file:
        output_file = 'camera_calibration.pkl'
    
    # Calibrate camera
    calibrate_camera(
        chessboard_size=chessboard_size,
        square_size=square_size,
        save_dir=save_dir,
        output_file=output_file
    )

def test_menu():
    """Menu for testing calibration."""
    clear_screen()
    print("Test Calibration")
    print("===============")
    
    # Get calibration file
    calibration_file = input("Enter path to calibration file [camera_calibration.pkl]: ").strip()
    if not calibration_file:
        calibration_file = 'camera_calibration.pkl'
    
    # Get camera resolution
    resolution_choice = input("Enter camera resolution (width,height) [1280,720]: ").strip()
    if resolution_choice:
        try:
            width, height = map(int, resolution_choice.split(','))
            resolution = (width, height)
        except ValueError:
            print("Invalid resolution format. Using default 1280x720.")
            resolution = (1280, 720)
    else:
        resolution = (1280, 720)
    
    # Test calibration
    test_calibration(
        calibration_file=calibration_file,
        resolution=resolution
    )

def main():
    """Main function for the camera calibration tool."""
    while True:
        choice = get_menu_choice()
        
        if choice == 1:
            capture_menu()
        elif choice == 2:
            calibrate_menu()
        elif choice == 3:
            test_menu()
        elif choice == 4:
            run_all_steps()
        elif choice == 5:
            print("Exiting camera calibration tool. Goodbye!")
            break

if __name__ == "__main__":
    main()
