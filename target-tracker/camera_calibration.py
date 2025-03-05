# camera_calibration.py
import numpy as np
import cv2
import glob
import pickle
import os

def capture_calibration_images(camera_id=0, num_images=20):
    """Capture images of a chessboard pattern for camera calibration."""
    # Create directory for calibration images
    if not os.path.exists('calibration_images'):
        os.makedirs('calibration_images')
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    img_count = 0
    print("Press 'c' to capture an image, 'q' to quit")
    
    while img_count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display instruction on frame
        cv2.putText(frame, f"Captured: {img_count}/{num_images}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'c' to capture", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('Calibration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save image
            img_path = f'calibration_images/calib_{img_count}.jpg'
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_path}")
            img_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {img_count} images for calibration")

def calibrate_camera(chessboard_size=(9, 6), square_size=2.5):
    """Calibrate camera using chessboard images."""
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Path to calibration images
    images = glob.glob('calibration_images/*.jpg')
    
    if not images:
        print("No calibration images found in 'calibration_images' directory.")
        return None
    
    # Size for the first image
    img_size = None
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    if not objpoints:
        print("No chessboard patterns found in the images.")
        return None
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    
    # Save calibration results
    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'img_size': img_size
    }
    
    with open('camera_calibration.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)
    
    print("Calibration complete and data saved!")
    print("Camera Matrix:")
    print(camera_matrix)
    print("Distortion Coefficients:")
    print(dist_coeffs)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"Total reprojection error: {mean_error/len(objpoints)}")
    
    return calibration_data

if __name__ == "__main__":
    print("Camera Calibration Tool")
    print("1. Capture calibration images")
    print("2. Run calibration")
    choice = input("Enter your choice (1/2): ")
    
    if choice == '1':
        camera_id = int(input("Enter camera ID (default 0): ") or 0)
        num_images = int(input("Enter number of images to capture (default 20): ") or 20)
        capture_calibration_images(camera_id, num_images)
    elif choice == '2':
        chessboard_size = input("Enter chessboard size (width height) [default: 9 6]: ")
        if chessboard_size:
            width, height = map(int, chessboard_size.split())
            chessboard_size = (width, height)
        else:
            chessboard_size = (9, 6)
        
        square_size = float(input("Enter square size in cm [default: 2.5]: ") or 2.5)
        calibrate_camera(chessboard_size, square_size)
    else:
        print("Invalid choice")
