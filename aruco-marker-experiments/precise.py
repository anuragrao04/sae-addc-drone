import time
import cv2
import numpy as np
import pickle
import math
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil  # MAVLink communication
from pyzbar.pyzbar import decode
from math import radians, cos, sin, sqrt, atan2

# Function to calculate distance using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of Earth in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2- lon1)
    a = sin(dlat/2) * sin(dlat/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2) * sin(dlon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c  # Distance in meters

# Connect to drone
connection_string = "udp:127.0.0.1:14550"
print(f"Connecting to vehicle on: {connection_string}")
vehicle = connect(connection_string, wait_ready=True)

# Function to arm and takeoff
def arm_and_takeoff(target_altitude):
    print("Waiting for arm conditions")
    while not vehicle.is_armable:
        time.sleep(1)
    
    print("Arming")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(1)
    
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(1)
    print("Armed!")
    
    vehicle.simple_takeoff(target_altitude)
    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        print(f"Current Altitude: {current_altitude:.2f} m")
        if current_altitude >= target_altitude * 0.95:
            print("Target altitude reached!")
            break
        time.sleep(1)

# Step 1: Takeoff
launch_location = vehicle.location.global_relative_frame
arm_and_takeoff(30)

# Step 2: Fly to the target waypoint
wp1 = LocationGlobalRelative(12.79987404761409, 77.81942172358943, 100)
vehicle.simple_goto(wp1)

time.sleep(2)
while True:
    current_location = vehicle.location.global_relative_frame
    distance = haversine(current_location.lat, current_location.lon, wp1.lat, wp1.lon)
    print(f"Distance to waypoint: {distance:.2f} meters")
    if distance < 10.0:  # Activate QR detection when close
        print("Activating QR-based precision landing!")
        break
    time.sleep(1)

# Step 3: QR-based Precision Landing Class
class QRCodeTracker:
    def __init__(self, camera_id=0, calibration_file='camera_calibration.pkl', known_qr_size_cm=10.0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")

        self.known_qr_size_m = known_qr_size_cm / 100.0
        self.load_calibration(calibration_file)

        # MAVLink connection through DroneKit
        self.mavlink_connection = vehicle

    def load_calibration(self, calibration_file):
        try:
            with open(calibration_file, 'rb') as f:
                calibration_data = pickle.load(f)
            self.camera_matrix = calibration_data['camera_matrix']
            self.dist_coeffs = calibration_data['dist_coeffs']
        except FileNotFoundError:
            print("Calibration file not found. Using default parameters.")
            self.camera_matrix = np.eye(3)
            self.dist_coeffs = np.zeros((5, 1))

    def send_landing_target(self, x_offset, y_offset, distance):
        """ Send LANDING_TARGET message to the flight controller """
        msg = self.mavlink_connection.message_factory.landing_target_encode(
            0,          # Time since boot (not needed)
            0,          # Target number (0 = main landing target)
            mavutil.mavlink.MAV_FRAME_BODY_NED,  # Frame (relative to drone)
            x_offset,   # X offset (meters, right = positive, left = negative)
            y_offset,   # Y offset (meters, forward = positive, back = negative)
            distance,   # Distance (meters) to landing target
            0, 0        # Unused angle parameters
        )
        self.mavlink_connection.send_mavlink(msg)
        self.mavlink_connection.flush()
        print(f"Sent LANDING_TARGET: x={x_offset}, y={y_offset}, distance={distance}")

    def detect_qr_code(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return decode(gray)

    def calculate_alignment(self, qr_code):
        """ Calculate drone's position relative to QR code and send MAVLink message """
        corners = np.array([(p.x, p.y) for p in qr_code.polygon], dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            corners, self.camera_matrix, self.dist_coeffs
        )
        if success:
            alignment = tvec.flatten()  # [x_offset, y_offset, distance]
            self.send_landing_target(alignment[0], alignment[1], alignment[2])
            return alignment
        return None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            qr_codes = self.detect_qr_code(frame)
            for qr_code in qr_codes:
                alignment = self.calculate_alignment(qr_code)
                if alignment:
                    print(f"Alignment: {alignment}")  # Logs for debugging
        self.cap.release()
        return None

# Step 4: Align with QR code and descend using MAVLink precision landing
qr_tracker = QRCodeTracker()
print("Starting QR-based precision landing...")

while True:
    alignment = qr_tracker.run()
    if alignment:
        print("QR detected, sending LANDING_TARGET messages...")
    
    if vehicle.location.global_relative_frame.alt <= 1.0:
        print("Landed. Now dropping the payload.")
        time.sleep(2)
        break

# Step 5: Take off again to return to launch location
arm_and_takeoff(30)
print("Returning to launch location...")
vehicle.simple_goto(launch_location)

while True:
    current_location = vehicle.location.global_relative_frame
    distance = haversine(current_location.lat, current_location.lon, launch_location.lat, launch_location.lon)
    print(f"Distance to launch location: {distance:.2f} meters")
    if distance < 1.0:
        print("Back at launch location!")
        break
    time.sleep(1)

# Step 6: Final Landing
print("Landing...")
vehicle.mode = VehicleMode("LAND")
while vehicle.location.global_relative_frame.alt > 0.5:
    print(f"Current Altitude: {vehicle.location.global_relative_frame.alt:.2f} m")
    time.sleep(1)
print("Landed back at launch location.")

# Step 7: Close vehicle connection
vehicle.close()
print("Mission completed")

