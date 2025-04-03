import math
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import motor
from pymavlink import mavutil
class Driver():
    def __init__(self, drop_location: tuple, safe_height: int) -> None:
        # 1. connect to the drone
        print("Trying to connect to the damn drone")
        self.vehicle = connect('/dev/ttyAMA0', baud=57600)
        print("Connect ho gaya")
        # self.vehicle.parameters['PLND_ENABLED'] = 1
        # self.vehicle.parameters['PLND_TYPE'] = 1 ##1 for companion computer
        # self.vehicle.parameters['PLND_EST_TYPE'] = 0 ##0 for raw sensor, 1 for kalman filter pos estimation
        # self.vehicle.parameters['LAND_SPEED'] = 20 ##Descent speed of 30cm/s
        # Connect drone with raspberry pi, using physical port and not udp, 
        # coz it will be faster
        self.motor = motor.MotorController(self.vehicle)
        # 2. TODO: other initialization checks
        
        # 3. sets variables
        self.drop_location = drop_location
        self.safe_height = safe_height

    def set_home_location(self) -> bool:
        # sets the current location of the drone as home location
        self.home_location = self.vehicle.location.global_relative_frame
        return True

    def _set_guided_mode(self) -> bool:
        # sets the drone to guided mode, returns success
        self.vehicle.mode = VehicleMode('GUIDED')
        # Wait until the mode change is complete
        while not self.vehicle.mode.name == 'GUIDED':
            print("Waiting for GUIDED mode...")
            time.sleep(1)
        print("Vehicle is now in GUIDED mode")
        return True

    def arm_and_takeoff(self) -> bool:
        # arms and takes off to safe_height
        # returns success
        # it is a blocking call, until takeoff is over, the function doesn't return
        self._set_guided_mode()
        while self.vehicle.is_armable != True:
            print("Waiting for vehicle to become armable.")
            time.sleep(1)
        print("Vehicle is now armable")
    
        self.vehicle.armed = True
        # Wait until the vehicle is armed
        while not self.vehicle.armed:
            print("Waiting for vehicle to arm...")
            time.sleep(1)
        print("Vehicle is now armed")
        #sleep time to let the motors reach proper speed
        time.sleep(3)

        # Takeoff to target altitude
        print("Taking off")
        self.vehicle.simple_takeoff(self.safe_height)
        # Wait until the vehicle reaches the target altitude
        while True:
            print(f"Altitude: {self.vehicle.location.global_relative_frame.alt}")
            if self.vehicle.location.global_relative_frame.alt >= self.safe_height * 0.95:  # 95% of target altitude
                #Using 0.95 coz 100% accuracy is not possible
                print(f"Reached target altitude of {self.safe_height}m")
                break
            time.sleep(1)
        return True

    def get_distance_meters(self, targetLocation , currentLocation):
        dLat=targetLocation.lat - currentLocation.lat
        dLon=targetLocation.lon - currentLocation.lon
        return math.sqrt((dLon*dLon)+(dLat*dLat))*1.113195e5


    def go_to_drop_location(self) -> None:
        # goes to drop location, preserves height (it will be safe_height)
        point = LocationGlobalRelative(self.drop_location[0], self.drop_location[1], self.safe_height)
        self.vehicle.simple_goto(point)
        # Wait until the vehicle reaches the target location
        while True:
            current_location = self.vehicle.location.global_relative_frame
            target_distance = self.get_distance_meters(current_location, point)
            print(f"Distance to target: {target_distance}m")
            if target_distance <= 1:  # Within 1 meter of target
                print("Reached target location")
                break
            time.sleep(1)

    def drop_height(self, drop_height: float) -> bool:
        # drops the height of the drone by drop_height (represented in meters)
        # for example: if the current height is 30m, and drop_height is 2, the drone is supposed to go to 30-2 = 28m altitude
        # returns success
        current_height= self.vehicle.location.global_frame.alt
        desired_height=current_height-drop_height
        current_location=self.vehicle.location.global_frame
        point = LocationGlobalRelative(
            current_location.lat,      # Use current latitude
            current_location.lon,      # Use current longitude
            desired_height # Set the desired target altitude
        )
        self.vehicle.simple_goto(point)
        new_height=self.vehicle.location.global_frame.alt
        while not (0.95 * desired_height <= new_height <= 1.05 * desired_height):
            print(f"Altitude: {self.vehicle.location.global_relative_frame.alt:.2f}m")
            new_height=self.vehicle.location.global_frame.alt
        return True

    def lower_to_detect_landing_target(self) -> None:
        # lowers the drone to 10m (experimental value - we should find the ideal height where the camera starts detecting a landing target, 10m is just a number I pulled out of my ass, it might change in the future)
        current_location = self.vehicle.location.global_relative_frame
        point = LocationGlobalRelative(
            current_location.lat,      # Use current latitude
            current_location.lon,      # Use current longitude
            10 # Set the desired target altitude
        )
        self.vehicle.simple_goto(point)
    # the values are obtained from landing_target.py
    # the main function then calls this method to send it to the drone
    # see: https://github.com/dronedojo/pidronescripts/blob/a74d509a7b3a3b64aae7c0fcd3109f8136bf9b6b/dk/drone/taco_delivery.py#L152
    def send_landing_target_vals(self, angle_x: float, angle_y: float) -> None:
        # sends the landing target vals to the drone
        msg = self.vehicle.message_factory.landing_target_encode(
            0,
            0,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            angle_x,
            angle_y,
            0,
            0,
            0,)
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def switch_to_land_mode(self):
        # sets the drone to land mode, returns success
        self.vehicle.mode = VehicleMode('LAND')
        # Wait until the mode change is complete
        while not self.vehicle.mode.name == 'LAND':
            print("Waiting for LAND mode...")
            time.sleep(1)
        print("Vehicle is now in LAND mode")
        return True

    def is_landed(self) -> bool:
        # returns whether the drone has landed or not
        # if it's arm, it's udi udi
        # else it's on floor
        return not self.vehicle.armed

    def drop_the_anda(self):
        # drops the damn anda
        self.motor.open()
        time.sleep(3)
        self.motor.stop()

    def go_home(self):
        # goes to home location
        point = LocationGlobalRelative(self.home_location.lat, self.home_location.lon, self.safe_height)
        self.vehicle.simple_goto(point)
        while True:
            current_location = self.vehicle.location.global_relative_frame
            target_distance = self.get_distance_meters(current_location, point)
            print(f"Distance to home: {target_distance}m")
            if target_distance <= 1:
                print("Reached home location, starting landing procedure.")
                break
            time.sleep(1)
        
        print("Lowering to detect landing target...")
        
        
    """
    def land_on_aruco(self):
        self.lower_to_detect_landing_target()

        detection_attempts = 0
        max_attempts = 10  # Fail-safe limit

        while detection_attempts < max_attempts:
            # Get ArUco marker detection values
            marker_detected, angle_x, angle_y = detect_aruco_marker()  # Ensure this function exists and works

            if marker_detected:
                print(f"Marker detected: X={angle_x:.2f}, Y={angle_y:.2f}")
                self.send_landing_target_vals(angle_x, angle_y)
                self.drop_height(2)  # Gradual descent of 2m
            else:
                print("Landing target not detected. Hovering and retrying...")
                detection_attempts += 1
                time.sleep(1)

        if detection_attempts >= max_attempts:
            print("ArUco marker not detected after multiple attempts. Landing normally.")
        
        self.switch_to_land_mode()

        # Wait until the drone lands
        while not self.is_landed():
            print("Landing in progress...")
            time.sleep(2)

        print("Drone has landed successfully.")
        """  
