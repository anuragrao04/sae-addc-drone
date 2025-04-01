from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

class Driver():
    drop_location = () # tuple of (lat, long, heading, alt)
    home_location = () # tuple of (lat, long, heading, alt)
    safe_height = None
    vehicle = None  # DroneKit vehicle object

    def __init__(self, drop_location: tuple, safe_height: int) -> None:
        # 1. connect to the drone
        self.vehicle = connect('/dev/ttyAMA0', wait_ready=True, baud=57600)
        # Connect drone with raspberry pi, using physical port and not udp, 
        # coz it will be faster
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

    def go_to_drop_location(self) -> None:
        # goes to drop location, preserves height (it will be safe_height), then drops down to 10m altitude
        point = LocationGlobalRelative(self.drop_location[0], self.drop_location[1], self.safe_height)
        self.vehicle.simple_goto(point)
        # Wait until the vehicle reaches the target location
        while True:
            current_location = self.vehicle.location.global_relative_frame
            target_distance = self.get_distance_metres(current_location, point)
            print(f"Distance to target: {target_distance}m")
            if target_distance <= 1:  # Within 1 meter of target
                print("Reached target location")
                break
            time.sleep(1)
        #  Add logic to drop to 10m altitude
        #Stays at the same point,drops altitude to 10m
        point = LocationGlobalRelative(self.drop_location[0], self.drop_location[1], 10)
        self.vehicle.simple_goto(point)
        while True:
            current_location = self.vehicle.location.global_relative_frame
            target_distance = self.get_distance_metres(current_location, point)
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
        point = LocationGlobalRelative(self.drop_location[0], self.drop_location[1], desired_height)
        self.vehicle.simple_goto(point)
        new_height=self.vehicle.location.global_frame.alt
        while not (0.95 * desired_height <= new_height <= 1.05 * desired_height):
            print(f"Altitude: {self.vehicle.location.global_relative_frame.alt:.2f}m")
        if 0.95 * desired_height <= new_height <= 1.05 * desired_height:
            return True     
        
    
    def lower_to_detect_landing_target(self) -> None:
        # lowers the drone to 10m (experimental value - we should find the ideal height where the camera starts detecting a landing target, 10m is just a number I pulled out of my ass, it might change in the future)
        point = LocationGlobalRelative(self.drop_location[0], self.drop_location[1], 10)
        self.vehicle.simple_goto(point)
    
    # the values are obtained from landing_target.py
    # the main function then calls this method to send it to the drone
    # see: https://github.com/dronedojo/pidronescripts/blob/a74d509a7b3a3b64aae7c0fcd3109f8136bf9b6b/dk/drone/taco_delivery.py#L152
    def send_landing_target_vals(self, vals: tuple) -> None:
        # sends the landing target vals to the drone
        pass
    
    
