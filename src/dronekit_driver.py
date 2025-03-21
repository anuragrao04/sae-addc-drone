class Driver:
    drop_location = () # tuple of (lat, long, heading, alt)
    home_location = () # tuple of (lat, long, heading, alt)
    safe_height = 15
    def __init__(self, drop_location: tuple, safe_height: int) -> None:
        # 1. connect to vehicle
        # 2. other initialization checks
        # 3. sets variables
        pass

    def set_home_location(self) -> bool:
        # sets the current location of the drone as home location
        return True

    def _set_guided_mode(self) -> bool:
        # sets the drone to guided mode, returns success
        return True

    def arm_and_takeoff(self) -> bool:
        # arms and takes off to safe_height
        # returns success
        # it is a blocking call, until takeoff is over, the function doesn't return
        # also sets the drone into guided mode if it's not already in it. Utilize self._set_guided_mode()
        return True

    def go_to_drop_location(self) -> None:
        # goes to drop location, preserves height (it will be safe_height), then drops down to 10m altitude
        pass

    def drop_height(self, drop_height: float) -> bool:
        # drops the height of the drone by drop_height (represented in meters)
        # for example: if the current height is 30m, and drop_height is 2, the drone is supposed to go to 30-2 = 28m altitude
        # returns success
        return True
    
    def lower_to_detect_landing_target(self) -> None:
        # lowers the drone to 10m (experimental value - we should find the ideal height where the camera starts detecting a landing target, 10m is just a number I pulled out of my ass, it might change in the future)
        pass
    
    # the values are obtained from landing_target.py
    # the main function then calls this method to send it to the drone
    # see: https://github.com/dronedojo/pidronescripts/blob/a74d509a7b3a3b64aae7c0fcd3109f8136bf9b6b/dk/drone/taco_delivery.py#L152
    def send_landing_target_vals(self, vals: tuple) -> None:
        # sends the landing target vals to the drone
        pass
