class Driver:
    drop_location = () # tuple of (lat, long, heading, alt)
    home_location = () # tuple of (lat, long, heading, alt)
    safe_height = 15
    def __init__(self, drop_location: tuple, safe_height: int) -> None:
        # 1. connect to vehicle
        # 2. other initialization checks
        # 3. sets variables
        pass

    def set_home_location

    def set_guided_mode(self) -> bool:
        # sets the drone to guided mode, returns success
        return True

    def arm_and_takeoff(self) -> bool:
        # arms and takes off to safe_height
        # returns success
        # it is a blocking call, until takeoff is over, the function doesn't return
        return True

    def go_to_drop_location(self) -> None:
        # goes to drop location, preserves height (it will be safe_height), then drops down to 10m altitude
        pass

    def drop_height(self, drop_height: int) -> bool:
        # drops the height of the drone by drop_height.
        # for example: if the current height is 30m, and drop_height is 2, the drone is supposed to go to 30-2 = 28m altitude
        # returns success
        return True

