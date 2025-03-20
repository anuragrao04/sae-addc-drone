class LandingTarget:
    def __init__(self):
        pass

    # this function returns the dictionary for the set of values required for landing target protocol. 
    # see: https://github.com/dronedojo/pidronescripts/blob/a74d509a7b3a3b64aae7c0fcd3109f8136bf9b6b/dk/drone/taco_delivery.py#L152
    # It should mux between the aruco marker for higher altitudes and qr code when getting closer
    # if (qr_readable?): return landing target vals wrt qr
    # elif (aruco_readable?): return landing target vals wrt aruco
    # else: return None
    def get_lading_target_vals(self) -> tuple:
        return ()
    
    # methods prepended with _ are helper methods
    # attempts to read a qr code from the given frame. Returns it's coordinates in the frame. 
    # returns None if qr not found
    def _get_qr_position(self, frame):
        pass
    
    # same thing as _get_qr_position, but with aruco marker
    def _get_aruco_position(self, frame):
        pass

    # you can add more helper methods to this class as needed

