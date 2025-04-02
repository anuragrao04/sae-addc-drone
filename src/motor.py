from pymavlink import mavutil

class MotorController:
    def __init__(self, vehicle):
        self.vehicle = vehicle  # Store vehicle instance
        self.servoNo = 9  # ArduPilot AUX port for PWM output
        self.open_pwm = 80  # Adjust as needed to open the mechanism
        self.stop_pwm = 0  # PWM value to stop the servo (adjust as needed)

    def control_servo(self, servo_number, pwm_value):
        """Sends a MAVLink command to control a servo."""
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            servo_number,
            pwm_value,
            0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)

    
    def open(self):
        """Opens the payload dropping mechanism."""
        print("Opening payload mechanism...")
        self.control_servo(self.servoNo, self.open_pwm)

    def stop(self):
        """Stops the payload dropping mechanism."""
        print("Stopping payload mechanism...")
        self.control_servo(self.servoNo, self.stop_pwm)
        return True
    
