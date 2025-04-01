from dronekit import connect, VehicleMode
from pymavlink import mavutil
class servo_controller:
    def __init__(self) -> None:
        servoNo= 9 #used to tell ardupilot which aux port is the pwm output set : SERVO9_FUNCTION = 0 if the servo is connected a Pixhawks AUX OUT2
        open_pwm=80 # need to test according to the motor driver
        stop_pwm=0  #these are asuming we are using l298
        pass

    def controlServo(servo_number,pwm_value,vehicle):
        msg = vehicle.message_factory.command_long_encode(
                0,
                0,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                servo_number,
                pwm_value,
                0,
                0,
                0,
                0,
                0)
        vehicle.send_mavlink(msg)