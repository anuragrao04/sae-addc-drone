print ("Starting")
from dronekit import connect, VehicleMode
import time

# Import DroneKit-Python

# Connect to the Vehicle.
vehicle = connect('COM3', baud=57600)
home_location = vehicle.location.global_relative_frame
# sets the drone to guided mode, returns success
vehicle.mode = VehicleMode('GUIDED')
# Wait until the mode change is complete
while not vehicle.mode.name == 'GUIDED':
    print("Waiting for GUIDED mode...")
    time.sleep(1)
print("Vehicle is now in GUIDED mode")

# Arming the drone
while vehicle.is_armable != True:
    print("Waiting for vehicle to become armable.")
    time.sleep(1)
print("Vehicle is now armable")
    
vehicle.armed = True
        # Wait until the vehicle is armed
while not vehicle.armed:
    print("Waiting for vehicle to arm...")
    time.sleep(1)
print("Vehicle is now armed")
        
time.sleep(10)

vehicle.arm=False
while  vehicle.armed:
    print("Waiting for vehicle to disarm...")
    time.sleep(1)
print("Vehicle is now disarmed")