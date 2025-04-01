print("Starting")
from dronekit import connect, VehicleMode
import time

# Connect to the Vehicle.
vehicle = connect('COM3', baud=57600)

# Wait for GPS lock before setting home location
# while not vehicle.home_location:
#     print("Waiting for GPS lock...")
#     time.sleep(1)
# home_location = vehicle.home_location
# print(f"Home location set to: {home_location}")

# Set drone to GUIDED mode
vehicle.mode = VehicleMode('GUIDED')

while vehicle.mode.name != 'GUIDED':
    print("Waiting for GUIDED mode...")
    time.sleep(1)
print("Vehicle is now in GUIDED mode")

# Wait for the vehicle to become armable
while not vehicle.is_armable:
    print("Waiting for vehicle to become armable...")
    time.sleep(1)
print("Vehicle is now armable")

# Arm the vehicle
vehicle.armed = True

while not vehicle.armed:
    print("Waiting for vehicle to arm...")
    time.sleep(1)
print("Vehicle is now armed")

time.sleep(3)

# Takeoff
target_altitude = 5
print(f"Taking off to {target_altitude}m!")
vehicle.simple_takeoff(target_altitude)

# Monitor altitude
while True:
    print(f"Altitude: {vehicle.location.global_relative_frame.alt:.2f}m")

    # Check for excessive tilt
    # if (abs(vehicle.gimbal.roll) > 20 or 
    #     abs(vehicle.gimbal.pitch) > 20 or 
    #     abs(vehicle.gimbal.yaw) > 20):
    #     print("Excessive tilt detected! Initiating LAND mode...")
    #     vehicle.mode = VehicleMode("LAND")
    #     break

    # Break when altitude is reached
    if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
        print("Reached target altitude")
        break
    
    time.sleep(1)



# Land the vehicle
print("Initiating landing...")
vehicle.mode = VehicleMode("LAND")

# Wait for disarming
while vehicle.armed:
    vehicle.armed = False
    print("Waiting for vehicle to disarm...")
    time.sleep(1)

print("Vehicle is now disarmed")
vehicle.close()
print("Flight completed")
