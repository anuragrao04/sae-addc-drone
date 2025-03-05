
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative
connection_string = "udp:0.0.0.0:14551"
# Connect to the Vehicle.
print("Connecting to vehicle on: %s" % (connection_string,))
vehicle = connect(connection_string, wait_ready=True)

# Get some vehicle attributes (state)
print("Get some vehicle attribute values:")
print(" GPS: %s" % vehicle.gps_0)
print(" Battery: %s" % vehicle.battery)
print(" Last Heartbeat: %s" % vehicle.last_heartbeat)
print(" Is Armable?: %s" % vehicle.is_armable)
print(" System status: %s" % vehicle.system_status.state)
print("Mode: %s" % vehicle.mode.name)    # settable

def arm_and_takeoff():
    print("Waiting for arm conditions")
    while not vehicle.is_armable:
        time.sleep(1)

    print("Arming")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(1)
        print("Waiting to switch to guided mode", vehicle.mode)
    print("vehicle mode: ", vehicle.mode)
    vehicle.armed = True
    print("Armed!")
    vehicle.simple_takeoff(30)
    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        if current_altitude >= 29:
            print("30m reached!")
            break
        print("Current Alt: ", current_altitude)
        time.sleep(1)

arm_and_takeoff()    
vehicle.airspeed = 15

wp1 = LocationGlobalRelative(12.79987404761409,77.81942172358943, 100)

vehicle.simple_goto(wp1)



# Close vehicle object before exiting script
vehicle.close()

# Shut down simulator
print("Completed")
