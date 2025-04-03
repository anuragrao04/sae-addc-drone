print ("Starting")
from dronekit import connect, VehicleMode

# Import DroneKit-Python

# Connect to the Vehicle.
vehicle = connect('/dev/ttyAMA0',  baud=57600)

# Get some vehicle attributes (state)
print ("Get some vehicle attribute values:")
print (" GPS: %s" % vehicle.gps_0)
print (" Battery: %s" % vehicle.battery)
print (" Last Heartbeat: %s" % vehicle.last_heartbeat)
print (" Is Armable?: %s" % vehicle.is_armable)
print (" System status: %s" % vehicle.system_status.state)
print (" Mode: %s" % vehicle.mode.name )   # settable

# Close vehicle object before exiting script
vehicle.close()


print("Completed")
