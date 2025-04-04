###########DEPENDENCIES################
import time
import socket
import exceptions
import math
import argparse

from dronekit import connect, VehicleMode,LocationGlobalRelative,APIException
from pymavlink import mavutil

import cv2
import cv2.aruco as aruco
import numpy as np

from imutils.video import WebcamVideoStream
import imutils

######GPS COORDINATES OF TACO DROPOFF#####
lat_taco=46.710140
lon_taco=-92.095865

#######VARIABLES####################

##Taco Delivery Variables
time_to_sleep=10 ###Seconds drone will wait after delivery before flying home
servo=14
dropoff_pwm=1900 ###PWM that will write servo to drop the taco off 
holdon_pwm=1100  ###PWM that holds on to the taco


##OpenCV aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()

ids_to_find = [129,72]
marker_sizes = [40,19] #cm
marker_heights = [7,4]
takeoff_height = 8
velocity = .5
##

##Camera
horizontal_res = 640
vertical_res = 480
cap = WebcamVideoStream(src=0, width=horizontal_res, height=vertical_res).start()

horizontal_fov = 62.2 * (math.pi / 180 ) ##Pi cam V1: 53.5 V2: 62.2
vertical_fov = 48.8 * (math.pi / 180)    ##Pi cam V1: 41.41 V2: 48.8


##REQUIRED: Calibration files for camera
##Look up https://github.com/dronedojo/video2calibration for info
calib_path="/home/pi/video2calibration/calibrationFiles/"

cameraMatrix   = np.loadtxt(calib_path+'cameraMatrix.txt', delimiter=',')
cameraDistortion   = np.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',')
#########

##Counters and script triggers
found_count=0
notfound_count=0

first_run=0 #Used to set initial time of function to determine FPS
start_time=0
end_time=0
script_mode = 1 ##1 for arm and takeoff, 2 for manual LOITER to GUIDED land 
ready_to_land=0 ##1 to trigger landing

manualArm=True ##If True, arming from RC controller, If False, arming from this script. 

#########FUNCTIONS#################

def connectMyCopter():

	parser = argparse.ArgumentParser(description='commands')
	parser.add_argument('--connect')
	args = parser.parse_args()

	connection_string = args.connect

	if not connection_string:
            connection_string='127.0.0.1:14550'

	vehicle = connect(connection_string,wait_ready=True)

	return vehicle

def arm_and_takeoff(targetHeight):
	while vehicle.is_armable!=True:
		print("Waiting for vehicle to become armable.")
		time.sleep(1)
	print("Vehicle is now armable")
    
	vehicle.mode = VehicleMode("GUIDED")
            
	while vehicle.mode!='GUIDED':
		print("Waiting for drone to enter GUIDED flight mode")
		time.sleep(1)
	print("Vehicle now in GUIDED MODE. Have fun!!")

        if manualArm==False:
            vehicle.armed = True
            while vehicle.armed==False:
                print("Waiting for vehicle to become armed.")
                time.sleep(1)
        else:
            if vehicle.armed == False:
                print("Exiting script. manualArm set to True but vehicle not armed.")
                print("Set manualArm to True if desiring script to arm the drone.")
                return None
        print("Look out! Props are spinning!!")
            
	vehicle.simple_takeoff(targetHeight) ##meters

	while True:
		print("Current Altitude: %d"%vehicle.location.global_relative_frame.alt)
		if vehicle.location.global_relative_frame.alt>=.95*targetHeight:
			break
		time.sleep(1)
	print("Target altitude reached!!")

	return None

def controlServo(servo_number,pwm_value):
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
    
def send_local_ned_velocity(vx, vy, vz):
	msg = vehicle.message_factory.set_position_target_local_ned_encode(
		0,
		0, 0,
		mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
		0b0000111111000111,
		0, 0, 0,
		vx, vy, vz,
		0, 0, 0,
		0, 0)
	vehicle.send_mavlink(msg)
	vehicle.flush()
    
def send_land_message(x,y):
    msg = vehicle.message_factory.landing_target_encode(
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        x,
        y,
        0,
        0,
        0,)
    vehicle.send_mavlink(msg)
    vehicle.flush()

def goto(targetLocation):
    distanceToTargetLocation = get_distance_meters(targetLocation,vehicle.location.global_relative_frame)

    vehicle.simple_goto(targetLocation)

    while vehicle.mode.name=="GUIDED":
        currentDistance = get_distance_meters(targetLocation,vehicle.location.global_relative_frame)
        if currentDistance<distanceToTargetLocation*.02:
            print("Reached target waypoint.")
            time.sleep(2)
            break
        time.sleep(1)
    return None

def get_distance_meters(targetLocation,currentLocation):
    dLat=targetLocation.lat - currentLocation.lat
    dLon=targetLocation.lon - currentLocation.lon

    return math.sqrt((dLon*dLon)+(dLat*dLat))*1.113195e5
    
def lander():
    global first_run,notfound_count,found_count,marker_size,start_time
    if first_run==0:
        print("First run of lander!!")
        first_run=1
        start_time=time.time()

    id_found=0
    frame = cap.read()
    frame = cv2.resize(frame,(horizontal_res,vertical_res))
    frame_np = np.array(frame)
    gray_img = cv2.cvtColor(frame_np,cv2.COLOR_BGR2GRAY)
    ids=''
    corners, ids, rejected = aruco.detectMarkers(image=gray_img,dictionary=aruco_dict,parameters=parameters)
    if vehicle.mode!='LAND':
        vehicle.mode=VehicleMode("LAND")
        while vehicle.mode!='LAND':
            print('WAITING FOR DRONE TO ENTER LAND MODE')
            time.sleep(1)

    counter=0
    corners_np = np.asarray(corners)

    id_to_find=0
    marker_height=0
    marker_size=0
    altitude = vehicle.location.global_relative_frame.alt ##Use vehicle.rangefinder.distance if rangefinder data desired instead

    if altitude > marker_heights[1]:
        id_to_find=ids_to_find[0]
        marker_height=marker_heights[0]
        marker_size=marker_sizes[0]
    elif altitude < marker_heights[1]:
        id_to_find=ids_to_find[1]
        marker_height=marker_heights[1]
        marker_size=marker_sizes[1]

    print("Marker "+str(id_to_find)+"at "+str(marker_size)+" cms.")

    try:
        if ids is not None:
            for id in ids:
                if id == id_to_find:
                    corners_single = [corners[counter]]
                    corners_single_np = np.asarray(corners_single)
                    ret = aruco.estimatePoseSingleMarkers(corners_single,marker_size,cameraMatrix=cameraMatrix,distCoeffs=cameraDistortion)
                    (rvec, tvec) = (ret[0][0, 0, :], ret[1][0, 0, :])
                    x = '{:.2f}'.format(tvec[0])
                    y = '{:.2f}'.format(tvec[1])
                    z = '{:.2f}'.format(tvec[2])

                    y_sum = 0
                    x_sum = 0

                    x_sum = corners_single_np[0][0][0][0]+ corners_single_np[0][0][1][0]+ corners_single_np[0][0][2][0]+ corners_single_np[0][0][3][0]
                    y_sum = corners_single_np[0][0][0][1]+ corners_single_np[0][0][1][1]+ corners_single_np[0][0][2][1]+ corners_single_np[0][0][3][1]

                    x_avg = x_sum*.25
                    y_avg = y_sum*.25
            
                    x_ang = (x_avg - horizontal_res*.5)*(horizontal_fov/horizontal_res)
                    y_ang = (y_avg - vertical_res*.5)*(vertical_fov/vertical_res)
            
                    if vehicle.mode!='LAND':
                        vehicle.mode = VehicleMode('LAND')
                        while vehicle.mode!='LAND':
                            time.sleep(1)
                        print("------------------------")
                        print("Vehicle now in LAND mode")
                        print("------------------------")
                        send_land_message(x_ang,y_ang)
                    else:
                        send_land_message(x_ang,y_ang)
                        pass
                    print("X CENTER PIXEL: "+str(x_avg)+" Y CENTER PIXEL: "+str(y_avg))
                    print("FOUND COUNT: "+str(found_count)+" NOTFOUND COUNT: "+str(notfound_count))
                    print("MARKER POSITION: x=" +x+" y= "+y+" z="+z)
                    found_count=found_count+1
                    id_found=1
                    print("")
                counter=counter+1
            if id_found==0:
                notfound_count=notfound_count+1
        else:
            notfound_count=notfound_count+1
    except Exception as e:
        print('Target likely not found. Error: '+str(e))
        notfound_count=notfound_count+1
    
     
 
 
######################################################

#######################MAIN###########################

######################################################

vehicle = connectMyCopter()

##
##SETUP PARAMETERS TO ENABLE PRECISION LANDING
##
vehicle.parameters['PLND_ENABLED'] = 1
vehicle.parameters['PLND_TYPE'] = 1 ##1 for companion computer
vehicle.parameters['PLND_EST_TYPE'] = 0 ##0 for raw sensor, 1 for kalman filter pos estimation
vehicle.parameters['LAND_SPEED'] = 20 ##Descent speed of 30cm/s

###Record home waypoint so we can fly home after delivery
lat_home=vehicle.location.global_relative_frame.lat
lon_home=vehicle.location.global_relative_frame.lon

wp_home=LocationGlobalRelative(lat_home,lon_home,takeoff_height)
wp_taco=LocationGlobalRelative(lat_taco,lon_taco,takeoff_height)

distanceBetweenLaunchAndTaco=get_distance_meters(wp_taco,wp_home)
print("Taco dropoff is "+str(distanceBetweenLaunchAndTaco)+" meters from home wp.")

controlServo(servo,holdon_pwm) ##Write servo to position to hold onto taco
time.sleep(1)
############FLY DRONE TO TACO DROPOFF############
arm_and_takeoff(takeoff_height)
goto(wp_taco)

##If we are here the drone has reached the target waypoint and is ready to land
while vehicle.armed==True:
    lander() ##Precision Landing function

print("")
print("----------------------------------")
print("Arrived at the taco destination!")
print("Dropping tacos and heading home.")
print("----------ENJOY!----------------")

controlServo(servo,dropoff_pwm)
time.sleep(1)
#controlServo(servo,holdon_pwm)
time.sleep(time_to_sleep)

#############FLY THE DRONE TO HOME WAYPOINT######

arm_and_takeoff(takeoff_height)
goto(wp_home)

##If we are here drone has reached the target waypoint and is ready to land
while vehicle.armed==True:
    lander() ##Precision Landing function
print("")
print("----------------------------------")
print("Made it home for another delivery!")
print("----------------------------------")