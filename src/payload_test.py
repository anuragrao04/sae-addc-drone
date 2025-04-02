import argparse
import time
import dronekit_driver
import landing_target


def main():
    # the drop location is passed in through command line args:
    # python main.py --drop_location 12.3456,78.9101,180,20 --safe_height 15
    # where the values are lat, long, heading, alt
    # safe_height = the cruising height

    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_location', type=str, required=True)
    parser.add_argument('--safe_height', type=float, required=False, default=15) 
    args = parser.parse_args()
    drop_location = tuple(map(float, args.drop_location.split(',')))
    # drop location will now be a tuple of lat, long, heading, alt
    safe_height = args.safe_height
    #create servo instance
    # 1. create an instance of LandingTarget
    landingTarget = landing_target.LandingTarget()
    # 2. create an instance of Driver
    driver = dronekit_driver.Driver(drop_location=drop_location, safe_height=safe_height)


    # 3. set home location
    # success = driver.set_home_location()
    # if (not success):
    #     print("Failed to set home location. Exiting.")
    #     return

    # 4. arm and takeoff
    # driver.arm_and_takeoff()
    #
    # # # 5. go to drop location
    # # driver.go_to_drop_location()
    #
    # # 6. lower the drone to 10m
    # driver.lower_to_detect_landing_target()
    #
    # # 69. Switch drone to land mode
    # driver.switch_to_land_mode()
    #
    # # 10. land
    # while(not driver.is_landed()):
    #     la_target = landingTarget.get_landing_target_vals()
    #     if (la_target is None):
    #         print("No landing target detected.")
    #     else:
    #         driver.send_landing_target_vals(*la_target)
    # 
    # 11. actuate the motor to drop the anda
    driver.drop_the_anda()
    # 12. wait for a few seconds for the anda to drop
    # time.sleep(5)
    # # 13. arm and takeoff
    # driver.arm_and_takeoff()
    #
    # # 14. RTL home location set at step 3
    # driver.go_home()
    # 17. party!

if __name__ == '__main__':
    main()
