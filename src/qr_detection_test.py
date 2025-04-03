import cv2
import numpy as np
from landing_target import LandingTarget  # Assuming your class is in landing_target.py

def main():
    # Initialize LandingTarget
    try:
        landing_target = LandingTarget()
    except NotImplementedError:
        print("Camera calibration file not found. Exiting.")
        return
    
    while True:
        frame = landing_target._get_frame()
        qr_corners, qr_data = landing_target._get_qr_corners(frame)
        
        # Convert frame to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if qr_corners is not None: 
            # Draw detected QR code
            qr_corners = qr_corners.astype(int)
            for i in range(len(qr_corners)):
                cv2.line(frame_bgr, tuple(qr_corners[i]), tuple(qr_corners[(i+1) % 4]), (0, 255, 0), 2)
            
            # Display QR code data if available
            if qr_data:
                cv2.putText(frame_bgr, f"QR: {qr_data}", (qr_corners[0][0], qr_corners[0][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Get landing target values
            landing_vals = landing_target.calculate_landing_target_params(qr_corners)
            if landing_vals:
                text = f"AngleX: {landing_vals['angle_x']:.2f}, AngleY: {landing_vals['angle_y']:.2f}, Distance: {landing_vals['distance']:.2f}m"
                cv2.putText(frame_bgr, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow("QR Detection Feed", frame_bgr)
        
        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

