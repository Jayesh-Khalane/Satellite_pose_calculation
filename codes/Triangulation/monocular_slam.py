import pyzed.sl as sl
import cv2
import numpy as np
import sys

def main():
    # --- 1. Initialize ZED SDK (Strictly for Data Extraction) ---
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    # Crucial for VIO: We need the rawest data possible, no SDK spatial mapping
    init_params.depth_mode = sl.DEPTH_MODE.NONE 

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED Camera")
        sys.exit()

    # --- 2. Data Containers ---
    image_zed = sl.Mat()
    sensors_data = sl.SensorsData()
    
    # Feature tracking variables
    old_gray = None
    p0 = None # Good features to track
    
    # Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    print("\n--- Monocular + IMU Extraction Pipeline Running ---")
    print("Press 'q' to exit.\n")

    while True:
        # Grab a new frame from the camera
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            
            # --- A. Extract Tightly Coupled Hardware Data ---
            # 1. Get the left monocular image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            
            # 2. Get the IMU data synchronized to exactly when this image was taken
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
            imu_data = sensors_data.get_imu_data()
            
            # Extract linear acceleration (m/s^2) and angular velocity (rad/s)
            linear_accel = imu_data.get_linear_acceleration()
            angular_vel = imu_data.get_angular_velocity()
            
            # --- B. Visual Tracking Frontend (SVO-style optical flow) ---
            if old_gray is None:
                # Initialization: Find strong corners to track in the first frame
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                old_gray = frame_gray.copy()
                continue
                
            # Track features using Lucas-Kanade
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            else:
                good_new, good_old = [], []

            # --- C. Visualization ---
            display_frame = frame.copy()
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Draw motion tracks
                display_frame = cv2.line(display_frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                display_frame = cv2.circle(display_frame, (int(a), int(b)), 4, (0, 0, 255), -1)

            # Overlay IMU data on the screen
            imu_text = f"Accel (m/s^2): X:{linear_accel[0]:.2f} Y:{linear_accel[1]:.2f} Z:{linear_accel[2]:.2f}"
            gyro_text = f"Gyro (rad/s): X:{angular_vel[0]:.2f} Y:{angular_vel[1]:.2f} Z:{angular_vel[2]:.2f}"
            cv2.putText(display_frame, imu_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, gyro_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Monocular VIO Frontend", display_frame)

            # --- D. The "Missing" Backend Step ---
            # In a real SLAM system, right here you would:
            # 1. Take the pixel shift of `good_new` vs `good_old`
            # 2. Take the `linear_accel` and `angular_vel`
            # 3. Feed them into an IMU Preintegration module.
            # 4. Pass the result to a Non-Linear Least Squares solver (like Ceres) to get the 6DOF pose.

            # Update previous frame and points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            
            # If we lose too many features, re-detect
            if len(p0) < 30:
                 p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()