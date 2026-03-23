import pyzed.sl as sl
import cv2
import numpy as np
from collections import deque

def draw_3d_axis(image, rotation_matrix, center, size=100):
    """Draws a 3D axis marker on the 2D image based on the IMU rotation matrix."""
    # Define the 3D axes in the local camera frame
    axes_3d = np.float32([[size, 0, 0], [0, size, 0], [0, 0, size]])
    
    # Rotate the axes using the IMU's rotation matrix
    rotated_axes = np.dot(axes_3d, rotation_matrix.T)
    
    # Project back to 2D (simplified orthographic projection for the marker)
    origin = tuple(map(int, center))
    x_axis = tuple(map(int, center + rotated_axes[0][:2])) # Red
    y_axis = tuple(map(int, center - rotated_axes[1][:2])) # Green (flipped Y for OpenCV)
    z_axis = tuple(map(int, center - rotated_axes[2][:2])) # Blue
    
    cv2.line(image, origin, x_axis, (0, 0, 255), 3) # X - Red
    cv2.line(image, origin, y_axis, (0, 255, 0), 3) # Y - Green
    cv2.line(image, origin, z_axis, (255, 0, 0), 3) # Z - Blue

def main():
    # 1. Initialize ZED Camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_serial_number(33140394) # Target your specific ZED
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.CENTIMETER

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {err}")
        return

    # Buffers to hold recent IMU data for Jitter and P2P calculation
    window_size = 50
    roll_hist = deque(maxlen=window_size)
    pitch_hist = deque(maxlen=window_size)
    yaw_hist = deque(maxlen=window_size)

    image_zed = sl.Mat()
    sensors_data = sl.SensorsData()

    print("Reading IMU Data... Press 'q' to quit.")

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image for display
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()

            # 2. Get IMU Data
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                imu_data = sensors_data.get_imu_data()
                
                # Get Orientation (Euler Angles in degrees)
                euler = imu_data.get_pose().get_euler_angles()
                roll, pitch, yaw = euler[0], euler[1], euler[2]
                
                # Get Rotation Matrix for the visual marker
                rot_matrix = imu_data.get_pose().get_rotation_matrix().r

                # 3. Calculate Jitter and P2P
                roll_hist.append(roll)
                pitch_hist.append(pitch)
                yaw_hist.append(yaw)

                if len(roll_hist) == window_size:
                    # Calculate Peak-to-Peak (Max - Min)
                    r_p2p = np.ptp(roll_hist)
                    p_p2p = np.ptp(pitch_hist)
                    y_p2p = np.ptp(yaw_hist)

                    # Calculate Jitter (Standard Deviation)
                    r_jitter = np.std(roll_hist)
                    p_jitter = np.std(pitch_hist)
                    y_jitter = np.std(yaw_hist)

                    # 4. Display text overlays
                    text_color = (0, 255, 255)
                    cv2.putText(frame, f"Roll: {roll:.2f} | P2P: {r_p2p:.3f} | Jit: {r_jitter:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    cv2.putText(frame, f"Pitch: {pitch:.2f} | P2P: {p_p2p:.3f} | Jit: {p_jitter:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    cv2.putText(frame, f"Yaw: {yaw:.2f} | P2P: {y_p2p:.3f} | Jit: {y_jitter:.3f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                # 5. Draw the Orientation Marker
                height, width, _ = frame.shape
                marker_center = (width - 150, height - 150) # Bottom right corner
                draw_3d_axis(frame, rot_matrix, marker_center)

            # Show the frame
            cv2.imshow("ZED IMU & Orientation Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()