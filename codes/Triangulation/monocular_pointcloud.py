import pyzed.sl as sl
import cv2
import numpy as np
import open3d as o3d
import time
import sys

def create_camera_frustum(scale=0.1):
    """Creates a wireframe box representing the camera."""
    points = [
        [0, 0, 0], 
        [-scale, -scale, scale*2], [scale, -scale, scale*2],
        [scale, scale, scale*2], [-scale, scale, scale*2]
    ]
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum

def main():
    # --- 1. Initialize ZED ---
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NONE 
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        return

    # Get camera intrinsics
    cam_info = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    K = np.array([[cam_info.fx, 0, cam_info.cx],
                  [0, cam_info.fy, cam_info.cy],
                  [0, 0, 1]])

    # --- 2. Initialize Open3D Visualizer Safely ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live Monocular Point Cloud", width=1280, height=720)
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    cam_frustum = create_camera_frustum(scale=0.15)
    
    # FIX: Initialize PointCloud with a single dummy point at the origin 
    # to prevent the "0 points axis-aligned bounding box" warning.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
    pcd.colors = o3d.utility.Vector3dVector(np.array([[0.0, 1.0, 0.0]]))

    vis.add_geometry(coord_frame)
    vis.add_geometry(cam_frustum)
    vis.add_geometry(pcd)
    
    # Run a few mandatory window render updates to bind the window size context
    vis.poll_events()
    vis.update_renderer()

    # --- 3. Tracking Variables ---
    image_zed = sl.Mat()
    sensors_data = sl.SensorsData()
    
    old_gray = None
    p0 = None
    
    pos = np.zeros((3, 1))
    vel = np.zeros((3, 1))
    R_current = np.eye(3)
    last_timestamp = None

    print("\n=======================================================")
    print("  Live Monocular Point Cloud & IMU Tracker Running     ")
    print("=======================================================")
    print("!! IMPORTANT: Slide the camera left/right to seed depth !!")
    print("Press 'q' inside the Open3D window to exit.\n")

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            
            # --- A. Extract Images and IMU ---
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame_raw = image_zed.get_data()
            frame_gray = cv2.cvtColor(frame_raw, cv2.COLOR_BGRA2GRAY)
            
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
            imu_data = sensors_data.get_imu_data()
            
            accel = np.array([imu_data.get_linear_acceleration()]).T 
            gyro = np.array([imu_data.get_angular_velocity()]).T     
            curr_timestamp = imu_data.timestamp.get_seconds()
            
            # --- B. IMU Pose Integration ---
            if last_timestamp is not None:
                dt = curr_timestamp - last_timestamp
                if dt > 0:
                    # Basic rotation tracking via gyroscope
                    theta = np.linalg.norm(gyro) * dt
                    if theta > 1e-5:
                        axis = (gyro / np.linalg.norm(gyro)).flatten()
                        rot_matrix, _ = cv2.Rodrigues(axis * theta)
                        R_current = R_current @ rot_matrix
                    
                    # Rough gravity cancellation (Assumes gravity is along Y-axis downwards)
                    g_world = np.array([[0], [9.81], [0]]) 
                    accel_world = (R_current @ accel) - g_world
                    
                    pos = pos + vel * dt + 0.5 * accel_world * (dt ** 2)
                    vel = vel + accel_world * dt
                    
                    # Update Camera Box Matrix
                    pose_matrix = np.eye(4)
                    pose_matrix[:3, :3] = R_current
                    pose_matrix[:3, 3] = pos.flatten()
                    
                    # Reset frustum geometry position to raw transformed state
                    cam_frustum.points = create_camera_frustum(scale=0.15).points
                    cam_frustum.transform(pose_matrix)

            last_timestamp = curr_timestamp

            # --- C. Epipolar Feature Triangulation ---
            if old_gray is None:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, maxCorners=400, qualityLevel=0.01, minDistance=10)
                old_gray = frame_gray.copy()
                continue
                
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
            
            if p1 is not None and len(p1) > 0:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            else:
                good_new, good_old = [], []
            
            # Match check
            if len(good_new) > 15:
                E, mask = cv2.findEssentialMat(good_new, good_old, K, method=cv2.RANSAC, prob=0.99, threshold=1.0)
                
                if E is not None and E.shape == (3, 3):
                    _, R_vis, t_vis, pose_mask = cv2.recoverPose(E, good_new, good_old, K)
                    
                    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
                    P2 = K @ np.hstack((R_vis, t_vis))
                    
                    # Triangulate
                    points_4d = cv2.triangulatePoints(P1, P2, good_old.T, good_new.T)
                    points_3d = points_4d[:3, :] / points_4d[3, :]
                    
                    # Mask out points behind camera frame
                    valid_mask = (points_3d[2, :] > 0.1) & (points_3d[2, :] < 10.0)
                    valid_points = points_3d[:, valid_mask].T
                    
                    if len(valid_points) > 0:
                        # Map local camera points to world space
                        world_points = (R_current @ valid_points.T).T + pos.flatten()
                        
                        # Direct injection into Open3D PointCloud structure
                        pcd.points = o3d.utility.Vector3dVector(world_points)
                        pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (len(world_points), 1)))
                        
                        print(f"Tracking Status | Active Features: {len(good_new)} | 3D Points Generated: {len(world_points)}", end="\r")
                    else:
                        print("Tracking Status | Parallax insufficient / Points behind lens.             ", end="\r")
                else:
                    print("Tracking Status | Essential Matrix generation failed.                       ", end="\r")
            else:
                print("Tracking Status | Lost tracking. Finding new tracking seeds...               ", end="\r")

            # Reset tracking elements
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
            
            if p0 is None or len(p0) < 50:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, maxCorners=400, qualityLevel=0.01, minDistance=10)

            # --- D. Cycle Open3D Rendering Engine ---
            vis.update_geometry(cam_frustum)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Prevent rendering logic from fully pegging a single CPU core thread
            time.sleep(0.005)

    zed.close()
    vis.destroy_window()

if __name__ == "__main__":
    main()