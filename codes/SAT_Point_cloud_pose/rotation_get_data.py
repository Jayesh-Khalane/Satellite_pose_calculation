import pyzed.sl as sl
import open3d as o3d
import numpy as np
import cv2
import csv
import os
from scipy.spatial.transform import Rotation as R_tool

# --- CONFIGURATION ---
MAX_DIST_CM = 100.0
LOG_DIR = "log"
ZED_SN = 33140394

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- INITIALIZE CAMERA ---
    cam = sl.Camera()
    init_params = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.HD720,
        depth_mode=sl.DEPTH_MODE.NEURAL_PLUS,
        coordinate_units=sl.UNIT.CENTIMETER, 
        coordinate_system=sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    )
    init_params.set_from_serial_number(ZED_SN)

    if cam.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Camera Initialization failed.")
        return

    # Enable tracking for stable IMU integration
    tracking_params = sl.PositionalTrackingParameters()
    cam.enable_positional_tracking(tracking_params)

    # --- OPEN3D SETUP ---
    vis = o3d.visualization.Visualizer()
    vis.create_window("Live ZED Feed (<100cm) & IMU", width=1280, height=720)
    vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])

    # 1. Geometry for the LIVE camera feed
    live_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(live_pcd)
    live_geom_added = True

    # 2. Geometry for the LIVE IMU marker
    imu_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15.0, origin=[0, 0, 0])
    vis.add_geometry(imu_marker)
    prev_rot_matrix = np.eye(3)

    # State variables
    runtime = sl.RuntimeParameters()
    sensors_data = sl.SensorsData()
    pc = sl.Mat()
    
    snapshot_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] # R, G, B for Open3D
    captured_data_list = []
    pose_log = []
    captures_done = 0

    # OpenCV Control Panel
    cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control Panel", 400, 200)

    print("\n==================================================")
    print("   INTERACTIVE CAPTURE MODE ACTIVATED")
    print("==================================================")
    print(" -> Keep the 'Control Panel' window in focus.")
    print(" -> Press 'c' to capture the current frame.")
    print(" -> Press 'q' to quit early.")

    try:
        while captures_done < 3:
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                
                # --- 1. LIVE IMU UPDATE ---
                current_euler = [0.0, 0.0, 0.0]
                if cam.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                    current_rot = sensors_data.get_imu_data().get_pose().get_rotation_matrix().r
                    current_euler = R_tool.from_matrix(current_rot).as_euler('xyz', degrees=True).tolist()
                    
                    # Spin the 3D marker
                    imu_marker.rotate(prev_rot_matrix.T, center=[0, 0, 0]) 
                    imu_marker.rotate(current_rot, center=[0, 0, 0])       
                    prev_rot_matrix = current_rot
                    vis.update_geometry(imu_marker)

                # --- 2. LIVE POINT CLOUD UPDATE ---
                cam.retrieve_measure(pc, sl.MEASURE.XYZRGBA)
                pc_np = pc.get_data()
                
                xyz = pc_np[:, :, :3].reshape(-1, 3)
                valid_mask = np.isfinite(xyz).all(axis=1)
                xyz_valid = xyz[valid_mask]
                
                # Filter points closer than 100cm
                dist_mask = np.linalg.norm(xyz_valid, axis=1) < MAX_DIST_CM
                xyz_filtered = xyz_valid[dist_mask]
                
                if len(xyz_filtered) > 0:
                    # Extract natural colors for the live feed
                    rgba_float = pc_np[:, :, 3].reshape(-1)[valid_mask][dist_mask]
                    rgba_int = rgba_float.view(np.uint32)
                    colors = np.stack([
                        (rgba_int & 0xFF).astype(np.uint8), 
                        ((rgba_int >> 8) & 0xFF).astype(np.uint8), 
                        ((rgba_int >> 16) & 0xFF).astype(np.uint8)
                    ], axis=1) / 255.0

                    live_pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
                    live_pcd.colors = o3d.utility.Vector3dVector(colors)
                    vis.update_geometry(live_pcd)

                vis.poll_events()
                vis.update_renderer()

                # --- 3. CONTROL PANEL & CAPTURE LOGIC ---
                # Draw a simple UI on the OpenCV window
                panel = np.zeros((200, 400, 3), dtype=np.uint8)
                cv2.putText(panel, f"Captures: {captures_done}/3", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(panel, f"Roll:  {current_euler[0]:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(panel, f"Pitch: {current_euler[1]:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(panel, f"Yaw:   {current_euler[2]:.1f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(panel, "[Press 'C' to Snap]", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Control Panel", panel)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting early...")
                    break
                elif key == ord('c') and len(xyz_filtered) > 0:
                    # Capture triggered! 
                    print(f"\n[!] SNAP! Capture {captures_done + 1} locked.")
                    
                    # 1. Save the pose
                    pose_log.append([captures_done + 1, 0.0, 0.0, 0.0, current_euler[0], current_euler[1], current_euler[2]])
                    
                    # 2. Save the point cloud data internally (converting back to 0-255 scale for CSV)
                    r_col, g_col, b_col = snapshot_colors[captures_done]
                    color_array_255 = np.full((xyz_filtered.shape[0], 3), [r_col * 255, g_col * 255, b_col * 255])
                    captured_data_list.append(np.hstack((xyz_filtered, color_array_255)))

                    # 3. Leave a colored "Ghost" in the Open3D viewer so you know what you already captured
                    ghost_pcd = o3d.geometry.PointCloud()
                    ghost_pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
                    ghost_pcd.paint_uniform_color([r_col, g_col, b_col])
                    vis.add_geometry(ghost_pcd)

                    captures_done += 1
                    
                    # Flash the screen slightly to confirm
                    panel[:] = (255, 255, 255)
                    cv2.imshow("Control Panel", panel)
                    cv2.waitKey(100)

    except KeyboardInterrupt:
        pass

    # --- FINAL EXPORT ---
    print("\nClosing streams and saving data...")
    if captured_data_list:
        # Save Point Cloud
        pc_path = os.path.join(LOG_DIR, "sat_point_cloud_capture.csv")
        np.savetxt(pc_path, np.vstack(captured_data_list), delimiter=",", header="X,Y,Z,R,G,B", comments="", fmt="%.4f")

        # Save Poses
        pose_path = os.path.join(LOG_DIR, "capture_poses.csv")
        with open(pose_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Capture_ID", "X_cm", "Y_cm", "Z_cm", "Roll", "Pitch", "Yaw"])
            writer.writerows(pose_log)
            
        print(f"SUCCESS: Saved {len(np.vstack(captured_data_list))} points across {captures_done} captures.")

    cam.close()
    vis.destroy_window()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()