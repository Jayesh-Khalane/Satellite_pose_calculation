import open3d as o3d
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation as R_tool

def stitch_with_manual_pose():
    # File paths
    poses_file = "log/capture_poses.csv"
    pcd_file = "log/sat_point_cloud_capture.csv"

    if not os.path.exists(poses_file) or not os.path.exists(pcd_file):
        print("Error: Missing CSV files in the 'log' folder.")
        return

    print("Loading point cloud and manual pose data...")
    poses_df = pd.read_csv(poses_file)
    pcd_data = pd.read_csv(pcd_file)

    geometries = []

    # 1. World Origin (0,0,0) - This is your reference "Zero" point
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30.0, origin=[0, 0, 0])
    geometries.append(world_frame)

    # 2. Process each capture
    for i, row in poses_df.iterrows():
        # Identify points by color (Capture 1=Red, 2=Green, 3=Blue)
        if i == 0: mask = pcd_data['R'] > 128
        elif i == 1: mask = pcd_data['G'] > 128
        elif i == 2: mask = pcd_data['B'] > 128
        else: continue

        chunk = pcd_data[mask]
        if len(chunk) == 0:
            print(f"Warning: Capture {i+1} has no points.")
            continue

        # Create Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(chunk[['X', 'Y', 'Z']].values)
        pcd.colors = o3d.utility.Vector3dVector(chunk[['R', 'G', 'B']].values / 255.0)

        # --- BUILD THE TRANSFORMATION ---
        # Get Manual Position (X, Y, Z)
        tx, ty, tz = row['X_cm'], row['Y_cm'], row['Z_cm']
        
        # Get IMU Orientation (Roll, Pitch, Yaw)
        roll, pitch, yaw = row['Roll'], row['Pitch'], row['Yaw']
        
        # Generate 3x3 Rotation Matrix
        rot_matrix = R_tool.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

        # Build 4x4 Homogeneous Matrix
        T = np.eye(4)
        T[:3, :3] = rot_matrix  # Apply Rotation
        T[:3, 3] = [tx, ty, tz] # Apply Manual Translation

        # --- APPLY MATH ---
        pcd.transform(T)
        geometries.append(pcd)

        # Draw a Coordinate Frame for the camera's location
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=12.0)
        cam_frame.transform(T)
        geometries.append(cam_frame)

        print(f"Stitched Capture {i+1} at Pos:[{tx}, {ty}, {tz}] Rot:[{roll}, {pitch}, {yaw}]")

    # 3. Final Visualization
    print("\nOpening 3D Viewer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Manual Pose Stitching Result", width=1280, height=720)
    vis.get_render_option().background_color = np.asarray([0.05, 0.05, 0.05]) # Very dark gray
    vis.get_render_option().point_size = 2.0
    
    for geom in geometries:
        vis.add_geometry(geom)
        
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    stitch_with_manual_pose()