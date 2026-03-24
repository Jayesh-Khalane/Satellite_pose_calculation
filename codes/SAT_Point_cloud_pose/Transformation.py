import open3d as o3d
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation as R_tool

def verify_stitching():
    poses_file = "log\capture_poses.csv"
    pcd_file = "log\sat_point_cloud_capture.csv"

    # 1. Verify files exist
    if not os.path.exists(poses_file) or not os.path.exists(pcd_file):
        print("Error: Missing CSV files. Run the capture script first")
        return

    print("Loading data...")
    poses_df = pd.read_csv(poses_file)
    pcd_data = pd.read_csv(pcd_file)

    geometries = []

    # Create a global coordinate frame (Ceiling Camera's Origin)
    # X=Red, Y=Green, Z=Blue
    global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
    geometries.append(global_frame)

    print(f"Found {len(poses_df)} poses. Processing...")

    # 2. Iterate through poses and apply transformations
    for i, row in poses_df.iterrows():
        # --- Separate the point cloud by Color ---
        # Capture 0 = Red, Capture 1 = Green, Capture 2 = Blue
        if i == 0:
            mask = (pcd_data['R'] > 128)
        elif i == 1:
            mask = (pcd_data['G'] > 128)
        elif i == 2:
            mask = (pcd_data['B'] > 128)
        else:
            continue # Failsafe if there are more than 3 poses

        chunk = pcd_data[mask]
        
        if len(chunk) == 0:
            print(f"Warning: No points found for Capture {i+1}")
            continue

        # --- Create Open3D PointCloud ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(chunk[['X', 'Y', 'Z']].values)
        
        # Open3D expects colors in the [0.0, 1.0] range
        pcd.colors = o3d.utility.Vector3dVector(chunk[['R', 'G', 'B']].values / 255.0)

        # --- Build the 4x4 Transformation Matrix ---
        pos = [row['X_cm'], row['Y_cm'], row['Z_cm']]
        euler_deg = [row['Roll'], row['Pitch'], row['Yaw']]
        
        # Convert Euler angles to 3x3 Rotation Matrix
        rot_matrix = R_tool.from_euler('xyz', euler_deg, degrees=True).as_matrix()

        T = np.eye(4)
        T[:3, :3] = rot_matrix  # 3x3 Rotation
        T[:3, 3] = pos          # 3x1 Translation

        # --- Apply the Transformation ---
        # This physically moves the points from the Local Frame to the Global Frame
        pcd.transform(T)
        geometries.append(pcd)

        # --- Optional: Draw a marker for where the camera was ---
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
        cam_frame.transform(T)
        geometries.append(cam_frame)

        print(f"Capture {i+1} stitched: {len(chunk)} points transformed.")

    # 3. Render the final global environment
    print("\nOpening 3D Viewer...")
    print(" -> The large coordinate frame is the Ceiling Camera (0,0,0).")
    print(" -> The smaller coordinate frames are your Local Camera poses.")
    print("Controls: Mouse to rotate, Shift+Mouse to pan, Scroll to zoom.")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Global Frame Verification", width=1280, height=720)
    
    # Set a dark background to make the colors pop
    vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])
    
    for geom in geometries:
        vis.add_geometry(geom)
        
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    verify_stitching()