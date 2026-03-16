import open3d as o3d
import numpy as np
import pandas as pd
import os
from scipy.spatial.transform import Rotation as R

def visualize_debug_data():
    # File paths
    sat = "sat_point_cloud_capture.csv"
    poses_file = "capture_poses.csv"

    if not os.path.exists(sat) or not os.path.exists(poses_file):
        print("Error: Missing CSV files. Please run the capture script first.")
        return

    print("Loading camera poses...")
    poses_df = pd.read_csv(poses_file)
    
    print("Loading satellite point cloud data...")
    pcd_data = pd.read_csv(sat)

    geometries = []

    # 1. Create a coordinate frame for the World Origin (0,0,0) - Ceiling Camera
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    geometries.append(world_frame)

    # RGB colors for your 3 captures: Red, Green, Blue
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # 2. Iterate through each pose to process its respective point cloud
    for i, row in poses_df.iterrows():
        
        # --- Extract the specific points for this capture ---
        # If your CSV has a column identifying which capture a point belongs to:
        if 'capture_id' in pcd_data.columns:
            capture_points = pcd_data[pcd_data['capture_id'] == i]
        else:
            # Fallback: Assume the data is stacked sequentially in 3 equal chunks
            chunk_size = len(pcd_data) // len(poses_df)
            capture_points = pcd_data.iloc[i*chunk_size : (i+1)*chunk_size]

        # Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(capture_points[['X', 'Y', 'Z']].values)
        
        # Paint this capture its respective RGB color
        color_idx = i % 3 
        pcd.paint_uniform_color(colors[color_idx])

        # --- Build the 4x4 Transformation Matrix ---
        pos = [row['X_cm'], row['Y_cm'], row['Z_cm']]
        rot_matrix = R.from_euler('xyz', [row['Roll'], row['Pitch'], row['Yaw']], degrees=True).as_matrix()
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix  # Set 3x3 rotation
        transform_matrix[:3, 3] = pos          # Set 3x1 translation

        # --- Apply the math: Move from Local to Global ---
        pcd.transform(transform_matrix)
        geometries.append(pcd)

        # --- Create and transform the Camera Icon ---
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8.0)
        # We can apply the exact same transformation matrix to the camera frame!
        cam_frame.transform(transform_matrix) 
        geometries.append(cam_frame)
        
        # Add a line connecting the poses to show the path
        if i > 0:
            prev_pos = [poses_df.iloc[i-1]['X_cm'], poses_df.iloc[i-1]['Y_cm'], poses_df.iloc[i-1]['Z_cm']]
            points = [prev_pos, pos]
            lines = [[0, 1]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[1, 1, 0]]) # Yellow path
            geometries.append(line_set)

    print("Opening 3D Viewer...")
    print("Controls: Mouse to rotate, Shift+Mouse to pan, Scroll to zoom.")
    o3d.visualization.draw_geometries(geometries, window_name="Stitching Debug Viewer", 
                                      width=1280, height=720)

if __name__ == "__main__":
    visualize_debug_data()