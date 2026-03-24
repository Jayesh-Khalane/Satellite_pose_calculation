import open3d as o3d
import pandas as pd
import numpy as np
import os

def pick_points(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. Load the CSV
    print("Loading satellite data...")
    df = pd.read_csv(csv_path)
    xyz = df[['X', 'Y', 'Z']].values
    rgb = df[['R', 'G', 'B']].values / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print("-" * 50)
    print("INSTRUCTIONS FOR POINT PICKING:")
    print("1) Hold [Shift] and [Left-Click] to select a point.")
    print("2) A small sphere will appear on the point you picked.")
    print("3) Press [Shift] and [Right-Click] to undo a selection.")
    print("4) After picking your two points, CLOSE THE WINDOW to see coordinates.")
    print("-" * 50)

    # 2. Launch the visualizer with point picking enabled
    # This is a built-in Open3D utility for exactly this purpose
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Shift+LeftClick to pick P1 and P2", width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Optional: Make points easier to see
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])  # Black background

    vis.run() # Script pauses here until you close the window
    vis.destroy_window()

    # 3. Retrieve the picked points
    picked_indices = vis.get_picked_points()
    
    if len(picked_indices) >= 2:
        print("\n" + "="*30)
        print("PICKED POINTS DATA:")
        for i, idx in enumerate(picked_indices):
            point = xyz[idx]
            print(f"Point {i+1} (Index {idx}):")
            print(f"  X: {point[0]:.6f}")
            print(f"  Y: {point[1]:.6f}")
            print(f"  Z: {point[2]:.6f}")
        
        # Calculate Distance between the two points automatically
        p1 = xyz[picked_indices[0]]
        p2 = xyz[picked_indices[1]]
        dist = np.linalg.norm(p1 - p2)
        print(f"\nDistance between P1 and P2: {dist:.6f} units")
        print("="*30)
    else:
        print("\n[!] You didn't pick at least 2 points.")

if __name__ == "__main__":
    # Path to your cleaned satellite CSV
   # csv_file = r"logs\data\unscaled_satellite.csv"
    csv_file = r"logs\data\scaled_satellite.csv"
    
    pick_points(csv_file)