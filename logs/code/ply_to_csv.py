import open3d as o3d
import numpy as np
import os
import pandas as pd

def convert_ply_to_csv(ply_path, csv_path):
    # 1. Check if the source PLY exists
    if not os.path.exists(ply_path):
        print(f"Error: {ply_path} not found.")
        return

    print(f"Reading PLY: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)

    if pcd.is_empty():
        print("Error: Point cloud is empty.")
        return

    # 2. Extract Coordinates (X, Y, Z)
    # These are already in float64
    xyz = np.asarray(pcd.points)

    # 3. Extract Colors (R, G, B)
    # Open3D stores colors as floats [0.0 - 1.0]. 
    # We convert them back to integers [0 - 255] for your CSV standard.
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        print("Warning: No colors found. Setting default to white.")
        rgb = np.full(xyz.shape, 255, dtype=np.uint8)

    # 4. Combine into a single Matrix [X, Y, Z, R, G, B]
    full_data = np.hstack((xyz, rgb))

    # 5. Save using Pandas (Cleaner and handles headers better)
    print(f"Saving to CSV: {csv_path}...")
    df = pd.DataFrame(full_data, columns=['X', 'Y', 'Z', 'R', 'G', 'B'])
    
    # Using float precision for XYZ and integers for RGB
    df.to_csv(csv_path, index=False, float_format='%.6f')
    
    print(f"Successfully converted {len(df)} points.")

if __name__ == "__main__":
    # Update these paths to match your project structure
    input_ply = r"logs\data\sat.ply"
    output_csv = r"logs\data\sat.csv"
    
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    convert_ply_to_csv(input_ply, output_csv)