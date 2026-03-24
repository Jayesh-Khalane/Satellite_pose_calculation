import open3d as o3d
import numpy as np
import pandas as pd
import os

def export_dbscan_to_csv(input_csv, output_csv, eps=0.5, min_points=50):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    # 1. Load the High-Res Data
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    xyz = df[['X', 'Y', 'Z']].values
    rgb = df[['R', 'G', 'B']].values # Keep as 0-255 for now

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # Open3D needs colors as 0-1 floats for the algorithm
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    # 2. Run DBSCAN Clustering
    print(f"Clustering points (eps={eps}, min_samples={min_points})...")
    # This identifies dense groups of points
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    # 3. Process Labels
    max_label = labels.max()
    if max_label < 0:
        print("Optimization Failed: No clusters found. Increase 'eps'.")
        return

    # Find the largest cluster index (ignoring -1 noise)
    counts = np.bincount(labels[labels >= 0])
    target_label = np.argmax(counts)
    
    # 4. Filter the Data
    # Mask to keep only the satellite points
    mask = (labels == target_label)
    satellite_xyz = xyz[mask]
    satellite_rgb = rgb[mask]

    # 5. Save to CSV
    print(f"Exporting {len(satellite_xyz)} points to {output_csv}...")
    output_df = pd.DataFrame(
        np.hstack((satellite_xyz, satellite_rgb)), 
        columns=['X', 'Y', 'Z', 'R', 'G', 'B']
    )
    
    # Save with high precision for X,Y,Z and integers for R,G,B
    output_df.to_csv(output_csv, index=False, float_format='%.6f')

    # --- FINAL SUMMARY ---
    total_pts = len(xyz)
    clean_pts = len(satellite_xyz)
    noise_pts = total_pts - clean_pts
    print("-" * 30)
    print(f"Original Points: {total_pts}")
    print(f"Satellite Points: {clean_pts}")
    print(f"Noise Removed: {noise_pts} ({ (noise_pts/total_pts)*100:.1f}%)")
    print("-" * 30)

    # Quick visualization check of the SAVED data
    print("Opening preview of the saved CSV...")
    clean_pcd = o3d.geometry.PointCloud()
    clean_pcd.points = o3d.utility.Vector3dVector(satellite_xyz)
    clean_pcd.colors = o3d.utility.Vector3dVector(satellite_rgb / 255.0)
    o3d.visualization.draw_geometries([clean_pcd], window_name="Final Export Preview")

if __name__ == "__main__":
    # Input: Your raw high-res COLMAP output
    file_in = r"logs\data\sat.csv"
    
    # Output: The clean, isolated satellite
    file_out = r"logs\data\unscaled_satellite.csv"
    
    export_dbscan_to_csv(file_in, file_out, eps=0.1, min_points=3)