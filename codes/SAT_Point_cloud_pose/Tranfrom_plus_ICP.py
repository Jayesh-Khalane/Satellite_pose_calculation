import open3d as o3d
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R_tool
import os
import copy

def main():
    pc_file = "sat_point_cloud_capture.csv"
    pose_file = "capture_poses.csv"
    output_file = "ICP_result.csv"

    if not os.path.exists(pc_file) or not os.path.exists(pose_file):
        print("Error: Missing CSV files.")
        return

    # 1. Load Data
    poses_df = pd.read_csv(pose_file)
    df = pd.read_csv(pc_file)
    
    # Voxel size (0.1cm as per your request)
    voxel_size = 0.1

    pcds = []
    for i in range(3):
        mask = (df['R'] > 127 if i==0 else df['R'] < 127) & \
               (df['G'] > 127 if i==1 else df['G'] < 127) & \
               (df['B'] > 127 if i==2 else df['B'] < 127)
        
        subset = df[mask]
        if not subset.empty:
            p = o3d.geometry.PointCloud()
            p.points = o3d.utility.Vector3dVector(subset[['X', 'Y', 'Z']].values)
            p.colors = o3d.utility.Vector3dVector(subset[['R', 'G', 'B']].values / 255.0)
            
            # Apply initial pose
            row = poses_df.iloc[i]
            T_init = np.eye(4)
            T_init[:3, :3] = R_tool.from_euler('xyz', [row.Roll, row.Pitch, row.Yaw], degrees=True).as_matrix()
            T_init[:3, 3] = [row.X_cm, row.Y_cm, row.Z_cm]
            p.transform(T_init)
            
            # Clean up the cloud
            p = p.voxel_down_sample(voxel_size)
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            pcds.append(p)

    if len(pcds) < 2: 
        print("Not enough clouds to align.")
        return

    base = pcds[0]
    final_geoms = [base]

    for i in range(1, len(pcds)):
        source = pcds[i]
        
        # --- ROBUST GLOBAL REGISTRATION (RANSAC) ---
        print(f"Global aligning Capture {i+1}...")
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, base, 
            o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)),
            o3d.pipelines.registration.compute_fpfh_feature(base, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)),
            True, voxel_size * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        # --- LOCAL REFINEMENT (ICP) ---
        print(f"Final Polishing Capture {i+1}...")
        reg_refined = o3d.pipelines.registration.registration_icp(
            source, base, voxel_size, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        source.transform(reg_refined.transformation)
        final_geoms.append(source)

    # 2. SAVE THE RESULT TO CSV
    print(f"Exporting results to {output_file}...")
    all_points = []
    all_colors = []

    for geom in final_geoms:
        all_points.append(np.asarray(geom.points))
        # Convert colors back to 0-255 scale
        all_colors.append(np.asarray(geom.colors) * 255.0)

    export_xyz = np.vstack(all_points)
    export_rgb = np.vstack(all_colors)
    export_data = np.hstack((export_xyz, export_rgb))

    # Save using pandas for clean headers
    result_df = pd.DataFrame(export_data, columns=['X', 'Y', 'Z', 'R', 'G', 'B'])
    result_df.to_csv(output_file, index=False)
    print("Export Complete.")

    # 3. VISUALIZE
    o3d.visualization.draw_geometries(final_geoms)

if __name__ == "__main__":
    main()