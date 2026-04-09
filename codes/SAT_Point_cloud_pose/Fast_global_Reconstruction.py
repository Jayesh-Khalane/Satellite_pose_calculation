import open3d as o3d
import numpy as np
import copy

# Clean printing for the matrix
np.set_printoptions(suppress=True, precision=6)

def draw_registration_result(source, target, transformation, title="Point Cloud View"):
    """Helper to visualize the point clouds."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    # Paint for clarity: Yellow = Moving (Source), Blue = Static (Target)
    source_temp.paint_uniform_color([1, 0.706, 0])      
    target_temp.paint_uniform_color([0, 0.651, 0.929]) 
    
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=title)

def preprocess_point_cloud(pcd, voxel_size):
    """Downsamples and computes FPFH features required for FGR."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH features
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# 1. Load the Point Clouds
source_path = r"D:\Satellite_pose_calculation\logs\data\to_align_sat.ply"
target_path = r"D:\Satellite_pose_calculation\logs\data\reference_sat.ply"

source = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)

if source.is_empty() or target.is_empty():
    print("Error: Could not load files. Please check the paths.")
    exit()

# 2. Show Initial Unaligned State
print("Showing INITIAL state. Close this window to run Fast Global Reconstruction...")
draw_registration_result(source, target, np.identity(4), "Initial Unaligned State")

# 3. Parameters for FGR
# Adjust voxel_size based on your satellite scale (e.g., 0.5, 1.0, or 2.0)
voxel_size = 1.0 
distance_threshold = voxel_size * 1.5

print(f"Preprocessing with voxel_size {voxel_size}...")
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

# 4. Apply Fast Global Registration (Zhou et al.)
print("Computing Fast Global Registration...")
result_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=distance_threshold,
        iteration_number=64))

# 5. Output the Results
print("\n" + "="*50)
print("FAST GLOBAL RECONSTRUCTION RESULTS")
print("="*50)
print("Transformation Matrix:")
print(result_fgr.transformation)
print("="*50)

draw_registration_result(source, target, result_fgr.transformation, "FGR Alignment Result")