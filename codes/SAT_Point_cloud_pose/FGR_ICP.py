import open3d as o3d
import numpy as np
import copy

# Clean printing for the matrix
np.set_printoptions(suppress=True, precision=6)

def draw_registration_result(source, target, transformation, title):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])      # Yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929]) # Blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=title)

def preprocess_point_cloud(pcd, voxel_size):
    # For <10k points, we keep it very dense
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # Fixed the variable name here to match the return statement
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, fpfh

# 1. Load Data
source = o3d.io.read_point_cloud(r"D:\Satellite_pose_calculation\logs\data\to_align_half_sat.ply")
target = o3d.io.read_point_cloud(r"D:\Satellite_pose_calculation\logs\data\reference_sat.ply")

if source.is_empty() or target.is_empty():
    print("Error: Files not found or empty.")
    exit()

# 2. Setup Voxel Size
# If your points are very close together, you can even try 0.01
voxel_size = 0.1 

print("Step 1: Computing Fast Global Registration (FGR)...")
s_down, s_fpfh = preprocess_point_cloud(source, voxel_size)
t_down, t_fpfh = preprocess_point_cloud(target, voxel_size)

# --- FAST GLOBAL REGISTRATION ---
result_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    s_down, t_down, s_fpfh, t_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=voxel_size * 1.5))

print("FGR Global alignment finished.")

# --- ICP REFINEMENT ---
print("Step 2: Applying Point-to-Plane ICP (Fine Refinement)...")

# ICP requires normals on the original clouds to be highly accurate
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

# We use the FGR result as the 'seed' for ICP
result_icp = o3d.pipelines.registration.registration_icp(
    source, target, voxel_size * 2, result_fgr.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())

# 3. Final Outputs
print("\n" + "="*50)
print("FINAL TRANSFORMATION MATRIX")
print("="*50)
print(result_icp.transformation)
print("="*50)

# Show the results
draw_registration_result(source, target, result_icp.transformation, "Final  Alignment")