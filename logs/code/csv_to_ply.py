import open3d as o3d
import numpy as np

# --- 1. Load CSV ---
csv_file = r"logs\data\scaled_satellite.csv"
data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)

xyz = data[:, 0:3]
rgb = data[:, 3:6] / 255.0  # Normalize RGB for Open3D

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

# --- 2. Translate to Origin ---
centroid = np.mean(xyz, axis=0)
# By translating by the negative centroid, we shift the entire cloud 
# so its center rests perfectly at (0, 0, 0)
pcd.translate(-centroid)

print(f"Original Centroid: {centroid}")
print("Point cloud translated to origin.")

# --- 3. Rotate -90 degrees around -Y axis ---
# Open3D expects angles in radians. -90 degrees is -pi / 2
# The rotation matrix expects (rx, ry, rz)
R = pcd.get_rotation_matrix_from_xyz((0, -np.pi / 2, 0))

# CRITICAL: We set center=(0,0,0) so it rotates exactly in place. 
pcd.rotate(R, center=(0, 0, 0))
print("Point cloud rotated -90 degrees around the Y-axis.")

# --- 4. Save to PLY ---
# PLY is a fantastic format for this because it natively stores the RGB colors 
# alongside the XYZ coordinates without needing custom parsing.
output_file = r"logs\data\scaled_satellite.ply"
o3d.io.write_point_cloud(output_file, pcd)

print(f"Success! Transformed cloud saved to: {output_file}")

# --- Optional: Visualize to verify ---
# Uncomment the lines below if you want to see the final result instantly
# origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.5, origin=[0,0,0])
# o3d.visualization.draw_geometries([pcd, origin_frame], window_name="Centered & Rotated Satellite")