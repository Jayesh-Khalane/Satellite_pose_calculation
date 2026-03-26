import pyvista as pv
import numpy as np

# --- Load CSV ---
csv_file = r"logs\data\sat.csv"
data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)

xyz = data[:, 0:3]
rgb = data[:, 3:6].astype(np.uint8)

# --- Create PyVista point cloud ---
point_cloud = pv.PolyData(xyz)
point_cloud["RGB"] = rgb

# --- Plot ---
plotter = pv.Plotter()
plotter.add_points(point_cloud, scalars="RGB", rgb=True, point_size=2)

plotter.show()


# import open3d as o3d
# import numpy as np

# # --- Load CSV ---
# csv_file = r"logs\data\scaled_satellite.csv"
# data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)

# xyz = data[:, 0:3]
# # Open3D expects colors in the [0, 1] range
# rgb = data[:, 3:6] / 255.0 

# # --- Create Open3D point cloud ---
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)
# pcd.colors = o3d.utility.Vector3dVector(rgb)

# # --- Calculate Centroid and Distance ---
# centroid = np.mean(xyz, axis=0)
# origin = np.array([0.0, 0.0, 0.0])
# distance = np.linalg.norm(centroid - origin)

# print(f"--- Translation Data ---")
# print(f"Centroid (x, y, z): {centroid}")
# print(f"Distance from origin: {distance:.4f} units")

# # --- Create Visual Markers ---
# # 1. Orientation marker at the origin (Scaled up 5x to 2.5)
# origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.5, origin=origin)

# # 2. A sphere to mark the centroid clearly
# centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
# centroid_sphere.translate(centroid)
# centroid_sphere.paint_uniform_color([1.0, 0.0, 0.0]) # Solid red

# # 3. A line connecting the origin to the centroid
# line_points = [origin, centroid]
# line_indices = [[0, 1]]
# line_set = o3d.geometry.LineSet(
#     points=o3d.utility.Vector3dVector(line_points),
#     lines=o3d.utility.Vector2iVector(line_indices)
# )
# # Changed to white [1.0, 1.0, 1.0] so it pops on the black background
# line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 1.0]]) 

# # --- Advanced Plotting in Open3D ---
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Satellite Translation Vector")

# # Add all our geometries
# vis.add_geometry(pcd)
# vis.add_geometry(origin_frame)
# vis.add_geometry(centroid_sphere)
# vis.add_geometry(line_set)

# # Access render options to change background and point size
# opt = vis.get_render_option()
# opt.background_color = np.asarray([0, 0, 0]) # Black background
# opt.point_size = 1.0 # Smallest point size

# vis.run()
# vis.destroy_window()