import pyvista as pv
import numpy as np
from sklearn.cluster import DBSCAN

# -----------------------------
ply_file = r"D:\Satellite_pose_calculation\data\MVS_dataset\002\002.ply"
eps = 0.5
min_samples = 25
# -----------------------------

# Load point cloud
cloud = pv.read(ply_file)
points = cloud.points

# -----------------------------
# Step 1: DBSCAN to remove outliers
# -----------------------------
clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
labels = clustering.labels_

valid_labels = labels[labels >= 0]
if len(valid_labels) == 0:
    print("No clusters found. Increase eps.")
    exit()

# Keep largest cluster
unique, counts = np.unique(valid_labels, return_counts=True)
largest_label = unique[np.argmax(counts)]
mask = labels == largest_label
filtered_points = points[mask]
filtered_cloud = pv.PolyData(filtered_points)

# -----------------------------
# Step 2: Extract outer surface
# -----------------------------
# delaunay_3d creates a volumetric mesh
volume_mesh = filtered_cloud.delaunay_3d(alpha=0.2)  # alpha controls how tight the surface is
# extract_geometry extracts the surface edges and vertices
surface_wireframe = volume_mesh.extract_geometry()

# -----------------------------
# Step 3: Visualization
# -----------------------------
plotter = pv.Plotter()
# Show surface wireframe
plotter.add_mesh(surface_wireframe, color="white", line_width=2, render_points_as_spheres=True, point_size=4)
plotter.set_background("black")
plotter.show()
