import pyvista as pv
import numpy as np
from sklearn.cluster import DBSCAN

# -----------------------------
#ply_file =r"D:\SPEED_dataset\Speed.ply"
ply_file = r"D:\Satellite_pose_calculation\data\MVS_dataset\002\002.ply"
eps = 0.5                                               #0.5
min_samples = 25                                        #25

SHOW_MESH = False   #TOGGLE FOR MESH 
# -----------------------------

# -----------------------------
# Load point cloud
# -----------------------------
cloud = pv.read(ply_file)
points = cloud.points

# Try to detect RGB data
rgb_data = None

if "RGB" in cloud.point_data:
    rgb_data = cloud.point_data["RGB"]

elif all(k in cloud.point_data for k in ["red", "green", "blue"]):
    r = cloud.point_data["red"]
    g = cloud.point_data["green"]
    b = cloud.point_data["blue"]
    rgb_data = np.column_stack((r, g, b))

elif "rgba" in cloud.point_data:
    rgb_data = cloud.point_data["rgba"][:, :3]

# -----------------------------
# Step 1: DBSCAN to remove outliers
# -----------------------------
clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
labels = clustering.labels_

valid_labels = labels[labels >= 0]
if len(valid_labels) == 0:
    print("No clusters found. Increase eps.")
    exit()

unique, counts = np.unique(valid_labels, return_counts=True)
largest_label = unique[np.argmax(counts)]

mask = labels == largest_label
filtered_points = points[mask]
filtered_cloud = pv.PolyData(filtered_points)

# Attach filtered RGB if available
if rgb_data is not None:
    filtered_rgb = rgb_data[mask]
    filtered_cloud["RGB"] = filtered_rgb

# -----------------------------
# Visualization
# -----------------------------
plotter = pv.Plotter()

if SHOW_MESH:
    volume_mesh = filtered_cloud.delaunay_3d(alpha=0.2)
    surface_wireframe = volume_mesh.extract_geometry()

    plotter.add_mesh(surface_wireframe,
                     scalars="RGB",
                         rgb=True,
                         render_points_as_spheres=True,
                         point_size=2,
                     line_width=2)
else:
    if rgb_data is not None:
        plotter.add_mesh(filtered_cloud,
                         scalars="RGB",
                         rgb=True,
                         render_points_as_spheres=True,
                         point_size=2)
  

plotter.set_background("black")
plotter.show()
