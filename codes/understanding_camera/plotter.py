import pyvista as pv
import numpy as np

# --- Load CSV ---
csv_file = "zed_pointcloud.csv"
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
