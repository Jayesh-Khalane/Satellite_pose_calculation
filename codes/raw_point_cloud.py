import pyzed.sl as sl
import numpy as np
import csv

# --- Initialize ZED camera ---
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER
init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera:", status)
    exit(1)

print("Camera opened successfully!")


runtime_params = sl.RuntimeParameters()
zed.grab(runtime_params)

# --- Retrieve point cloud ---
point_cloud = sl.Mat()
zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

# Get data as numpy array: shape (720, 1280, 4)
xyzrgba = point_cloud.get_data()
h, w, _ = xyzrgba.shape
xyzrgba = xyzrgba.reshape(-1, 4)

# --- Vectorized Filtering and Unpacking ---

# 1. Filter out all rows containing any NaN values
# (Fastest way to clean the point cloud)

valid_points = xyzrgba

# 2. Extract XYZ
x = valid_points[:, 0]
y = valid_points[:, 1]
z = valid_points[:, 2]

# 3. Unpack ABGR from the 4th column (float32)
# We view the raw bits as uint32 to perform bitwise operations
rgba_bits = valid_points[:, 3].view(np.uint32)

# For ABGR format:
# Red is at the lowest 8 bits, Green middle, Blue high.
r = (rgba_bits >> 0) & 0xFF
g = (rgba_bits >> 8) & 0xFF
b = (rgba_bits >> 16) & 0xFF
# a = (rgba_bits >> 24) & 0xFF (Alpha is usually 255/ignored for CSV)

# 4. Combine into final rows
rows = np.column_stack((x, y, z, r, g, b))

# --- Save CSV ---
csv_file = "zed_pointcloud.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["X", "Y", "Z", "R", "G", "B"])
    writer.writerows(rows)

print(f"Saved {len(rows)} valid points to {csv_file}")
zed.close()