import pyzed.sl as sl
import numpy as np
import csv

# --- Initialize ZED camera ---
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER
# Setting the coordinate system as requested
init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera:", status)
    exit(1)

print("Camera opened successfully!")

runtime_params = sl.RuntimeParameters()

# Grab a frame
if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    # --- Retrieve point cloud ---
    point_cloud = sl.Mat()
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

    # Get data as numpy array: shape (720, 1280, 4)
    data = point_cloud.get_data()
    height, width, _ = data.shape

    csv_file = "pixel_to_pointcloud.csv"
    print(f"Processing {width}x{height} image to CSV...")

    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Header: U, V are pixel coordinates; X, Y, Z are spatial; R, G, B are color
        writer.writerow(["Pixel_U", "Pixel_V", "X", "Y", "Z", "R", "G", "B"])

        # We iterate by row (v) and column (u) to maintain the image structure
        for v in range(height):
            for u in range(width):
                point = data[v, u]
                
                # Check for valid depth (not NaN or Inf)
                if np.isnan(point[0]) or np.isinf(point[0]):
                    continue

                # Extract XYZ
                x, y, z = point[0], point[1], point[2]

                # Unpack RGB from the 4th float element
                # ZED stores color as a single float bit-packed as ABGR
                rgba_bits = point[3].view(np.uint32)
                r = (rgba_bits >> 0) & 0xFF
                g = (rgba_bits >> 8) & 0xFF
                b = (rgba_bits >> 16) & 0xFF

                writer.writerow([u, v, x, y, z, r, g, b])

    print(f"Successfully saved data to {csv_file}")
else:
    print("Could not grab frame from ZED.")

zed.close()