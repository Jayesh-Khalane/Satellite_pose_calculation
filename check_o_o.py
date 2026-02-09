import pyzed.sl as sl
import numpy as np
import cv2

# --- Initialize ZED camera ---
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_units = sl.UNIT.METER
init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera:", status)
    exit(1)

runtime_params = sl.RuntimeParameters()

# Camera resolution
h, w = 720, 1280

# Target point index in your flattened CSV
point_index = 485770 - 1
target_row = point_index // w
target_col = point_index % w
print(f"Target pixel coordinates: row={target_row}, col={target_col}")

# --- Display loop ---
window_name = "ZED Live with Dot"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

image = sl.Mat()
point_cloud = sl.Mat()

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # Get left camera image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()  # shape: (720,1280,4), RGBA
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Get point cloud
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        xyzrgba = point_cloud.get_data().reshape(-1, 4)

        # --- Unpack color from ABGR float32 ---
        rgba_bits = np.array([xyzrgba[point_index, 3]], dtype=np.float32).view(np.uint32)[0]
        r = (rgba_bits >> 0) & 0xFF
        g = (rgba_bits >> 8) & 0xFF
        b = (rgba_bits >> 16) & 0xFF

        # OpenCV uses BGR
        color = (int(b), int(g), int(r))

        # Draw a dot
        cv2.circle(frame_bgr, (target_col, target_row), 5, color, -1)

        cv2.imshow(window_name, frame_bgr)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

zed.close()
cv2.destroyAllWindows()
