import cv2 as cv
import pyzed.sl as sl
import numpy as np

# Global variables
points = []  # List of points: (x_pixel, y_pixel)

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Added point at pixel: {(x, y)}")
    elif event == cv.EVENT_RBUTTONDOWN:
        points.clear()
        print("All points cleared")

def draw_crosshair(img, cx, cy, color=(0,0,255), size=15, thickness=2):
    cv.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv.line(img, (cx, cy - size), (cx, cy + size), color, thickness)

def main():
    global points

    # Initialize ZED 2i
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Better for precise 3D coords
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE # Standard for CV

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED 2i open failed. Check connection.")
        return

    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()
    point_cloud = sl.Mat()

    cv.namedWindow("ZED Viewer")
    cv.setMouseCallback("ZED Viewer", mouse_callback)

    print("--- Controls ---")
    print("Left Click: Get 3D Coords | Right Click: Clear | ESC: Exit")

    while True:
        # Grab a new frame
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # Retrieve RGB image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            # Retrieve Point Cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # Convert ZED Mat to OpenCV format
            frame = image_zed.get_data()
            # ZED provides BGRA, OpenCV usually uses BGR for display
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

            # Draw center crosshair for reference
            h, w = frame.shape[:2]
            draw_crosshair(frame, w//2, h//2)

            # Process clicked points
            for i, (px, py) in enumerate(points):
                # Ensure coordinates are within image bounds
                if 0 <= px < w and 0 <= py < h:
                    # Get 3D value at the specific pixel
                    err, point_cloud_value = point_cloud.get_value(px, py)
                    
                    # point_cloud_value is a float4: [X, Y, Z, W]
                    x_3d = point_cloud_value[0]
                    y_3d = point_cloud_value[1]
                    z_3d = point_cloud_value[2]

                    if np.isfinite(x_3d) and np.isfinite(y_3d) and np.isfinite(z_3d):
                        label = f"P{i+1}: X={x_3d:.2f}, Y={y_3d:.2f}, Z={z_3d:.2f}m"
                        color = (0, 255, 0)
                    else:
                        label = f"P{i+1}: Out of Range"
                        color = (0, 0, 255)

                    cv.drawMarker(frame, (px, py), color, cv.MARKER_CROSS, 20, 2)
                    cv.putText(frame, label, (px + 10, py - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv.imshow("ZED Viewer", frame)
            
        key = cv.waitKey(1)
        if key == 27: # ESC
            break

    zed.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()