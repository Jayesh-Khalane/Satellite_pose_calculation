import cv2 as cv
import pyzed.sl as sl
import numpy as np

# Global variables
pc_data_global = None
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
    global pc_data_global, points

    # Initialize ZED 2i
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 15
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED 2i open failed")
        return

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    pc = sl.Mat()

    cv.namedWindow("ZED Viewer")
    cv.setMouseCallback("ZED Viewer", mouse_callback)

    print("Left click to get 3D coordinates. Right click to reset points. ESC to exit.")

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        # Retrieve image and point cloud
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(pc, sl.MEASURE.XYZ)

        frame = image.get_data()
        if frame.shape[2] == 4:
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        pc_data_global = pc.get_data()

        # Draw center crosshair
        h, w = frame.shape[:2]
        draw_crosshair(frame, w//2, h//2)

        # Draw POIs and show 3D coordinates
        for i, (px, py) in enumerate(points):
            if 0 <= py < pc_data_global.shape[0] and 0 <= px < pc_data_global.shape[1]:
                Xp, Yp, Zp, _ = pc_data_global[py, px]
                if np.isfinite(Xp + Yp + Zp):
                    label = f"P{i+1}: X={Xp:.2f}, Y={Yp:.2f}, Z={Zp:.2f} m"
                    cv.drawMarker(frame, (px, py), (0,255,0), cv.MARKER_CROSS, 20, 2)
                    cv.putText(frame, label, (px + 10, py - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                else:
                    cv.putText(frame, f"P{i+1}: Invalid depth", (px + 10, py - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        cv.imshow("ZED Viewer", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

    zed.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
