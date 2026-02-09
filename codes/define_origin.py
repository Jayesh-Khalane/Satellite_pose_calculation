import cv2 as cv
import pyzed.sl as sl
import numpy as np

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
    elif event == cv.EVENT_RBUTTONDOWN:
        clicked_points.clear()

def main():

    # ---------- ZED INIT ----------
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        return

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    point_cloud = sl.Mat()

    cv.namedWindow("ZED View", cv.WINDOW_NORMAL)
    cv.setMouseCallback("ZED View", mouse_callback)

    origin_set = False
    origin = (0.0, 0.0, 0.0)

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

        frame = image.get_data()
        if frame.shape[2] == 4:
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        h, w, _ = frame.shape

        # ---------- Center pixel based on resolution ----------
        if w == 1280 and h == 720:          # HD720
            center_pixel = (649, 379)
        elif w == 1920 and h == 1080:       # HD1080
            center_pixel = (977, 578)
        elif w == 2208 and h == 1242:       # 2K
            center_pixel = (1123, 659)
        else:
            center_pixel = (w // 2, h // 2)

        cx, cy = center_pixel
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))

        # ---------- Set origin using center pixel ----------
        err, p = point_cloud.get_value(cx, cy)
        if err == sl.ERROR_CODE.SUCCESS and np.isfinite(p[2]):
            Xc, Yc, Zc = float(p[0]), float(p[1]), float(p[2])
            origin = (Xc, Yc, Zc / 2.0)
            origin_set = True

            cv.drawMarker(frame, (cx, cy), (0, 255, 0), cv.MARKER_CROSS, 25, 1)

        # ---------- Clicked points ----------
        for (u, v) in clicked_points:
            u = int(np.clip(u, 0, w - 1))
            v = int(np.clip(v, 0, h - 1))

            err, p = point_cloud.get_value(u, v)
            if err != sl.ERROR_CODE.SUCCESS or not np.isfinite(p[2]):
                continue

            X, Y, Z = float(p[0]), float(p[1]), float(p[2])

            # Relative to new origin
            ox, oy, oz = origin
            relX = X - ox
            relY = Y - oy
            relZ = Z - oz

            # Euclidean distance from origin
            dist = np.sqrt(relX**2 + relY**2 + relZ**2)

            cv.drawMarker(frame, (u, v), (0, 255, 0), cv.MARKER_CROSS, 15, 1)

            label = f"Rel:({relX:.2f},{relY:.2f},{relZ:.2f}) Dist:{dist:.2f}m"

            cv.putText(frame, label, (u + 5, v - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv.imshow("ZED View", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    zed.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
