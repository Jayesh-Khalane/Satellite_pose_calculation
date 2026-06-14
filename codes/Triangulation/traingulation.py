import cv2
import numpy as np
import pyzed.sl as sl

# -----------------------------
# ZED 2i Intrinsics (HD1080)
# -----------------------------
FX = 1064.5
FY = 1064.5
CX_L = 1104.8
CY_L = 621.6
CX_R = 1104.8
CY_R = 621.6
BASELINE = 0.12  # meters (120mm)

INTENSITY_THRESH = 755
MIN_BLOB_AREA = 20

triangulate_active = False

def get_largest_blob(binary_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    best_area = 0
    best_centroid = None
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_BLOB_AREA and area > best_area:
            best_area = area
            best_centroid = (int(centroids[i][0]), int(centroids[i][1]))
    return best_centroid, best_area

def build_bw_mask(frame):
    intensity = np.sum(frame[:, :, :3].astype(np.int32), axis=2)
    binary = (intensity > INTENSITY_THRESH).astype(np.uint8) * 255
    return binary

def main():
    global triangulate_active

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return

    left_mat = sl.Mat()
    right_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    cv2.namedWindow("Left BW", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right BW", cv2.WINDOW_NORMAL)

    while True:
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(left_mat, sl.VIEW.LEFT)
        zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

        left_frame = left_mat.get_data()
        right_frame = right_mat.get_data()

        left_bw = build_bw_mask(left_frame)
        right_bw = build_bw_mask(right_frame)

        left_disp = cv2.cvtColor(left_bw, cv2.COLOR_GRAY2BGR)
        right_disp = cv2.cvtColor(right_bw, cv2.COLOR_GRAY2BGR)

        l_centroid, l_area = get_largest_blob(left_bw)
        r_centroid, r_area = get_largest_blob(right_bw)

        if l_centroid:
            cv2.circle(left_disp, l_centroid, 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(left_disp, f"L:({l_centroid[0]},{l_centroid[1]})",
                        (l_centroid[0] + 8, l_centroid[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if r_centroid:
            cv2.circle(right_disp, r_centroid, 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(right_disp, f"R:({r_centroid[0]},{r_centroid[1]})",
                        (r_centroid[0] + 8, r_centroid[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if triangulate_active and l_centroid and r_centroid:
            uL = l_centroid[0]
            uR = r_centroid[0]
            disparity = uL - uR

            if disparity > 0:
                depth_m = (FX * BASELINE) / disparity
                depth_cm = depth_m * 100.0

                label = f"DEPTH: {depth_cm:.1f} cm  (disp={disparity:.1f}px)"

                row = l_centroid[1]
                cv2.line(left_disp,  (0, row), (left_disp.shape[1], row),  (0, 255, 0), 1, cv2.LINE_AA)
                cv2.line(right_disp, (0, row), (right_disp.shape[1], row), (0, 255, 0), 1, cv2.LINE_AA)

                cv2.putText(left_disp, label,
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)

                print(f"uL={uL}  uR={uR}  disp={disparity:.1f}px  depth={depth_cm:.2f} cm")
            else:
                cv2.putText(left_disp, "BAD DISPARITY (uL <= uR)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)

        status = "T: ON" if triangulate_active else "T: OFF  (press T)"
        cv2.putText(left_disp, status, (20, left_disp.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Left BW", left_disp)
        cv2.imshow("Right BW", right_disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('t') or key == ord('T'):
            triangulate_active = not triangulate_active
            print(f"Triangulation: {'ON' if triangulate_active else 'OFF'}")
        elif key == 27:
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()