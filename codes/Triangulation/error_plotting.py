import cv2
import numpy as np
import pyzed.sl as sl
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

INTENSITY_THRESH = 750
MIN_BLOB_AREA = 20
DISPARITY_BUFFER_SIZE = 15   # median over last N frames
DEPTH_HISTORY_SIZE = 200     # for graph

triangulate_active = False
disp_buffer = deque(maxlen=DISPARITY_BUFFER_SIZE)
depth_history = deque(maxlen=DEPTH_HISTORY_SIZE)
raw_depth_history = deque(maxlen=DEPTH_HISTORY_SIZE)

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

def draw_depth_graph(depth_hist, raw_hist, width=600, height=200):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if len(depth_hist) < 2:
        return canvas

    raw_arr = np.array(raw_hist)
    smooth_arr = np.array(depth_hist)
    all_vals = np.concatenate([raw_arr, smooth_arr])
    mn, mx = all_vals.min() - 5, all_vals.max() + 5
    rng = mx - mn if mx != mn else 1.0

    def to_px(val):
        return int(height - 1 - ((val - mn) / rng) * (height - 1))

    def x_px(i, total):
        return int(i / max(total - 1, 1) * (width - 1))

    # Draw raw depth (dim red)
    for i in range(1, len(raw_arr)):
        x1, x2 = x_px(i - 1, len(raw_arr)), x_px(i, len(raw_arr))
        y1, y2 = to_px(raw_arr[i - 1]), to_px(raw_arr[i])
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 120), 1, cv2.LINE_AA)

    # Draw smoothed depth (bright green)
    for i in range(1, len(smooth_arr)):
        x1, x2 = x_px(i - 1, len(smooth_arr)), x_px(i, len(smooth_arr))
        y1, y2 = to_px(smooth_arr[i - 1]), to_px(smooth_arr[i])
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

    # Labels
    cv2.putText(canvas, f"max:{mx-5:.1f}cm", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
    cv2.putText(canvas, f"min:{mn+5:.1f}cm", (5, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
    cv2.putText(canvas, "GREEN=smoothed  RED=raw", (width - 220, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

    return canvas

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
    cv2.namedWindow("Depth Graph", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Graph", 600, 200)

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

        l_centroid, _ = get_largest_blob(left_bw)
        r_centroid, _ = get_largest_blob(right_bw)

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
            raw_disparity = uL - uR

            if raw_disparity > 0:
                # Raw depth
                raw_depth_cm = (FX * BASELINE * 100.0) / raw_disparity

                # Push disparity into buffer, compute median
                disp_buffer.append(raw_disparity)
                smooth_disparity = np.median(disp_buffer)
                smooth_depth_cm = (FX * BASELINE * 100.0) / smooth_disparity

                depth_history.append(smooth_depth_cm)
                raw_depth_history.append(raw_depth_cm)

                # Epipolar line
                row = l_centroid[1]
                cv2.line(left_disp,  (0, row), (left_disp.shape[1], row),  (0, 255, 0), 1, cv2.LINE_AA)
                cv2.line(right_disp, (0, row), (right_disp.shape[1], row), (0, 255, 0), 1, cv2.LINE_AA)

                cv2.putText(left_disp,
                            f"RAW:  {raw_depth_cm:.1f} cm",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(left_disp,
                            f"SMOOTH: {smooth_depth_cm:.1f} cm (N={len(disp_buffer)})",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                print(f"uL={uL}  uR={uR}  raw_disp={raw_disparity}  smooth_disp={smooth_disparity:.1f}  "
                      f"raw={raw_depth_cm:.2f}cm  smooth={smooth_depth_cm:.2f}cm")

            else:
                cv2.putText(left_disp, "BAD DISPARITY (uL <= uR)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        status = "T: ON  (ESC quit)" if triangulate_active else "T: OFF  (press T)"
        cv2.putText(left_disp, status, (20, left_disp.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        graph = draw_depth_graph(depth_history, raw_depth_history)

        cv2.imshow("Left BW", left_disp)
        cv2.imshow("Right BW", right_disp)
        cv2.imshow("Depth Graph", graph)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('t') or key == ord('T'):
            triangulate_active = not triangulate_active
            disp_buffer.clear()
            print(f"Triangulation: {'ON' if triangulate_active else 'OFF'}")
        elif key == 27:
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()