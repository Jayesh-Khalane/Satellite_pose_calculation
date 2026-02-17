import cv2 as cv
import numpy as np
import pyzed.sl as sl
import itertools
import math

# -----------------------------
# Parameters
# -----------------------------
GRAY_THRESHOLD = 253   # LEDs brighter than this
N_LEDS = 3             # LEDs per marker
MIN_BLOB_SIZE = 10      # ignore tiny blobs
NEIGHBORHOOD = 50      # for averaging 3D position

# -----------------------------
# Helper functions
# -----------------------------
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def find_led_blobs(gray):
    """Detect LED blobs using threshold + connected components"""
    _, binary = cv.threshold(gray, GRAY_THRESHOLD, 255, cv.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary)
    led_pixels = []
    for i in range(1, num_labels):  # skip background
        if stats[i, cv.CC_STAT_AREA] >= MIN_BLOB_SIZE:
            cx, cy = centroids[i]
            led_pixels.append((int(cx), int(cy)))
    return led_pixels

def get_3d_position(pc_data, px, py, neighborhood=NEIGHBORHOOD):
    """Average 3D coordinates around a pixel"""
    h, w, _ = pc_data.shape
    x_min = max(px - neighborhood//2, 0)
    x_max = min(px + neighborhood//2 + 1, w)
    y_min = max(py - neighborhood//2, 0)
    y_max = min(py + neighborhood//2 + 1, h)
    region = pc_data[y_min:y_max, x_min:x_max, :3]
    valid = region[np.isfinite(region).all(axis=2)]
    if len(valid) == 0:
        return None
    return tuple(valid.mean(axis=0))

def assign_labels(world_coords, pixel_coords):
    """Label LEDs consistently as p1, p2, p3 using scalene triangle logic"""
    # Compute all distances
    pairs = list(itertools.combinations(range(3), 2))
    distances = [(i, j, euclidean_distance(world_coords[i], world_coords[j])) for i,j in pairs]
    # Sort distances descending
    distances.sort(key=lambda x: -x[2])
    # Longest distance
    i1, j1, _ = distances[0]
    # Second longest
    i2, j2, _ = distances[1]
    # Find common point
    common = set([i1,j1]).intersection([i2,j2])
    if len(common) != 1:
        # fallback if something goes wrong
        return ["p1","p2","p3"]
    p1_idx = common.pop()
    p3_idx = i1 if i1 != p1_idx else j1
    p2_idx = i2 if i2 != p1_idx else j2
    labels = [""]*3
    labels[p1_idx] = "p1"
    labels[p2_idx] = "p2"
    labels[p3_idx] = "p3"
    return labels

def draw_lines_and_labels(display, pixel_coords, world_coords, labels):
    """Draw lines between LEDs, label distances, LED names, and show 3D coordinates"""
    # Draw lines and distances
    for (i1, i2) in itertools.combinations(range(3),2):
        px1, py1 = pixel_coords[i1]
        px2, py2 = pixel_coords[i2]
        cv.line(display, (px1, py1), (px2, py2), (255,0,0), 2)
    #    d = euclidean_distance(world_coords[i1], world_coords[i2])
        mid_x = (px1 + px2)//2
        mid_y = (py1 + py2)//2
    #    cv.putText(display, f"{d:.2f}m", (mid_x, mid_y-5),
    #               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Draw LED labels + 3D coordinates
    for (px, py), label, coord in zip(pixel_coords, labels, world_coords):
        text = f"{label} ({coord[0]:.2f},{coord[1]:.2f},{coord[2]:.2f})"
        cv.putText(display, text, (px+5, py-5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)


# -----------------------------
# Main
# -----------------------------
def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed")
        return

    left = sl.Mat()
    point_cloud = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    cv.namedWindow("LED Pose Detection")
    marker_positions = []

    while True:
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(left, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
        frame = left.get_data()
        if frame.shape[2] == 4:
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Threshold logic
        thresh = np.zeros_like(gray, dtype=np.uint8)
        thresh[gray >= GRAY_THRESHOLD] = 255

        # Detect LED blobs
        led_pixels = find_led_blobs(gray)
        if len(led_pixels) < N_LEDS:
            display = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
            cv.imshow("LED Pose Detection", display)
            if cv.waitKey(1) & 0xFF == 27:
                break
            continue

        # Keep only top N_LEDS largest (by area can be approximated by intensity sum)
        # For simplicity here we take first N_LEDS
        led_pixels = led_pixels[:N_LEDS]

        pixel_coords = []
        world_coords = []
        pc_data = point_cloud.get_data()
        for px, py in led_pixels:
            pos3d = get_3d_position(pc_data, px, py)
            if pos3d is not None:
                pixel_coords.append((px, py))
                world_coords.append(pos3d)

        if len(world_coords) != N_LEDS:
            display = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
            cv.imshow("LED Pose Detection", display)
            if cv.waitKey(1) & 0xFF == 27:
                break
            continue

        # Add marker to list




        
        marker_positions.append(world_coords)

        # Assign consistent labels
        labels = assign_labels(world_coords, pixel_coords)

        # Draw lines, distances, and labels
        display = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        draw_lines_and_labels(display, pixel_coords, world_coords, labels)

        cv.imshow("LED Pose Detection", display)
        if cv.waitKey(1) & 0xFF == 27:
            break

    zed.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
