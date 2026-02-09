import cv2 as cv
import numpy as np
import pyzed.sl as sl
import itertools
import math

# -----------------------------
# Parameters
# -----------------------------
RGB_SUM_THRESHOLD = 760  
MIN_BLOB_SIZE = 10     
DOT_RADIUS = 3
LINE_COLOR = (0, 255, 255)
TEXT_COLOR = (0, 255, 255)
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
AXIS_LENGTH = 10.0 # Centimeters
LOG_INTERVAL = 30  # Log to terminal every 30 frames

# -----------------------------
# Logic Functions
# -----------------------------

def get_3d_position_at_pixel(pc, px, py):
    if 0 <= px < pc.shape[1] and 0 <= py < pc.shape[0]:
        point = pc[py, px, :3]
        if np.isfinite(point).all():
            return np.array(point)
    return None

def find_blob_centroids(binary_mask):
    _, _, stats, centroids = cv.connectedComponentsWithStats(binary_mask)
    return [(int(c[0]), int(c[1])) for i, c in enumerate(centroids[1:], start=1)
            if stats[i, cv.CC_STAT_AREA] >= MIN_BLOB_SIZE]

def assign_labels(world_coords):
    """
    Logic:
    P1 = Common endpoint of the two largest distances.
    P3 = The other endpoint of the LARGEST distance.
    P2 = The other endpoint of the SECOND LARGEST distance.
    """
    if len(world_coords) != 3:
        return [""] * len(world_coords)
    
    idx_pairs = list(itertools.combinations(range(3), 2))
    dists = []
    for i, j in idx_pairs:
        d = np.linalg.norm(world_coords[i] - world_coords[j])
        dists.append({'pair': (i, j), 'dist': d})
    
    # Sort by distance descending
    dists.sort(key=lambda x: x['dist'], reverse=True)
    
    L1_pair = dists[0]['pair']  # Largest
    L2_pair = dists[1]['pair']  # Second largest
    
    common = set(L1_pair).intersection(set(L2_pair))
    if len(common) != 1:
        return ["err", "err", "err"]
        
    p1_idx = list(common)[0]
    p3_idx = L1_pair[0] if L1_pair[0] != p1_idx else L1_pair[1]
    p2_idx = L2_pair[0] if L2_pair[0] != p1_idx else L2_pair[1]
    
    labels = [""] * 3
    labels[p1_idx] = "p1"
    labels[p2_idx] = "p2"
    labels[p3_idx] = "p3"
    return labels

def compute_pose_matrix(world_coords, labels):
    if len(world_coords) != 3:
        return None, None

    lookup = {label: coord for coord, label in zip(world_coords, labels)}
    if "p1" not in lookup or "p2" not in lookup or "p3" not in lookup:
        return None, None

    p1, p2, p3 = lookup["p1"], lookup["p2"], lookup["p3"]
    centroid = np.mean(world_coords, axis=0)

    x_axis = p2 - p1
    x_axis /= np.linalg.norm(x_axis)

    v1, v2 = p2 - p1, p3 - p1
    z_axis = np.cross(v1, v2)
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.column_stack((x_axis, y_axis, z_axis))
    return centroid, R

def get_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])

def project_point(point_3d, camera_params):
    fx, fy, cx, cy = camera_params.fx, camera_params.fy, camera_params.cx, camera_params.cy
    x, y, z = point_3d
    if z <= 0: return None 
    u = int((x * fx / z) + cx)
    v = int(cy - (y * fy / z)) 
    return (u, v)

def draw_orientation_axes(image, centroid, R, cam_params):
    if centroid is None or R is None: return
    origin_2d = project_point(centroid, cam_params)
    pts_3d = [centroid + R[:, i] * AXIS_LENGTH for i in range(3)]
    pts_2d = [project_point(p, cam_params) for p in pts_3d]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    lbls = ["X", "Y", "Z"]
    if origin_2d:
        for p2d, color, l in zip(pts_2d, colors, lbls):
            if p2d:
                cv.line(image, origin_2d, p2d, color, 2)
                cv.putText(image, l, p2d, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# -----------------------------
# Main
# -----------------------------
def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    # Using Neural depth as indicated in your logs
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL 

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed")
        return

    # FIXED: Updated for Latest SDK
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    
    left, point_cloud = sl.Mat(), sl.Mat()
    runtime_params = sl.RuntimeParameters()
    
    cv.namedWindow("LED Detection", cv.WINDOW_NORMAL)
    frame_count = 0

    while True:
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(left, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        
        frame = left.get_data()
        pc_data = point_cloud.get_data()
        rgb_sum = np.sum(frame[:, :, :3], axis=2)
        binary_mask = (rgb_sum > RGB_SUM_THRESHOLD).astype(np.uint8) * 255

        centroids = find_blob_centroids(binary_mask)
        pixel_coords, world_coords = [], []
        
        for cx, cy in centroids:
            w_point = get_3d_position_at_pixel(pc_data, cx, cy)
            if w_point is not None:
                pixel_coords.append((cx, cy))
                world_coords.append(w_point)

        display = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)

        if len(world_coords) == 3:
            labels = assign_labels(world_coords)
            
            # --- TERMINAL LOGGING ---
            if frame_count % LOG_INTERVAL == 0:
                print("-" * 30)
                log_map = {lbl: coord for coord, lbl in zip(world_coords, labels)}
                for lbl in ["p1", "p2", "p3"]:
                    c = log_map.get(lbl, [0,0,0])
                    print(f"{lbl.upper()}: x={c[0]:>7.2f}, y={c[1]:>7.2f}, z={c[2]:>7.2f}")
            
            for (px, py), (x, y, z), label in zip(pixel_coords, world_coords, labels):
                cv.circle(display, (px, py), DOT_RADIUS, (0, 0, 255), -1)
                txt = f"{label} ({x:.1f},{y:.1f},{z:.1f})"
                cv.putText(display, txt, (px, py), cv.FONT_HERSHEY_SIMPLEX, 
                           TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv.LINE_AA)

            cv.polylines(display, [np.array(pixel_coords)], True, LINE_COLOR, 1)
            
            centroid, R = compute_pose_matrix(world_coords, labels)
            if centroid is not None:
                draw_orientation_axes(display, centroid, R, calib)
                Rx, Ry, Rz = get_euler_angles(R)
                cv.putText(display, f"Pos: {centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}", 
                           (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.putText(display, f"Rot (Deg): X:{Rx:.1f} Y:{Ry:.1f} Z:{Rz:.1f}", 
                           (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frame_count += 1
        cv.imshow("LED Detection", display)
        if cv.waitKey(1) & 0xFF == 27: break

    zed.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()




    