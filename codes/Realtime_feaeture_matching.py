import pyzed.sl as sl
import cv2
import numpy as np
import time

# ==========================
# INITIALIZE ZED CAMERA
# ==========================
print("[INFO] Initializing ZED2i...")

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # use HD720 for real-time speed
init_params.depth_mode = sl.DEPTH_MODE.NONE
init_params.coordinate_units = sl.UNIT.METER

status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print("[ERROR] ZED Camera failed to open:", status)
    exit()

print("[INFO] ZED Camera opened successfully.")

runtime_parameters = sl.RuntimeParameters()

left_image = sl.Mat()
right_image = sl.Mat()

# ==========================
# FEATURE DETECTOR (SIFT)
# ==========================
print("[INFO] Initializing SIFT detector...")
sift = cv2.SIFT_create(nfeatures=3000)

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

print("[INFO] Starting real-time feature matching...")
print("Press 'q' to exit.\n")

# ==========================
# MAIN LOOP
# ==========================
while True:
    start_time = time.time()

    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

        # Retrieve left and right images
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        zed.retrieve_image(right_image, sl.VIEW.RIGHT)

        left_frame = left_image.get_data()
        right_frame = right_image.get_data()

        # Convert BGRA to BGR
        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGRA2BGR)
        right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGRA2BGR)

        # Convert to grayscale
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints
        kp1, des1 = sift.detectAndCompute(left_gray, None)
        kp2, des2 = sift.detectAndCompute(right_gray, None)

        good_matches = []

        if des1 is not None and des2 is not None:
            matches = flann.knnMatch(des1, des2, k=2)

            # Lowe's ratio test
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        # Draw matches (limit to 50 for speed)
        match_img = cv2.drawMatches(
            left_frame, kp1,
            right_frame, kp2,
            good_matches[:50],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)

        # Overlay info
        cv2.putText(match_img, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(match_img, f"Left KP: {len(kp1)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(match_img, f"Right KP: {len(kp2)}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(match_img, f"Matches: {len(good_matches)}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("ZED2i Real-Time Feature Matching", match_img)

        # Terminal logging
        print(f"FPS: {fps:.2f} | KP1: {len(kp1)} | KP2: {len(kp2)} | Matches: {len(good_matches)}")

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ==========================
# CLEANUP
# ==========================
zed.close()
cv2.destroyAllWindows()
print("\n[INFO] Camera closed successfully.")
