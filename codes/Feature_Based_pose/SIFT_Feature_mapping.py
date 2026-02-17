import cv2
import numpy as np
import os
import time

# ==========================
# CONFIG
# ==========================
LEFT_IMAGE_PATH = r"data\stereo_pair_images\2k\left_0011.png"
RIGHT_IMAGE_PATH = r"data\stereo_pair_images\2k\right_0011.png"

RESIZE_FOR_DISPLAY = True
DISPLAY_SCALE = 0.5  # Only affects visualization

# ==========================
# STEP 1: LOAD IMAGES
# ==========================
print("\n[INFO] Loading images...")

if not os.path.exists(LEFT_IMAGE_PATH):
    print(f"[ERROR] Left image not found: {LEFT_IMAGE_PATH}")
    exit()

if not os.path.exists(RIGHT_IMAGE_PATH):
    print(f"[ERROR] Right image not found: {RIGHT_IMAGE_PATH}")
    exit()

left_img = cv2.imread(LEFT_IMAGE_PATH)
right_img = cv2.imread(RIGHT_IMAGE_PATH)

print("[INFO] Images loaded successfully.")
print(f"[INFO] Left Image Shape: {left_img.shape}")
print(f"[INFO] Right Image Shape: {right_img.shape}")

# Convert to grayscale
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

print("[INFO] Converted images to grayscale.")

# ==========================
# STEP 2: FEATURE DETECTION (SIFT)
# ==========================
print("\n[INFO] Initializing SIFT detector...")
sift = cv2.SIFT_create(nfeatures=10000)

start_time = time.time()
kp1, des1 = sift.detectAndCompute(left_gray, None)
kp2, des2 = sift.detectAndCompute(right_gray, None)
end_time = time.time()

print("[INFO] SIFT detection completed.")
print(f"[INFO] Left image keypoints: {len(kp1)}")
print(f"[INFO] Right image keypoints: {len(kp2)}")
print(f"[INFO] Descriptor shape (left): {des1.shape}")
print(f"[INFO] Descriptor shape (right): {des2.shape}")
print(f"[INFO] Detection time: {end_time - start_time:.3f} seconds")

# Draw keypoints
left_kp_img = cv2.drawKeypoints(left_img, kp1, None, color=(0,255,0))
right_kp_img = cv2.drawKeypoints(right_img, kp2, None, color=(0,255,0))

# ==========================
# STEP 3: FEATURE MATCHING (FLANN)
# ==========================
print("\n[INFO] Matching features using FLANN...")

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

start_time = time.time()
matches = flann.knnMatch(des1, des2, k=2)
end_time = time.time()

print(f"[INFO] Raw matches found: {len(matches)}")
print(f"[INFO] Matching time: {end_time - start_time:.3f} seconds")

# ==========================
# STEP 4: RATIO TEST (Lowe's test)
# ==========================
print("\n[INFO] Applying Lowe's ratio test...")

good_matches = []
ratio_threshold = 0.5

for m, n in matches:
    if m.distance < ratio_threshold * n.distance:
        good_matches.append(m)

print(f"[INFO] Good matches after ratio test: {len(good_matches)}")

# ==========================
# STEP 5: DRAW MATCHES
# ==========================
match_img = cv2.drawMatches(
    left_img, kp1,
    right_img, kp2,
    good_matches[:100],  # show top 100 matches
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

print("\n[INFO] Drawing matches (showing top 100)...")

# ==========================
# DISPLAY SECTION
# ==========================
def resize_for_display(img):
    if not RESIZE_FOR_DISPLAY:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))

cv2.imshow("Left Keypoints", resize_for_display(left_kp_img))
cv2.imshow("Right Keypoints", resize_for_display(right_kp_img))
cv2.imshow("Feature Matches (Top 100)", resize_for_display(match_img))

print("\n[INFO] Press any key in image window to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==========================
# FINAL STATS
# ==========================
print("\n================ FINAL REPORT ================")
print(f"Total Left Keypoints: {len(kp1)}")
print(f"Total Right Keypoints: {len(kp2)}")
print(f"Raw Matches: {len(matches)}")
print(f"Good Matches: {len(good_matches)}")

match_ratio = len(good_matches) / len(matches) if len(matches) > 0 else 0
print(f"Match Quality Ratio: {match_ratio:.4f}")
print("=============================================\n")
