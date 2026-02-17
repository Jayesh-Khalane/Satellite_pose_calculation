
#################################### For stero pair capture ###############################################
# import pyzed.sl as sl
# import cv2
# import os
# import re

# # -----------------------------------------
# # Configuration
# # -----------------------------------------
# BASE_DIR = "data/stereo_pair_images"

# RESOLUTIONS = {
#     "2k": sl.RESOLUTION.HD2K,
#     "1080p": sl.RESOLUTION.HD1080,
#     "720p": sl.RESOLUTION.HD720
# }


# # -----------------------------------------
# # Get Next Available Index
# # -----------------------------------------
# def get_next_index(folder_path):
#     os.makedirs(folder_path, exist_ok=True)

#     existing_files = os.listdir(folder_path)

#     pattern = re.compile(r"left_(\d{4})\.png")
#     indices = []

#     for file in existing_files:
#         match = pattern.match(file)
#         if match:
#             indices.append(int(match.group(1)))

#     if len(indices) == 0:
#         return 1
#     else:
#         return max(indices) + 1


# # -----------------------------------------
# # Proper ZED Image Conversion (FIXED COLOR)
# # -----------------------------------------
# def convert_zed_image(mat):
#     img = mat.get_data()

#     # ZED returns BGRA in most SDK versions
#     if img.shape[2] == 4:
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

#     return img


# # -----------------------------------------
# # Capture Function
# # -----------------------------------------
# def capture_at_resolution(res_name, res_value):

#     print(f"\nStarting capture at {res_name.upper()} resolution")

#     output_dir = os.path.join(BASE_DIR, res_name)
#     image_index = get_next_index(output_dir)

#     zed = sl.Camera()
#     init_params = sl.InitParameters()
#     init_params.camera_resolution = res_value
#     init_params.camera_fps = 30

#     if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
#         print(f"Failed to open camera at {res_name}")
#         return

#     runtime_params = sl.RuntimeParameters()
#     image_left = sl.Mat()
#     image_right = sl.Mat()

#     # Warmup frames for stable exposure
#     for _ in range(30):
#         zed.grab(runtime_params)

#     if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

#         zed.retrieve_image(image_left, sl.VIEW.LEFT)
#         zed.retrieve_image(image_right, sl.VIEW.RIGHT)

#         left_img = convert_zed_image(image_left)
#         right_img = convert_zed_image(image_right)

#         left_filename = os.path.join(output_dir, f"left_{image_index:04d}.png")
#         right_filename = os.path.join(output_dir, f"right_{image_index:04d}.png")

#         cv2.imwrite(left_filename, left_img)
#         cv2.imwrite(right_filename, right_img)

#         print("Saved:")
#         print(f"  {left_filename}")
#         print(f"  {right_filename}")

#     zed.close()
#     print(f"{res_name.upper()} capture complete.")


# # -----------------------------------------
# # Main Execution
# # -----------------------------------------
# if __name__ == "__main__":

#     print("\nAutomatic Multi-Resolution Stereo Capture\n")

#     for res_name, res_value in RESOLUTIONS.items():
#         capture_at_resolution(res_name, res_value)

#     print("\nAll captures completed successfully.\n")











########################### For video capture #######################################
import pyzed.sl as sl
import cv2
import os
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
output_folder = r"D:\Satellite_pose_calculation\data\Captures"
os.makedirs(output_folder, exist_ok=True)

video_length_seconds = 40       # seconds per video (will stop early if user presses 'q')
fps = 60
width, height = 1920, 1080     # 1080p resolution
# -----------------------------

# Initialize ZED camera
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD1080
init.camera_fps = fps
init.coordinate_units = sl.UNIT.METER
init.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
init.depth_mode = sl.DEPTH_MODE.NONE

zed = sl.Camera()
status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"ZED Error: {status}")
    exit(1)

# Helper function to find next available filename
def get_next_filename(folder):
    index = 1
    while True:
        filename = os.path.join(folder, f"satellite_{index:03d}.mp4")
        if not os.path.exists(filename):
            return filename
        index += 1

output_filename = get_next_filename(output_folder)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

print(f"Recording to {output_filename} ...")
print("Press 'q' in the preview window to stop early.")

# Capture loop
start_time = time.time()
image = sl.Mat()

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()  # BGRA

        # Convert to BGR (drop alpha channel)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Resize to output resolution
        frame_resized = cv2.resize(frame_bgr, (width, height))

        # Write to video
        out.write(frame_resized)

        # Optional preview
        cv2.imshow("ZED Capture", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user.")
            break

    # Stop automatically after video_length_seconds
    if time.time() - start_time >= video_length_seconds:
        print("Reached max video duration.")
        break

# Release everything safely
out.release()
cv2.destroyAllWindows()
zed.close()
print(f"Video saved and camera released: {output_filename}")
