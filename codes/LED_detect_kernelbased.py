import cv2 as cv
import numpy as np
import pyzed.sl as sl

# -----------------------------
# Parameters
# -----------------------------
GRAY_THRESHOLD = 253      # intensity threshold for LEDs
THRESHOLD_SUM = 20        # number of bright pixels in kernel to detect LED
KERNEL_SIZE = 5           # size of kernel for detection

def main():
    # -----------------------------
    # Initialize ZED
    # -----------------------------
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

    # kernel for counting bright pixels
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)

    print("LED Pose Detection running | ESC to quit")

    while True:
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue

        # Retrieve image and XYZ point cloud
        zed.retrieve_image(left, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

        frame = left.get_data()

        
        if frame.shape[2] == 4:
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        # Full-resolution grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # -----------------------------
        # Harsh threshold output
        # -----------------------------
        harsh_gray = np.zeros_like(gray, dtype=np.uint8)
        harsh_gray[gray > GRAY_THRESHOLD] = gray[gray > GRAY_THRESHOLD]

        # -----------------------------
        # Kernel-based LED detection
        # -----------------------------
        bright_mask = (gray > GRAY_THRESHOLD).astype(np.float32)
        conv = cv.filter2D(bright_mask, ddepth=-1, kernel=kernel, borderType=cv.BORDER_CONSTANT)
        ys, xs = np.where(conv > THRESHOLD_SUM)

        # Convert to BGR for overlay
        display = cv.cvtColor(harsh_gray, cv.COLOR_GRAY2BGR)
        pc_data = point_cloud.get_data()  # shape (H, W, 4)

        for x, y in zip(xs, ys):
            px, py = int(x), int(y)
            if 0 <= py < pc_data.shape[0] and 0 <= px < pc_data.shape[1]:
                Xp, Yp, Zp, _ = pc_data[py, px]
                if np.isfinite(Xp + Yp + Zp):
                    text = f"X:{Xp:.2f} Y:{Yp:.2f} Z:{Zp:.2f}"

                    # Get text size
                    (text_width, text_height), baseline = cv.getTextSize(
                        text, cv.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )

                    # Draw rectangle behind text
                    top_left = (px, py - text_height - baseline)
                    bottom_right = (px + text_width, py + baseline)
                    cv.rectangle(display, top_left, bottom_right, (0, 0, 0), -1)

                    # Draw text on top
                    cv.putText(display, text, (px, py),
                               cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv.LINE_AA)

                    # Terminal logging
                    print(f"LED Pixel px={px}, py={py} -> X={Xp:.3f}, Y={Yp:.3f}, Z={Zp:.3f} m")

        cv.imshow("LED Pose Detection", display)
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    zed.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
