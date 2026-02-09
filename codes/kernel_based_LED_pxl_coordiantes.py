import cv2 as cv
import numpy as np
import pyzed.sl as sl

# -----------------------------
# Parameters
# -----------------------------
GRAY_THRESHOLD = 252   # pixel intensity to count as bright
THRESHOLD_SUM = 20     # number of bright pixels in kernel to trigger detection
KERNEL_SIZE = 5        # kernel size (e.g., 5x5)

POINT_RADIUS = 1       # circle radius for visualization (optional)

def rescaleFrame(frame, scale=1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

def main():
    # Initialize ZED
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed")
        return

    left = sl.Mat()
    rt = sl.RuntimeParameters()

    window_name = "Kernel Bright Pixel Detection"
    cv.namedWindow(window_name)

    print("Realtime kernel detection | ESC to quit")

    # Define kernel (all ones)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)

    try:
        while True:
            if zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(left, sl.VIEW.LEFT)
            frame = left.get_data()

            if frame is None or frame.size == 0:
                continue

            # BGRA -> BGR
            if frame.shape[2] == 4:
                frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

            frame = rescaleFrame(frame)

            # Grayscale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # -----------------------------
            # Binary mask of bright pixels
            # -----------------------------
            bright_mask = (gray > GRAY_THRESHOLD).astype(np.float32)

            # -----------------------------
            # Convolution with kernel
            # -----------------------------
            conv = cv.filter2D(bright_mask, ddepth=-1, kernel=kernel, borderType=cv.BORDER_CONSTANT)

            # -----------------------------
            # Find kernel centers where sum > THRESHOLD_SUM
            # -----------------------------
            ys, xs = np.where(conv > THRESHOLD_SUM)

            # -----------------------------
            # Draw detection points on grayscale output
            # -----------------------------
            output = gray.copy()
            for x, y in zip(xs, ys):
                cv.circle(output, (x, y), POINT_RADIUS, 255, 1)  # white circle

            # Optional: print coordinates of middle pixels
            if len(xs) > 0:
                for x, y in zip(xs, ys):
                    print(f"Detected bright kernel at x={x}, y={y}")

            cv.imshow(window_name, output)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        zed.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
