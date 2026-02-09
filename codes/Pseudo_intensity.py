import cv2 as cv
import numpy as np
import pyzed.sl as sl

# -----------------------------
# Globals for mouse interaction
# -----------------------------
points = []
current_gray = None  # will hold latest grayscale frame

def rescaleFrame(frame, scale=1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

def mouse_callback(event, x, y, flags, param):
    global points, current_gray

    if current_gray is None:
        return

    if event == cv.EVENT_LBUTTONDOWN:
        # Add point
        if 0 <= x < current_gray.shape[1] and 0 <= y < current_gray.shape[0]:
            value = int(current_gray[y, x])
            points.append((x, y, value))

    elif event == cv.EVENT_RBUTTONDOWN:
        # Clear all points
        points.clear()

def main():
    global current_gray

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 30

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed")
        return

    left = sl.Mat()
    rt = sl.RuntimeParameters()


    window_name = "ZED Viewer"
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_callback)

    print("Left click: add point | Right click: clear points | ESC: quit")

    try:
        while True:
            if zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(left, sl.VIEW.LEFT)
            frame = left.get_data()

            if frame is None or frame.size == 0:
                continue

            # BGRA -> BGR
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

            frame = rescaleFrame(frame)

            # Convert to grayscale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Ensure uint8
            if gray.dtype.kind == 'f':
                gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
            else:
                gray = np.clip(gray, 0, 255).astype(np.uint8)

            current_gray = gray.copy()

            # Threshold (your original logic)
            _, thresh = cv.threshold(gray, 250, 255, cv.THRESH_BINARY)

            # Convert to BGR so we can draw colored overlays
            display = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)

            # Draw points + values
            for (x, y, val) in points:
                cv.circle(display, (x, y), 4, (0, 0, 255), -1)
                cv.putText(
                    display,
                    str(val),
                    (x + 5, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv.LINE_AA
                )

            cv.imshow(window_name, display)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord('s'):
                cv.imwrite("left_frame.png", frame)
                print("Saved left_frame.png")

    finally:
        zed.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
