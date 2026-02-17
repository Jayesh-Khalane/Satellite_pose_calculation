import cv2 as cv
import pyzed.sl as sl
import numpy as np

# Global for mouse click
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked pixel: {clicked_point}")

def main():
    global clicked_point

    # Initialize ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS # or PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER


    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed")
        return

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    cv.namedWindow("ZED View")
    cv.setMouseCallback("ZED View", mouse_callback)

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        # Get left image and depth map
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        frame = image.get_data()
        depth_map = depth.get_data()  # depth in meters

        if frame.shape[2] == 4:
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        # If user clicked, show depth
        if clicked_point:
            u, v = clicked_point
            u = np.clip(u, 0, frame.shape[1]-1)
            v = np.clip(v, 0, frame.shape[0]-1)
            Z = depth_map[v, u]  # Depth at clicked pixel
            if np.isfinite(Z):
                cv.drawMarker(frame, (u, v), (0,255,0), cv.MARKER_CROSS, 20, 2)
                cv.putText(frame, f"Depth: {Z:.2f} m", (u+10, v-10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                print(f"Clicked pixel: ({u},{v}) -> Depth: {Z:.2f} m")
            else:
                print(f"Clicked pixel: ({u},{v}) -> Depth invalid")

        cv.imshow("ZED View", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

    zed.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()










# format binary_little_endian 1.0
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
