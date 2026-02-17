import pyzed.sl as sl
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import cv2

def main():
    # --- 1. ZED Camera Setup ---
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.CENTIMETER 
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED")
        return

    # --- 2. PyQtGraph Setup ---
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="ZED Jitter Analysis (Fixed +/- 2cm Window)")
    win.resize(1000, 900)

    # Create 3 Plots
    p_x = win.addPlot(row=0, col=0, title="X-Axis Jitter (cm)")
    p_y = win.addPlot(row=1, col=0, title="Y-Axis Jitter (cm)")
    p_z = win.addPlot(row=2, col=0, title="Z-Axis Jitter (cm)")

    # Font for labels
    font = QtGui.QFont()
    font.setPixelSize(20)
    font.setBold(True)

    # Setup Labels and Grids
    history_limit = 200
    err_labels = []
    plots = [p_x, p_y, p_z]
    
    for p in plots:
        p.setXRange(0, history_limit, padding=0)
        p.showGrid(x=True, y=True, alpha=0.4)
        label = pg.TextItem(text="Error: 0.000", color='w', anchor=(0,0))
        label.setFont(font)
        p.addItem(label)
        err_labels.append(label)

    # Curves (Raw Data)
    curve_x = p_x.plot(pen=pg.mkPen('r', width=1.5))
    curve_y = p_y.plot(pen=pg.mkPen('g', width=1.5))
    curve_z = p_z.plot(pen=pg.mkPen('b', width=1.5))

    # Mean Lines (White Dotted)
    dotted_pen = pg.mkPen(color='w', width=1.5, style=QtCore.Qt.PenStyle.DashLine)
    avg_line_x = p_x.plot(pen=dotted_pen)
    avg_line_y = p_y.plot(pen=dotted_pen)
    avg_line_z = p_z.plot(pen=dotted_pen)

    # Buffers
    data_x, data_y, data_z = [], [], []

    # --- 3. Interaction Setup ---
    clicked_pixel = None
    def on_mouse(event, x, y, flags, param):
        nonlocal clicked_pixel
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pixel = (x, y)

    cv2.namedWindow("ZED View")
    cv2.setMouseCallback("ZED View", on_mouse)

    point_cloud = sl.Mat()
    image_zed = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    print("Click on a point in the 'ZED View' window to start tracking jitter.")
    print("Press ESC or 'Q' to exit.")

    # --- 4. Main Loop ---
    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                
                # Get OpenCV frame
                frame = image_zed.get_data()
                
                if clicked_pixel:
                    # Retrieve raw data from single pixel
                    err, point_val = point_cloud.get_value(clicked_pixel[0], clicked_pixel[1])
                    
                    if np.isfinite(point_val[0]):
                        data_x.append(point_val[0])
                        data_y.append(point_val[1])
                        data_z.append(point_val[2])
                        
                        if len(data_x) > history_limit:
                            data_x.pop(0); data_y.pop(0); data_z.pop(0)
                        
                        # Update Plot Data
                        curve_x.setData(data_x)
                        curve_y.setData(data_y)
                        curve_z.setData(data_z)

                        # Process Plotting Logic for each axis
                        axes_data = [data_x, data_y, data_z]
                        avg_curves = [avg_line_x, avg_line_y, avg_line_z]
                        
                        for i in range(3):
                            current_data = axes_data[i]
                            mean_val = np.mean(current_data)
                            max_val = np.max(current_data)
                            min_val = np.min(current_data)
                            p2p_error = max_val - min_val

                            # Update Mean Dotted Line
                            avg_curves[i].setData([mean_val] * len(current_data))
                            
                            # Set Fixed Scale +/- 2cm
                            plots[i].setYRange(mean_val - 2, mean_val + 2, padding=0)
                            
                            # Update Error Label Text and Position
                            axis_names = ["X", "Y", "Z"]
                            err_labels[i].setText(f"{axis_names[i]} Jitter (P2P): {p2p_error:.3f} cm")
                            err_labels[i].setPos(5, mean_val + 1.85)

                    # Draw crosshair on target
                    cv2.drawMarker(frame, clicked_pixel, (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

                cv2.imshow("ZED View", frame)
                
                # Allow PyQtGraph to process GUI updates
                QtWidgets.QApplication.processEvents() 
            
            # Key Handling
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'): # 27 is ESC
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        zed.close()
        cv2.destroyAllWindows()
        app.quit()


        
        print("Camera closed and program exited.")

if __name__ == "__main__":
    main()