import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pyzed.sl as sl
import numpy as np
import pandas as pd
import threading
import time
import math

def rotation_matrix_to_euler(R):
    """Calculates Roll, Pitch, and Yaw (in degrees) from a 3x3 Rotation Matrix."""
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2]) # Roll
        y = math.atan2(-R[2,0], sy)     # Pitch
        z = math.atan2(R[1,0], R[0,0])  # Yaw
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])

class SatelliteTrackerApp:
    def __init__(self, target_csv_path):
        # 1. Initialize GUI
        self.app = gui.Application.instance
        self.app.initialize()
        
        self.window = gui.Application.instance.create_window("Live Satellite 6D Tracker", 1280, 720)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([0.1, 0.1, 0.1, 1.0]) # Dark grey background
        
        # 2. Setup HUD (Heads Up Display) for logging
        self.info_panel = gui.Label("Initializing Tracker...")
        self.window.add_child(self.widget3d)
        self.window.add_child(self.info_panel)
        self.window.set_on_layout(self._on_layout)

        # 3. Load Golden Model (Target)
        print("Loading Scaled Reference...")
        df = pd.read_csv(target_csv_path)
        self.target_pcd = o3d.geometry.PointCloud()
        self.target_pcd.points = o3d.utility.Vector3dVector(df[['X', 'Y', 'Z']].values)
        self.target_pcd.paint_uniform_color([0, 1, 0]) # GREEN = Reference
        self.target_down = self.target_pcd.voxel_down_sample(1.0)
        self.target_down.estimate_normals()
        
        # Add to scene
        self.mat_target = rendering.MaterialRecord()
        self.mat_target.shader = "defaultUnlit"
        self.mat_target.point_size = 3.0
        self.widget3d.scene.add_geometry("GoldenModel", self.target_pcd, self.mat_target)

        # Setup Live Point Cloud holder
        self.live_pcd = o3d.geometry.PointCloud()
        self.mat_live = rendering.MaterialRecord()
        self.mat_live.shader = "defaultUnlit"
        self.mat_live.point_size = 4.0

        # Setup Camera View
        bbox = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bbox, bbox.get_center())
        
        # 4. State Variables
        self.is_running = True
        self.current_pose = np.identity(4)
        
        # 5. Start ZED Thread
        self.zed_thread = threading.Thread(target=self.tracking_loop)
        self.zed_thread.start()

    def _on_layout(self, layout_context):
        """Places the HUD text in the top left corner."""
        r = self.window.content_rect
        self.widget3d.frame = r
        pref = self.info_panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.info_panel.frame = gui.Rect(r.x + 20, r.y + 20, 350, 150)

    def update_gui(self, live_pts, tx, ty, tz, roll, pitch, yaw, fitness, rmse):
        """Safely updates the Open3D Window from the tracking thread."""
        def _update():
            # Update Live Point Cloud geometry (Red)
            self.live_pcd.points = o3d.utility.Vector3dVector(live_pts)
            self.live_pcd.paint_uniform_color([1, 0, 0]) # RED = Live Data
            
            # Refresh geometry in the scene
            if self.widget3d.scene.has_geometry("LiveModel"):
                self.widget3d.scene.remove_geometry("LiveModel")
            self.widget3d.scene.add_geometry("LiveModel", self.live_pcd, self.mat_live)

            # Update HUD Text
            hud_text = (
                f"--- 6D POSE (Relative to Golden Model) ---\n\n"
                f"Translation (cm):\n"
                f"  X: {tx:8.2f} | Y: {ty:8.2f} | Z: {tz:8.2f}\n\n"
                f"Rotation (deg):\n"
                f"  Roll: {roll:5.1f} | Pitch: {pitch:5.1f} | Yaw: {yaw:5.1f}\n\n"
                f"--- ICP DIAGNOSTICS ---\n"
                f"Points Tracked: {len(live_pts)}\n"
                f"Fitness: {fitness:.3f} (1.0 is perfect)\n"
                f"RMSE:    {rmse:.3f} cm"
            )
            self.info_panel.text = hud_text
            self.window.set_needs_layout()

        gui.Application.instance.post_to_main_thread(self.window, _update)

    def tracking_loop(self):
        """The main ZED capture and ICP logic running in the background."""
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(33140394) 
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA 
        init_params.coordinate_units = sl.UNIT.CENTIMETER 

        if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("ZED Camera failed to open!")
            return

        point_cloud_mat = sl.Mat()
        runtime_params = sl.RuntimeParameters()
        
        voxel_size = 1.0

        while self.is_running:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)
                raw_data = point_cloud_mat.get_data()
                
                # 1. Spatial Filter (10cm to 100cm)
                pts = raw_data[:, :, :3].reshape(-1, 3)
                pts = pts[~np.isnan(pts).any(axis=1)]
                mask = (pts[:, 2] < 100.0) & (pts[:, 2] > 10.0)
                filtered_pts = pts[mask]

                if len(filtered_pts) < 100:
                    continue # Not enough points to track

                # 2. Downsample and estimate normals
                source_pcd = o3d.geometry.PointCloud()
                source_pcd.points = o3d.utility.Vector3dVector(filtered_pts)
                source_down = source_pcd.voxel_down_sample(voxel_size)
                source_down.estimate_normals()

                # 3. FAST ICP (Using previous pose as starting point)
                reg_result = o3d.pipelines.registration.registration_icp(
                    source_down, self.target_down, voxel_size * 3.0, self.current_pose,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
                )

                if reg_result.fitness > 0.4: # Only accept if the match is decent
                    self.current_pose = reg_result.transformation
                
                # 4. Extract 6D Pose Data
                tx, ty, tz = self.current_pose[:3, 3]
                roll, pitch, yaw = rotation_matrix_to_euler(self.current_pose[:3, :3])

                # 5. Transform the Live Points visually so they snap onto the Green model
                source_pcd.transform(np.linalg.inv(self.current_pose))

                # 6. Send to GUI
                self.update_gui(np.asarray(source_pcd.points), tx, ty, tz, 
                                roll, pitch, yaw, reg_result.fitness, reg_result.inlier_rmse)
                
                time.sleep(0.01) # Small sleep to prevent thread locking

        zed.close()

    def run(self):
        self.app.run()
        self.is_running = False
        self.zed_thread.join()

if __name__ == "__main__":
    csv_path = r"D:\Satellite_pose_calculation\log\004_scaled_satellite.csv"
    app = SatelliteTrackerApp(csv_path)
    app.run()