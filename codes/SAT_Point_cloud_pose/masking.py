import pyzed.sl as sl
import open3d as o3d
import numpy as np

ZED0_SN = 33140394   # local camera

def main():

    # -----------------------------
    # Initialize ZED Camera
    # -----------------------------
    zed = sl.Camera()

    init = sl.InitParameters()
    init.set_from_serial_number(ZED0_SN)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init.coordinate_units = sl.UNIT.CENTIMETER
    init.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return

    point_cloud = sl.Mat()

    # -----------------------------
    # Open3D Visualizer
    # -----------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window("ZED2i Point Cloud", width=1280, height=720)

    pcd = o3d.geometry.PointCloud()
    geom_added = False

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.5

    try:

        while True:

            if zed.grab() == sl.ERROR_CODE.SUCCESS:

                # Retrieve XYZRGBA point cloud
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                pc_np = point_cloud.get_data()  # (H,W,4)

                # -----------------------------
                # Extract XYZ
                # -----------------------------
                xyz = pc_np[:, :, :3].reshape(-1, 3)

                # Convert coordinate system
                xyz_lh = np.zeros_like(xyz)
                xyz_lh[:, 0] = xyz[:, 0]
                xyz_lh[:, 1] = xyz[:, 2]
                xyz_lh[:, 2] = -xyz[:, 1]
                xyz = xyz_lh

                # -----------------------------
                # Remove invalid points
                # -----------------------------
                valid = np.isfinite(xyz).all(axis=1)
                xyz = xyz[valid]

                rgba = pc_np[:, :, 3].reshape(-1)[valid]

                # -----------------------------
                # Extract RGB colors
                # -----------------------------
                rgba_int = rgba.view(np.uint32)

                r = ((rgba_int >> 0) & 255).astype(np.uint8)
                g = ((rgba_int >> 8) & 255).astype(np.uint8)
                b = ((rgba_int >> 16) & 255).astype(np.uint8)

                colors = np.stack((r, g, b), axis=1) / 255.0

                # -----------------------------
                # Distance filtering (2 meters)
                # units = centimeters
                # -----------------------------
                distances = np.linalg.norm(xyz, axis=1)
                mask = distances < 100   # 100 cm = 1 meter

                xyz = xyz[mask]
                colors = colors[mask]

                # Skip frame if empty
                if xyz.shape[0] == 0:
                    continue

                # -----------------------------
                # Downsample for speed
                # -----------------------------
                max_points = 100000
                if xyz.shape[0] > max_points:

                    idx = np.random.choice(
                        xyz.shape[0],
                        max_points,
                        replace=False
                    )

                    xyz = xyz[idx]
                    colors = colors[idx]

                # -----------------------------
                # Update Open3D cloud
                # -----------------------------
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                if not geom_added:
                    vis.add_geometry(pcd)
                    geom_added = True

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

    except KeyboardInterrupt:
        print("Stopping visualization...")

    finally:
        zed.close()
        vis.destroy_window()


if __name__ == "__main__":
    main()