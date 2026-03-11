# 🛰️ Satellite Pose Calculation

![Project Status](https://img.shields.io/badge/Status-In--Progress-orange)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![ZED SDK](https://img.shields.io/badge/ZED_SDK-Required-green)

A comprehensive system for **Satellite Pose Estimation** (x, y, z, roll, pitch, yaw) using Multi-View Stereo (MVS) camera images. This project leverages the **ZED 2i Stereo Camera** to achieve high-accuracy pose tracking, targeting an error margin of less than 1.5 cm.

---

## 🚀 Overview

The core objective of this project is to accurately determine the 6D pose of a satellite model in a laboratory environment. By utilizing both image-based and point-cloud-based detection strategies, the system can identify specific markers and calculate their spatial orientation relative to the camera's coordinate system.

### Key Features
*   **Multi-Strategy Detection:** Combines image-based LED marker detection with 3D point cloud analysis.
*   **Robust Clustering:** Utilizes K-means and DBSCAN algorithms for precise marker localization.
*   **Geometry Analysis:** Specialized modules for satellite geometry and source point cloud processing.
*   **Real-time Visualization:** Integrated tools for 3D point cloud plotting and jitter analysis.

---

## 📂 Project Structure

The codebase is organized into functional modules for clarity and maintainability:

| Directory | Description |
| :--- | :--- |
| `codes/LED_Marker_pose/` | Algorithms for detecting LED markers in 2D and 3D. |
| `codes/SAT_Point_cloud_pose/` | Satellite-specific geometry analysis and point cloud processing. |
| `codes/understanding_camera/` | Core utilities for ZED camera interaction, 3D point calculation, and visualization. |
| `colmap-x64-windows-nocuda/` | Integrated COLMAP binaries for potential MVS reconstruction tasks. |
| `ffmpeg/` | Multimedia utilities for video processing and frame extraction. |

---

## 🛠️ Technical Implementation

### Detection Pipeline
1.  **Image Acquisition:** Capturing stereo pairs using the ZED 2i SDK.
2.  **Marker Localization:** 
    *   **2D:** Kernel-based pixel coordinate extraction and K-means clustering.
    *   **3D:** DBSCAN clustering on point clouds to isolate marker centroids.
3.  **Pose Calculation:** Mapping 3D coordinates back to the satellite's reference frame to derive full 6D pose.

### Core Dependencies
*   **ZED SDK / pyzed:** Primary interface for stereo camera data.
*   **Open3D:** High-performance 3D data processing.
*   **OpenCV:** Image processing and marker detection.
*   **Scikit-Learn:** Clustering algorithms (DBSCAN, K-means).
*   **PyVista / PyQtGraph:** Advanced 3D and real-time visualization.

---

## ⚙️ Setup & Installation

1.  **Install ZED SDK:** Ensure the ZED SDK is installed on your system (Required for `pyzed`).
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Jayesh-Khalane/Satellite_pose_calculation.git
    cd Satellite_pose_calculation
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📊 Performance & Goals
*   **Target Accuracy:** < 1.5 cm error in position.
*   **Current Focus:** Refining point cloud-based marker detection for improved robustness in varied lighting conditions.

---

## 📜 License
This project is for internal research at the **LAMA Lab**. All rights reserved.
