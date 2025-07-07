# Landmark_Based_Lane_Detection

This project implements a robust, landmark-based lane detection system using a single Intel RealSense camera. It detects colored cones (as landmarks) and creates a navigable path for autonomous vehicles, such as those used in FSAC(Formula Student Autonomous China).

## Features

- **Cone Detection & Classification:** Uses YOLO V8 deep learning models to detect and classify cones (Red-left / Yellow-right) in real time.
- **Depth Integration:** Utilizes RealSense depth data to accurately localize cones in 3D space.
- **Landmark-Based Centerline Generation:** Pairs detected cones and computes a centerline path between them, including through curves.
- **Visualization:** Overlays bounding boxes, depth measurements, and computed path on live video feed for easy debugging.
- **No LIDAR Required:** Achieves reliable path planning using only the RealSense camera (RGB + Depth).

## How It Works

1. **RealSense Camera Input:** Captures RGB and depth frames.
2. **Cone Detection:** Processes each RGB frame with a trained object detector (YOLO V8) to find and classify cones.
3. **Depth Processing:** For each detected cone, estimate its position relative to the vehicle using depth data.
4. **Landmark Pairing:** Matches left and right cones to create pairs, then computes midpoints (centerline waypoints).
5. **Path Generation:** Connects midpoints to form a navigable centerline path.
6. **Visualization:** Annotates the video stream with cone boxes, distances, and the generated path.

## Setup

### Requirements

- Intel RealSense camera (D435 or D455)
- Python 3.10
- OpenCV
- pyrealsense2
- ultralytics (YOLO)
- numpy
- (Optional) scipy for advanced path smoothing

### Installation

```bash
git clone https://github.com/<Raja Annamalai>/Landmark_Based_Lane_Detection.git
cd Landmark_Based_Lane_Detection
pip install -r requirements.txt
```
