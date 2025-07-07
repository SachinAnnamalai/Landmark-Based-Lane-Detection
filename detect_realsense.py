import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load YOLO model
model = YOLO("runs/detect/train2/weights/best.pt")

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# Depth scale
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

def get_cone_depth(depth_image, x1, y1, x2, y2):
    """Sample a lower region of the bounding box, exclude zeros and outliers."""
    bbox_height = y2 - y1
    roi_y_start = int(y2 - 0.2 * bbox_height)
    roi_y_end = y2
    roi_x_start = int(x1 + 0.25 * (x2 - x1))
    roi_x_end = int(x2 - 0.25 * (x2 - x1))
    depth_roi = depth_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    depth_vals = depth_roi.flatten()
    depth_vals = depth_vals[depth_vals > 0]
    if depth_vals.size == 0:
        return None
    lower = np.percentile(depth_vals, 10)
    upper = np.percentile(depth_vals, 90)
    filtered = depth_vals[(depth_vals >= lower) & (depth_vals <= upper)]
    if filtered.size == 0:
        return None
    return np.median(filtered) * depth_scale

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        height, width = color_image.shape[:2]

        results = model.predict(color_image, conf=0.5, verbose=False)
        annotated_frame = color_image.copy()
        left_cones, right_cones = [], []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cls_id = int(box.cls[0])
                depth_val = get_cone_depth(depth_image, x1, y1, x2, y2)
                if depth_val is None:
                    continue
                if cls_id == 0:  # Red cone (left)
                    left_cones.append({'pos': (cx, cy), 'depth': depth_val})
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{depth_val:.2f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif cls_id == 1:  # Yellow cone (right)
                    right_cones.append({'pos': (cx, cy), 'depth': depth_val})
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"{depth_val:.2f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Pair cones for path generation (by y for visual robustness)
        cone_pairs = []
        used_right_indices = set()
        for l in left_cones:
            min_diff, matched_r_idx = float('inf'), None
            for idx, r in enumerate(right_cones):
                if idx in used_right_indices:
                    continue
                diff = abs(l['pos'][1] - r['pos'][1])
                if diff < min_diff:
                    min_diff, matched_r_idx = diff, idx
            if matched_r_idx is not None:
                used_right_indices.add(matched_r_idx)
                r = right_cones[matched_r_idx]
                center_x = int((l['pos'][0] + r['pos'][0]) / 2)
                center_y = int((l['pos'][1] + r['pos'][1]) / 2)
                avg_depth = (l['depth'] + r['depth']) / 2
                cone_pairs.append({'center': (center_x, center_y), 'y': center_y, 'depth': avg_depth})

        # Sort by increasing depth for true path "forward"
        cone_pairs.sort(key=lambda pair: pair['depth'])
        centerline_points = [pair['center'] for pair in cone_pairs]

        # Draw path and "START" at first pair
        if len(centerline_points) >= 2:
            cv2.polylines(annotated_frame, [np.array(centerline_points)], False, (0, 255, 0), 3)
            # Draw start marker at the first pair
            cv2.circle(annotated_frame, centerline_points[0], 8, (0, 255, 0), -1)
            cv2.putText(annotated_frame, "START", (centerline_points[0][0] + 10, centerline_points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Depth colormap visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        combined_view = np.hstack((annotated_frame, depth_colormap))
        cv2.imshow("Detection + Depth Map", combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()