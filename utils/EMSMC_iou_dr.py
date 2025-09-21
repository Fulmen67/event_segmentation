import numpy as np
import cv2
import os
import csv
import re


def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return (mask > 128).astype(np.uint8)


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 1


def compute_detection_rate(pred_mask, gt_mask, threshold=0.5):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    false_positives = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    iou = intersection / union if union > 0 else 1
    return (iou >= threshold) and (intersection >= false_positives)

# Define your test scenes
scenes_to_test = [
    "carpet2_drone_combinations",
    "carpet2_dji_combinations",
    "carpet2_bird_combinations",
    "echo_bird_combinations",
    "city_helicopter_combinations",
    "city_plane_combinations",
]

# Base directories
prediction_root = "/home/yousef/Documents/event_segmentation/test_results/test_results_EMSMC/"
ground_truth_root = "/home/yousef/shared_for_test/test/ground_truth_images/"

# Output CSV
csv_path = "evaluation_results_all_scenes.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Scene", "Timestamp (s)", "Best Cluster", "IoU", "Detected"])

    for scene in scenes_to_test:
        gt_scene_dir = os.path.join(ground_truth_root, scene)
        pred_scene_dir = os.path.join(prediction_root, scene)
        gt_pattern = re.compile(rf"gt_{scene}\.bag_(\d+\.\d+)\.png")

        scene_ious = []
        scene_detections = 0
        scene_total_frames = 0

        cluster_ious = {1: [], 2: []}
        cluster_drs = {1: [], 2: []}

        if not os.path.isdir(gt_scene_dir):
            print(f"Ground truth directory not found for scene: {scene}")
            continue

        if not os.path.isdir(pred_scene_dir):
            print(f"Prediction directory not found for scene: {scene}")
            continue

        for filename in os.listdir(gt_scene_dir):
            match = gt_pattern.match(filename)
            if not match:
                continue

            timestamp_sec_str = match.group(1)
            timestamp_ns_str = str(int(float(timestamp_sec_str) * 1e9))

            gt_path = os.path.join(gt_scene_dir, filename)
            gt_mask = load_mask(gt_path)

            best_iou = -1
            best_dr = False
            best_cluster = None

            for cluster in [1, 2]:
                pred_filename = f"{scene}_{timestamp_ns_str}_cluster{cluster}.png"
                pred_path = os.path.join(pred_scene_dir, pred_filename)

                if not os.path.exists(pred_path):
                    print(f"Missing prediction mask: {pred_path}")
                    continue

                pred_mask = load_mask(pred_path)
                iou = compute_iou(pred_mask, gt_mask)
                dr = compute_detection_rate(pred_mask, gt_mask)

                cluster_ious[cluster].append(iou)
                cluster_drs[cluster].append(int(dr))

                if iou > best_iou:
                    best_iou = iou
                    best_dr = dr
                    best_cluster = cluster

            scene_total_frames += 1

            if best_iou >= 0:
                scene_ious.append(best_iou)
                if best_dr:
                    scene_detections += 1
                print(f"{scene} @ {timestamp_sec_str}s: Best cluster={best_cluster}, IoU={best_iou:.4f}, DR={'Yes' if best_dr else 'No'}")
                writer.writerow([scene, timestamp_sec_str, best_cluster, best_iou, int(best_dr)])
            else:
                print(f"No prediction found for {scene} @ {timestamp_sec_str}s")
                writer.writerow([scene, timestamp_sec_str, "-", 0.0, 0])

        # Scene summary
        mean_iou = np.mean(scene_ious) if scene_ious else 0
        dr = (scene_detections / scene_total_frames) * 100 if scene_total_frames else 0

        mean_iou_c1 = np.mean(cluster_ious[1]) if cluster_ious[1] else 0
        mean_dr_c1 = (sum(cluster_drs[1]) / len(cluster_drs[1])) * 100 if cluster_drs[1] else 0

        mean_iou_c2 = np.mean(cluster_ious[2]) if cluster_ious[2] else 0
        mean_dr_c2 = (sum(cluster_drs[2]) / len(cluster_drs[2])) * 100 if cluster_drs[2] else 0

        writer.writerow([])
        writer.writerow([f"{scene} - Mean IoU", mean_iou])
        writer.writerow([f"{scene} - Detection Rate (%)", dr])
        writer.writerow([f"{scene} - Cluster 1 Mean IoU", mean_iou_c1])
        writer.writerow([f"{scene} - Cluster 1 Detection Rate (%)", mean_dr_c1])
        writer.writerow([f"{scene} - Cluster 2 Mean IoU", mean_iou_c2])
        writer.writerow([f"{scene} - Cluster 2 Detection Rate (%)", mean_dr_c2])
        writer.writerow([])

        print(f"\n{scene} - Mean IoU: {mean_iou:.4f}")
        print(f"{scene} - Detection Rate: {dr:.2f}%")
        print(f"{scene} - Cluster 1 Mean IoU: {mean_iou_c1:.4f}, DR: {mean_dr_c1:.2f}%")
        print(f"{scene} - Cluster 2 Mean IoU: {mean_iou_c2:.4f}, DR: {mean_dr_c2:.2f}%")
