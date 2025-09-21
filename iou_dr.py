# in this code I want to try to calculate the Intersection over Union (IoU) and the Detection Rate (DR) 

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def load_mask(mask_path):
    """load a binary mask from a file and convert it to a boolean"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 128).astype(np.uint8)
    return mask

def compute_iou(pred_mask, gt_mask):
    """Computer Intersection over Union (IoU) between predicted and ground truth mask"""

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    iou = intersection / union if union > 0 else 1

    return iou

def compute_detection_rate(pred_mask, gt_mask, threshold=0.5):
    """Check if the predicted mask meets the DR criteria"""
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    false_positives = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()

    #compute iou
    iou = intersection/union if union > 0 else 1
   
    #condition 1: iou >= threshold
    condition1 = iou >= threshold

    #condition 2: intersection with ground truth is greater than false positives
    condition2 = intersection >= false_positives 

    #compute detection
    dt = condition1 and condition2

    return dt

# paths to round truth and predicted masks

ground_truth_dir = "/home/yousef/Documents/event_segmentation/dataset/test/ground_truth_images/april_tag_bird_combinations"
prediction_dir = "/home/yousef/Documents/event_segmentation/dataset/test/ground_truth_images/april_tag_bird_combinations"

# list all mask files (assuming sorted order or matching filenames)
gt_files = ["gt_april_tag_bird_combinations.bag_0.414745961.png"]
pred_files = ["gt_april_tag_bird_combinations.bag_0.433401257.png"]#sorted(os.listdir(prediction_dir))

ious = []
successful_detections = 0
total_frames = len(gt_files)



for gt_file, pred_file in zip(gt_files, pred_files):

    gt_mask   = load_mask(os.path.join(ground_truth_dir, gt_file))
    pred_mask = load_mask(os.path.join(prediction_dir, pred_file))
    
    iou = compute_iou(pred_mask, gt_mask)
    ious.append(iou)

    dt = compute_detection_rate(pred_mask, gt_mask)

    if dt:
        successful_detections = successful_detections + 1

    print(f"Frame {gt_file}: IoU = {iou:.4f}")

#computer average IoU
mean_iou = np.mean(ious)
print(f"Mean IoU across dataset: {mean_iou:.4f}")

# compute final DR percentage
dr = (successful_detections / total_frames) * 100
print(f"Detection Rate: {dr:.2f}%")

