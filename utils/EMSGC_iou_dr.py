import os
import numpy as np
import cv2
import csv

def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 1

def compute_detection_rate(gt_mask, pred_mask, threshold=0.5):
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

def load_ground_truth(gt_path):
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    return gt_mask > 127  # Convert to binary (foreground is white)

def load_prediction(txt_path, shape):
    labels_data = np.loadtxt(txt_path, dtype=int)
    

    # Assuming the image size is known (width, height)
    image_width = 640  # Replace with actual width of the image
    image_height = 480  # Replace with actual height of the image

    # Initialize a blank (black) binary mask of the same size as the image
    binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Loop through the sparse label data and set corresponding pixels
    for x, y, label in labels_data:
        # Change any label above 1 to 1
        if label > 1:
            label = 1
        binary_mask[y, x] = label * 255  # Set the corresponding pixel to white for label 1

    # Optionally, apply morphological operations (dilation) to group regions together
    kernel = np.ones((3, 3), np.uint8)  # Kernel for morphological operations
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=3)  # Dilate to group pixels

    # Perform connected components labeling to group labels correctly
    num_labels, labels = cv2.connectedComponents(dilated_mask)

    # Create a binary mask from the connected components (only keeping label 1)
    # Labels are automatically indexed starting from 0, so we convert label 1 back to 255
    final_mask = np.zeros_like(dilated_mask)
    final_mask[labels == 1] = 255  # Only keep the largest connected component

    # Optionally, refine the mask with further morphological operations (optional)
    refined_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps
                
    return refined_mask



def process_sequence(gt_folder, pred_folder, sequence):
    iou_scores = []
    detection_rates = []
    timestamp_list = []
    
    
    for txt_file in sorted(os.listdir(pred_folder)):
        if not txt_file.endswith(".txt"):
            continue
        timestamp = txt_file.replace(".txt", "")
        gt_filename = f"gt_{sequence}.bag_{timestamp}.png"
        if sequence == "echo_drone_combinations":
            sequence = "echo_X_drone"
            gt_filename = f"gt_{sequence}.bag_{timestamp}.png"
        gt_path = os.path.join(gt_folder, gt_filename)
        txt_path = os.path.join(pred_folder, txt_file)
        

        if not os.path.exists(gt_path):
            print(f"Missing ground truth for {timestamp}")
            continue
            
        
        gt_mask = load_ground_truth(gt_path)
        pred_mask = load_prediction(txt_path, gt_mask.shape)
        
        iou_scores.append(compute_iou(gt_mask, pred_mask))
        detection_rates.append(compute_detection_rate(gt_mask, pred_mask))
        timestamp_list.append(int(float(timestamp)))
    """
    with open(csv_file_path, "a", newline="") as file:
        headers = ["Sequence Name", "Timestamp (s)", "IoU (EMSGC)", "DR (EMSGC)", "Window"]
        writer = csv.writer(file)

        if file.tell() == 0:
            writer.writerow(headers)
        
        for i in range(len(iou_scores)):
            writer.writerow([sequence, timestamp_list[i], iou_scores[i], detection_rates[i], 50000])"""

    return np.mean(iou_scores), np.mean(detection_rates)

if __name__ == "__main__":
    base_folder = "/home/yousef/shared_for_test/test_results_EMSGC"
    background_names = [ "april_tag", "brick", "carpet2", "city", "echo"]
    foreground_names = ["bird", "dji", "drone", "helicopter", "plane"]
    gt_root = "/home/yousef/shared_for_test/test/ground_truth_images"
    
    csv_file_path = "test_results/EMGSC/test_results_EMSGC.csv"
    csv_mean_file_path = "test_results/EMGSC/test_results_EMSGC_mean.csv"


    # check if the directory exists, if it doesn't create it
    if not os.path.exists(os.path.dirname(csv_file_path)):
        os.makedirs(os.path.dirname(csv_file_path
                                   ))

    all_iou = []
    all_detection_rates = []
    sequence_name_list = []
    
    for bg in background_names:
        for fg in foreground_names:
            sequence = f"{bg}_{fg}_combinations" 
            gt_folder = os.path.join(gt_root, sequence)
            pred_folder = os.path.join(base_folder, sequence, "seg_labels")
            
            if not os.path.exists(pred_folder) or not any(os.scandir(pred_folder)):
                print(f"Skipping {sequence}, missing predictions")
                continue
            
            mean_iou, mean_detection_rate = process_sequence(gt_folder, pred_folder, sequence)
            all_iou.append(mean_iou)
            all_detection_rates.append(mean_detection_rate)
            sequence_name_list.append(sequence)
            print(f"{sequence}: IoU={mean_iou:.4f}, Detection Rate={mean_detection_rate:.4f}")
    """
    with open(csv_mean_file_path, "a", newline="") as file:
        headers = ["Sequence Name", "Mean IoU (EMSGC)", "Mean (EMSGC)", "Window"]
        writer = csv.writer(file)

        if file.tell() == 0:
            writer.writerow(headers)
        
        for i in range(len(all_iou)):
            writer.writerow([sequence_name_list[i], all_iou[i], all_detection_rates[i], 50000])"""
    
    print("Overall Mean IoU:", np.mean(all_iou))
    print("Overall Mean Detection Rate:", np.mean(all_detection_rates))
