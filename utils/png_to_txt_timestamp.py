import os
import re

def extract_timestamp(filename):
    match = re.search(r'background_foreground_combinations\.(\d+)\.png', filename)
    return match.group(1) if match else None

def process_folders(parent_folder):
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path) and folder.startswith("background_foreground_combinations"):
            txt_filename = os.path.join(folder_path, "background_foreground_combinations.txt")
            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    timestamp = extract_timestamp(file)
                    if timestamp:
                        with open(txt_filename, "a") as txt_file:
                            txt_file.write(timestamp + "\n")
                        print(f"Stored timestamp {timestamp} in {txt_filename}")

# Example usage:
parent_folder = "/home/yousef/shared_for_test/test/ground_truth_images_2/"  # Change this to your parent folder path
process_folders(parent_folder)
