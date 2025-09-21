import os
import re

# Regex pattern to extract timestamps from filenames
TIMESTAMP_PATTERN = re.compile(r'.*\.bag_(\d+\.\d+)\.png')

def extract_timestamps(folder_path):
    """Extracts timestamps from filenames in a given background_foreground folder."""
    timestamps = []
    
    for file_name in sorted(os.listdir(folder_path)):  # Sort to maintain order
        print(f"Checking file: {file_name}")  # Debugging print
        
        match = TIMESTAMP_PATTERN.match(file_name)
        if match:
            timestamps.append(float(match.group(1)))  # Convert timestamp to float
    
    print(f"Extracted timestamps: {timestamps}")  # Debugging print
    return timestamps

def process_test_set(root_dir):
    """Processes all background_foreground folders and saves timestamps with correct naming."""
    print(f"Checking folders in {root_dir}: {os.listdir(root_dir)}")  # Debugging print

    for folder_name in sorted(os.listdir(root_dir)):  # Iterate over background_foreground folders
        folder_path = os.path.join(root_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue  # Skip non-directory files
        
        print(f"Processing folder: {folder_name}")  # Debugging print

        # Remove old timestamps.txt files if they exist
        old_timestamps_file = os.path.join(folder_path, "timestamps.txt")
        if os.path.exists(old_timestamps_file):
            os.remove(old_timestamps_file)
            print(f"Deleted old timestamps file: {old_timestamps_file}")

        # Extract timestamps
        timestamps = extract_timestamps(folder_path)

        if timestamps:
            # Save timestamps in a file named after the folder inside the same folder
            timestamps_file = os.path.join(folder_path, f"{folder_name}.txt")
            with open(timestamps_file, "w") as f:
                for ts in timestamps:
                    f.write(f"{ts}\n")
            
            print(f"Saved timestamps to {timestamps_file}")
        else:
            print(f"No timestamps found for {folder_path}")

if __name__ == "__main__":
    dataset_path = "/home/yousef/shared_for_test/test/ground_truth_images_2/"
    process_test_set(dataset_path)
