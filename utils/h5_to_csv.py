import argparse
import pandas as pd
import os

def convert_txt_to_csv(txt_path, output_dir):
    filename = os.path.splitext(os.path.basename(txt_path))[0]
    csv_path = os.path.join(output_dir, f"{filename}.csv")

    # Read the txt file assuming space-separated values and no header
    df = pd.read_csv(txt_path, sep=" ", header=None)
    df.columns = ["ts", "x", "y", "p"]  # Rename columns if needed

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(csv_path, index=False, header=False)
    print(f"Converted {txt_path} to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/home/yousef/Documents/event_segmentation/dataset/test/recordings_h5/echo_bird_combinations")
    parser.add_argument("--output_dir", default="/home/yousef/Documents/event_segmentation/dataset/test/recordings_h5/echos_bird_combinations")
    args = parser.parse_args()

    # get files to process
    paths = []
    for root, dirs, files in os.walk(args.path):
        for file in files:
            if file.endswith(".txt"):
                paths.append(os.path.join(root, file))
    
    # make sure output directory exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # process files
    for path in paths:
        convert_txt_to_csv(path, args.output_dir)
