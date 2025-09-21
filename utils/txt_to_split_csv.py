import os
import pandas as pd

# --- Paths ---
scene_root = "/home/yousef/Documents/event_segmentation/dataset/test/recordings_h5/"  # Where the 6 scene folders are
gt_root = "/home/yousef/shared_for_test/test/ground_truth_images/"  # Where the GT txt files are
output_root = "dataset/test/csv_for_EMSMC/"  # Output base directory

# Only these folders are valid scenes
target_scenes = [
    "carpet2_drone_combinations",
    "carpet2_dji_combinations",
    "carpet2_bird_combinations",
    "echo_bird_combinations",
    "city_helicopter_combinations",
    "city_plane_combinations"
]

events_per_file = 50000
events_filename = "scene.txt"

# Ensure output structure exists
os.makedirs(output_root, exist_ok=True)

for scene in target_scenes:
    print(f"📦 Processing scene: {scene}")

    scene_dir = os.path.join(scene_root, scene)
    time_dir = os.path.join(gt_root, scene)
    gt_file = os.path.join(time_dir, f"{scene}.txt")
    output_dir = os.path.join(output_root, scene)
    os.makedirs(output_dir, exist_ok=True)

    # Check if required files exist
    scene_txt_path = os.path.join(scene_dir,f"{scene}.txt")
    if not os.path.exists(scene_txt_path):
        print(f"❌ Missing events file: {scene_txt_path}")
        continue
    if not os.path.exists(gt_file):
        print(f"❌ Missing GT file: {gt_file}")
        continue

    # Load events
    events = pd.read_csv(scene_txt_path, header=None, delim_whitespace=True)
    events[0] = events[0].astype(int)  # ensure timestamp column is int (ns)

    # Load GT timestamps (seconds → nanoseconds)
    with open(gt_file, 'r') as f:
        timestamps_ns = [int(float(line.strip()) * 1e9) for line in f]

    for ts in timestamps_ns:
        subset = events[events[0] >= ts].head(events_per_file)
        if subset.empty:
            print(f"⚠️  No events found at {ts} in {scene}")
            continue

        out_filename = f"{scene}_{ts}.csv"
        out_path = os.path.join(output_dir, out_filename)
        subset.to_csv(out_path, index=False, header=False)
        print(f"✅ Saved: {out_filename}")
