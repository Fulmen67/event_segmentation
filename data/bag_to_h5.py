import argparse
import os

import cv2
import hdf5plugin
import h5py
import numpy as np
import yaml
import rosbag

from h5_packager import H5Packager
import pandas as pd

# TODO: add option to remove hot pixels already in the preprocessing stage
def process(path, args, original_res=(480, 640)):
    
    filename = os.path.basename(path).replace(".bag", "")
    background_name = filename.split("_")[0]
    background_output_dir = os.path.join(args.output_dir, background_name)
    os.makedirs(background_output_dir, exist_ok=True)
    h5_path = background_output_dir + "/"  + filename + ".h5"
    if os.path.isfile(h5_path):
        #print that h5 file already exists
        print(filename,".h5 already exists")
        return
    ep = H5Packager(h5_path)
    t0 = -1
    idx = 0
    sensor_size = None
    max_events = 10000000
    num_pos, num_neg, last_timestamp = 0, 0, 0

    print("Processing events...")
    skip = 1
    while True:
        # Open the bag file
        bag = rosbag.Bag(path)
        
        # Get the event messages from the bag file
        events = bag.read_messages(topics='/cam0/events')

        # Convert the messages to a list of dictionaries
        event_list = []
        for msg in events:
            for event in msg.message.events:
                event_dict = {
                    'ts': msg.timestamp.to_nsec(),
                    'x': event.x,
                    'y': event.y,
                    'p': 0 if event.polarity == False else 1
                }
                event_list.append(event_dict)
        
        # Create a DataFrame object from the list of dictionaries
        df = pd.DataFrame(event_list)

        # Save the event messages to a CSV file
        path_txt = args.output_dir + filename.split(".")[0] + ".txt"
        df.to_csv(path_txt, index=False, header=False, sep=' ')

        events = np.loadtxt(path_txt, dtype=np.float64, delimiter=" ", skiprows=skip, max_rows=max_events)
        if events.shape[0] == 0:
            break
        skip += events.shape[0]

        # event timestamps
        timestamp = events[:, 0]
        if t0 == -1:
            t0 = timestamp[0]
        last_timestamp = timestamp[-1]

        # event pixel location and polarity
        x = events[:, 1].astype(np.int16)
        y = events[:, 2].astype(np.int16)
        p = events[:, 3].astype(np.bool_)

        # package events and update statistics
        ep.package_events(x, y, timestamp, p)
        num_pos += p.sum()
        num_neg += len(p) - p.sum()
        idx += max_events

    if sensor_size is None:
        sensor_size = [max(x) + 1, max(y) + 1]
        print("Sensor size inferred from events as {}".format(sensor_size))

    ep.add_metadata(
        num_pos,
        num_neg,
        last_timestamp - t0,
        t0,
        last_timestamp,
        0,
        0,
    )

    # delete txt file
    os.remove(path_txt)


if __name__ == "__main__":
    """
    Tool for converting EVIMO2 Dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/media/yousef/USB/bag_cb/")
    parser.add_argument("--output_dir", default="/home/yousef/Documents/event_segmentation/dataset/train/h5/new/")
    args = parser.parse_args()

    # get files to process
    paths = []
    for root, dirs, files in os.walk(args.path):
        for file in files:
            if file.endswith("computers_chair.bag"):
                paths.append(os.path.join(root, file))
    
    # make sure output directory exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # process files
    for path in paths:
        process(path, args)
