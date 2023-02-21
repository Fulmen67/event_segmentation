import argparse
import os

import cv2
import hdf5plugin
import h5py
import numpy as np
import yaml

from h5_packager import H5Packager
import pandas as pd

# TODO: add option to remove hot pixels already in the preprocessing stage
def process(path, args, original_res=(480, 640), crop = False):
    folders, filename = os.path.split(path)

    # open original file, create new file
    # get the names of folders in a list
    
    folders = folders.split("/")
    #output_path = "_".join(folders[4:9])
    ep = H5Packager(args.output_dir + "_".join(folders[4:9]) + ".h5")
    #ep = H5Packager(args.output_dir + filename.split(".")[0] + ".h5")
    t0 = -1
    idx = 0
    sensor_size = None
    max_events = 10000000
    num_pos, num_neg, last_timestamp = 0, 0, 0

    print("Processing events...")
    skip = 1
    while True:

        # read events
        events = np.loadtxt(path, dtype=np.float64, delimiter=" ", skiprows=skip, max_rows=max_events)
        if events.shape[0] == 0:
            break
        skip += events.shape[0]

        # event timestamps
        timestamp = events[:, 0]

        # event pixel location and polarity
        x = events[:, 1].astype(np.int16)
        y = events[:, 2].astype(np.int16)
        p = events[:, 3].astype(np.bool_)
        
        
        if crop:
            # Define the original and output resolutions
            original_res = (480, 640)
            output_res = (240, 240)

            # Assert that the output resolution is not larger than the original resolution
            assert (output_res[0] <= original_res[0]) and (output_res[1] <= original_res[1])

            # Calculate the crop values to take the center of the events
            crop = ((original_res[0] - output_res[0]) // 2, (original_res[1] - output_res[1]) // 2)

            # Create a boolean mask for the events within the crop area
            cropped_mask = np.logical_and(x >= crop[0], x < crop[0]+output_res[0]) & np.logical_and(y >= crop[1], y < crop[1]+output_res[1])

            # Get the cropped events with the mask
            cropped_x = x[cropped_mask]
            cropped_y = y[cropped_mask]
            cropped_p = p[cropped_mask]
            cropped_timestamps = timestamp[cropped_mask]
            
            timestamp = cropped_timestamps
            x = cropped_x
            y = cropped_y
            p = cropped_p
        
        if t0 == -1:
            t0 = timestamp[0]
        last_timestamp = timestamp[-1]

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
        #sensor_size,
        #[(-1, -1), (-1, -1)],
    )


if __name__ == "__main__":
    """
    Tool for converting EVIMO2 Dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/data/youssef/datasets/evimo_2/")
    parser.add_argument("--output_dir", default="/data/youssef/datasets/evimo_2_h5/eval/")
    args = parser.parse_args()

    # get files to process
    paths = []
    for root, dirs, files in os.walk(args.path):
        for file in files:
            # skip eval folders
            if "train" in root:
                continue
            if file.endswith("events.txt"):
                paths.append(os.path.join(root, file))
    
    # make sure output directory exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for path in paths:
        process(path, args)
