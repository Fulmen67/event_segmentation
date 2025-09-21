"""
Adapted from TUDelft-MAVLab https://github.com/tudelft/event_flow
"""

import argparse
import os

import cv2
import hdf5plugin
import h5py
import numpy as np
import yaml

from h5_packager import H5Packager


# TODO: add option to remove hot pixels already in the preprocessing stage
def process(path, output_path):
    _, filename = os.path.split(path)
    folder_name = os.path.basename(os.path.dirname(path))

    # open original file, create new file
    ep = H5Packager(os.path.join(output_path, folder_name + ".h5"))

    t0 = -1
    idx = 0
    sensor_size = None
    max_events = 10000000
    num_pos, num_neg, last_timestamp = 0, 0, 0

    print("Processing {path}...")
    skip = 0
    while True:

        # read events
        events = np.loadtxt(path, dtype=np.float64, delimiter=" ", skiprows=skip, max_rows=max_events)
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


if __name__ == "__main__":
    """
    Tool for converting EED Dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default = "/home/yousef/Documents/EED/")
    parser.add_argument("--output_dir", default="/home/yousef/Documents/EED/")
    args = parser.parse_args()

    # Iterate over each folder in EED/
    for folder in os.listdir(args.path):
        folder_path = os.path.join(args.path, folder)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            txt_file = os.path.join(folder_path, "events_filtered.txt")
            if os.path.exists(txt_file):
                process(txt_file, folder_path)  # Save .h5 inside the same folder
