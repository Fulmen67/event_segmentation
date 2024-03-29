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
def process(path, args, original_res=(180, 240)):
    _, filename = os.path.split(path)

    # open original file, create new file
    ep = H5Packager(args.output_dir + filename.split(".")[0] + ".h5")

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
        sensor_size,
        [(-1, -1), (-1, -1)],
    )


if __name__ == "__main__":
    """
    Tool for converting UZH-FPV Drone Racing Dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output_dir", default="/tmp/extracted_data")
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

    for path in paths:
        process(path, args)
