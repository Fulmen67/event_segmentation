import argparse
import os

import cv2
import hdf5plugin
import h5py
import numpy as np
import yaml

from h5_packager import H5Packager


# TODO: add option to remove hot pixels already in the preprocessing stage
def process(path, args, original_res=(480, 640)):
    filename, _ = os.path.split(path)
    if path.split("/")[-1][0] == ".":
        return

    # open original file, create new file
    file = h5py.File(path, "r")
    ep = H5Packager(args.output_dir + filename.split("/")[-1] + ".h5")

    # load recitifaction map
    rectify_file = h5py.File(filename + "/rectify_map.h5")
    rectify_map = rectify_file["rectify_map"][()]

    # extract events
    print("Processing events...")

    t0 = -1
    idx = 0
    sensor_size = None
    max_events = 1000000
    num_pos, num_neg, last_timestamp = 0, 0, 0

    t_offset = file["t_offset"][()]

    while True:

        # event timestamps
        timestamp = (file["events/t"][idx : idx + max_events] + t_offset) / 1e6
        timestamp = timestamp.astype(np.float64)
        if t0 == -1:
            t0 = timestamp[0]
        last_timestamp = timestamp[-1]

        # event distorted pixel location and polarity
        x = file["events/x"][idx : idx + max_events].astype(np.int16)
        y = file["events/y"][idx : idx + max_events].astype(np.int16)
        p = file["events/p"][idx : idx + max_events].astype(np.bool_)
        if len(x) < max_events:
            break

        # convert polarity from (-1, 1) to (0, 1)
        p[p < 0] = 0

        # package events and update statistics
        ep.package_events(x, y, timestamp, p)
        num_pos += p.sum()
        num_neg += len(p) - p.sum()
        idx += max_events

    # get sensor size from latest package of events (can be wrong, but we're not using it)
    if sensor_size is None:
        sensor_size = original_res
        print("Sensor size inferred from events as {}".format(sensor_size))

    # disparity maps
    disp_cnt = 0
    print("Processing disparity...")
    disparity_dir = filename + "/disparity/"
    if os.path.isdir(disparity_dir):
        disp_timestamp = np.genfromtxt(disparity_dir + "timestamps.txt")
        disp_timestamp /= 1e6

        for i in range(disp_timestamp.shape[0]):
            idx = i * 2  # image index jumps by 2
            disp_filename = disparity_dir + "{:06d}.png".format(idx)
            disp_16bit = cv2.imread(str(disp_filename), cv2.IMREAD_ANYDEPTH)
            dispmap = disp_16bit.astype("float32") / 256.0
            ep.package_disparity(dispmap, disp_timestamp[i], disp_cnt)
            disp_cnt += 1

    # optical flow maps
    flow_cnt = 0
    print("Processing optical flow...")
    flow_dir = filename + "/optical_flow_forward/"
    if os.path.isdir(flow_dir):
        flow_timestamp = np.genfromtxt(flow_dir + "timestamps.txt", skip_header=1, delimiter=",")

        flow_idx = []
        eval_sequence = False
        if flow_timestamp.shape[1] == 3:  # eval sequence
            eval_sequence = True
            flow_idx = flow_timestamp[:, -1].astype(int).tolist()
            flow_timestamp = flow_timestamp[:, :2]
        flow_timestamp /= 1e6
        t_offset /= 1e6

        # get files in directory
        flow_files = os.listdir(flow_dir)
        if len(flow_idx) == 0:
            for f in flow_files:
                if f.endswith(".png") and f[0] != ".":
                    flow_idx.append(int(os.path.splitext(f)[0]))
        flow_idx.sort()

        # augment sequences and timestamps
        if eval_sequence:
            flow_flag = []
            flow_idx_aug = []
            flow_timestamp_aug = []

            # startup period of 1.0 second before the valid GT
            for i in range(10, 0, -1):
                flow_flag.append(0)
                flow_idx_aug.append(flow_idx[0] - i * 2)
                flow_timestamp_aug.append(
                    [flow_timestamp[0, 1] - (i + 1) * 0.1, flow_timestamp[0, 1] - i * 0.1]
                )  # hardcoded 10Hz GT

            for i in range(len(flow_idx)):

                if flow_idx[i] - flow_idx_aug[-1] > 10:

                    j = 1
                    idx_aug = []
                    timestamp_aug = []
                    while flow_idx_aug[-1] + 2 * j < flow_idx[i]:
                        flow_flag.append(0)
                        idx_aug.append(flow_idx_aug[-1] + 2 * j)
                        tmp = [flow_timestamp_aug[-1][0] + j * 0.1, flow_timestamp_aug[-1][0] + (j + 1) * 0.1]
                        if tmp[1] > flow_timestamp[i, 0]:
                            tmp[1] = flow_timestamp[i, 0]
                        timestamp_aug.append(tmp)
                        j += 1
                    flow_idx_aug.extend(idx_aug)
                    flow_timestamp_aug.extend(timestamp_aug)

                flow_flag.append(1)
                flow_idx_aug.append(flow_idx[i])

                flow_flag.extend([0 for _ in range(1, 5)])
                flow_idx_aug.extend([flow_idx[i] + 2 * j for j in range(1, 5)])

                flow_timestamp_aug.append(flow_timestamp[i, :].tolist())
                for j in range(0, 4):
                    flow_timestamp_aug.append(
                        [flow_timestamp[i, 1] + j * 0.1, flow_timestamp[i, 1] + (j + 1) * 0.1]
                    )  # hardcoded 10Hz GT

            flow_idx = flow_idx_aug
            flow_timestamp = np.asarray(flow_timestamp_aug)
            flow_flag = np.asarray(flow_flag)
            np.save(args.output_dir + filename.split("/")[-1] + "_flag.npy", flow_flag)

        ts_idx = 0
        for idx in flow_idx:
            flow_filename = flow_dir + "{:06d}.png".format(idx)
            if not os.path.isfile(flow_filename):
                fake_gt = np.zeros((original_res[0], original_res[1], 3), dtype=np.float32)
                cv2.imwrite(flow_filename, fake_gt)

            flow_16bit = cv2.imread(str(flow_filename), -1)
            flow_16bit = np.flip(flow_16bit, axis=-1)
            assert flow_16bit[..., 2].max() <= 1, f"Maximum value in last channel should be 1: {flow_filename}"

            valid2D = flow_16bit[..., 2] == 1
            valid_map = np.where(valid2D)

            flow_16bit = flow_16bit.astype("float32")
            flow_map = np.zeros((flow_16bit.shape[0], flow_16bit.shape[1], 2))
            flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2**15) / 128
            flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2**15) / 128

            ep.package_flow(flow_map, flow_timestamp[ts_idx, :], flow_cnt)
            flow_cnt += 1
            ts_idx += 1

    # camera intrinsics and extrinsics
    print("Processing camera data...")
    with open(filename + "/cam_to_cam.yaml", "r") as stream:
        cam_to_cam = yaml.safe_load(stream)
        ep.package_cam(cam_to_cam)

    # rectification map
    print("Processing rectification map...")
    ep.package_rectification(rectify_map)

    ep.add_metadata(
        num_pos,
        num_neg,
        last_timestamp - t0,
        t0,
        last_timestamp,
        disp_cnt,
        flow_cnt,
        # sensor_size,
        # norect_range,
    )

    file.close()


if __name__ == "__main__":
    """
    Tool for converting DSEC.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output_dir", default="/tmp/extracted_data")
    args = parser.parse_args()

    # get files to process
    paths = []
    for root, dirs, files in os.walk(args.path):
        for file in files:
            if file.endswith("events.h5"):
                paths.append(os.path.join(root, file))

    # make sure output directory exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for path in paths:
        process(path, args)
