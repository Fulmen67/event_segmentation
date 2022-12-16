import os
import hdf5plugin
import h5py
import argparse
import numpy as np

from h5_packager import H5Packager


def binary_search_array(array, x, l=None, r=None, side="left"):
    """
    Binary search through a sorted array.
    """

    l = 0 if l is None else l
    r = len(array) - 1 if r is None else r
    mid = l + (r - l) // 2

    if l > r:
        return l if side == "left" else r

    if array[mid] == x:
        return mid
    elif x < array[mid]:
        return binary_search_array(array, x, l=l, r=mid - 1)
    else:
        return binary_search_array(array, x, l=mid + 1, r=r)


def find_ts_index(file, timestamp, dataset="events/ts"):
    idx = binary_search_array(file[dataset], timestamp)
    return idx


def get_events(file, idx0, idx1):
    xs = file["events/xs"][idx0:idx1]
    ys = file["events/ys"][idx0:idx1]
    ts = file["events/ts"][idx0:idx1]
    ps = file["events/ps"][idx0:idx1]
    return xs, ys, ts, ps


if __name__ == "__main__":
    """
    Tool for generating a training dataset out of a set of specified group of H5 datasets.
    The original sequences are cropped both in time and space.
    The resulting training sequences contain the raw images if available.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="directory of datasets to be used")
    parser.add_argument(
        "--output_dir",
        default="/tmp/training/",
        help="output directory containing the resulting training sequences",
    )
    parser.add_argument(
        "--mode",
        default="time",
        help="splits sequences based on: events/time",
    )
    parser.add_argument(
        "--window",
        default=2,
        help="events/time window used to split the sequences",
        type=int,
    )
    parser.add_argument("--output_res", default=(240, 240))
    parser.add_argument("--random_crop", default=True)
    parser.add_argument("--original_res", default=(480, 640))
    args = parser.parse_args()

    # spatial crop
    assert (args.output_res[0] <= args.original_res[0]) and (args.output_res[1] <= args.original_res[1])
    crop = ((args.original_res[0] - args.output_res[0]) // 2, (args.original_res[1] - args.output_res[1]) // 2)
    if args.random_crop:
        crop = (
            np.random.randint(args.original_res[0] - args.output_res[0]),
            np.random.randint(args.original_res[1] - args.output_res[1]),
        )

    print("Data will be extracted in folder: {}".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    path_from = []
    for root, dirs, files in os.walk(args.path):
        for file in files:
            if file.endswith(".h5"):
                path_from.append(os.path.join(root, file))

    # process dataset
    for path in path_from:
        hf = h5py.File(path, "r")
        print("Processing:", path)
        filename = path.split("/")[-1].split(".")[0]

        rectify_map = hf["rectification/rectify_map"][:]
        x_min = np.argwhere(rectify_map[..., 0] >= crop[1])[:, 1].min()
        x_max = np.argwhere(rectify_map[..., 0] < crop[1] + args.output_res[1] - 1)[:, 1].max()
        y_min = np.argwhere(rectify_map[..., 1] >= crop[0])[:, 0].min()
        y_max = np.argwhere(rectify_map[..., 1] < crop[0] + args.output_res[0] - 1)[:, 0].max()
        norect_range = [(y_min, y_max + 1), (x_min, x_max + 1)]

        # start reading sequence
        row = 0
        if args.mode == "events":
            idx0 = 0
        elif args.mode == "time":
            row = hf["events/ts"][0]  # s
            idx0 = find_ts_index(hf, row)

        sequence_id = 0
        xs_seq, ys_seq, ts_seq, ps_seq = [], [], [], []
        while True:
            rectify_map = hf["rectification/rectify_map"][:]
            if args.mode == "events":
                idx1 = row + args.window
            elif args.mode == "time":
                idx1 = find_ts_index(hf, row + args.window)

            # events in temporal window
            xs, ys, ts, ps = get_events(hf, idx0, idx1)
            if len(xs) == 0:
                break

            # corresponding event rectified pixel location
            rectified_grid = rectify_map[ys, xs]
            rectified_x = rectified_grid[:, 0]
            rectified_y = rectified_grid[:, 1]

            # events in spatial window (cropping in the horizontal camera axis)
            x_out = np.argwhere(rectified_x < crop[1])
            x_out = np.concatenate((x_out, np.argwhere(rectified_x >= crop[1] + args.output_res[1] - 1)), axis=0)
            xs = np.delete(xs, x_out)
            ys = np.delete(ys, x_out)
            ps = np.delete(ps, x_out)
            ts = np.delete(ts, x_out)

            rectified_x = np.delete(rectified_x, x_out)
            rectified_y = np.delete(rectified_y, x_out)

            # events in spatial window (cropping in the vertical camera axis)
            y_out = np.argwhere(rectified_y < crop[0])
            y_out = np.concatenate((y_out, np.argwhere(rectified_y >= crop[0] + args.output_res[0] - 1)), axis=0)
            xs = np.delete(xs, y_out)
            ys = np.delete(ys, y_out)
            ps = np.delete(ps, y_out)
            ts = np.delete(ts, y_out)

            rectified_x = np.delete(rectified_x, y_out)
            rectified_y = np.delete(rectified_y, y_out)

            # fix event location in the cropped window
            xs -= norect_range[1][0]
            ys -= norect_range[0][0]

            x_out = np.argwhere(xs < 0)
            x_out = np.concatenate((x_out, np.argwhere(xs >= args.output_res[1])), axis=0)
            xs = np.delete(xs, x_out)
            ys = np.delete(ys, x_out)
            ps = np.delete(ps, x_out)
            ts = np.delete(ts, x_out)
            rectified_x = np.delete(rectified_x, x_out)
            rectified_y = np.delete(rectified_y, x_out)

            y_out = np.argwhere(ys < 0)
            y_out = np.concatenate((y_out, np.argwhere(ys >= args.output_res[0])), axis=0)
            xs = np.delete(xs, y_out)
            ys = np.delete(ys, y_out)
            ps = np.delete(ps, y_out)
            ts = np.delete(ts, y_out)
            rectified_x = np.delete(rectified_x, y_out)
            rectified_y = np.delete(rectified_y, y_out)

            # append to sequence events
            xs_seq.extend(xs.tolist())
            ys_seq.extend(ys.tolist())
            ts_seq.extend(ts.tolist())
            ps_seq.extend(ps.tolist())

            if args.mode == "time" and xs.shape[0] <= 10:
                print("Empty subsequence, recomputing crop")
                xs_seq, ys_seq, ts_seq, ps_seq = [], [], [], []
                if args.random_crop:
                    crop = (
                        np.random.randint(args.original_res[0] - args.output_res[0]),
                        np.random.randint(args.original_res[1] - args.output_res[1]),
                    )
                x_min = np.argwhere(rectify_map[..., 0] >= crop[1])[:, 1].min()
                x_max = np.argwhere(rectify_map[..., 0] < crop[1] + args.output_res[1] - 1)[:, 1].max()
                y_min = np.argwhere(rectify_map[..., 1] >= crop[0])[:, 0].min()
                y_max = np.argwhere(rectify_map[..., 1] < crop[0] + args.output_res[0] - 1)[:, 0].max()
                norect_range = [(y_min, y_max + 1), (x_min, x_max + 1)]
                continue

            # store data
            if (args.mode == "events" and len(xs_seq) >= args.window) or args.mode == "time":

                if args.mode == "events":
                    t0 = ts[0]
                    t1 = ts[-1]
                    xs_store = xs_seq[: args.window]
                    ys_store = ys_seq[: args.window]
                    ts_store = ts_seq[: args.window]
                    ps_store = ps_seq[: args.window]
                    xs_seq = xs_seq[args.window :]
                    ys_seq = ys_seq[args.window :]
                    ts_seq = ts_seq[args.window :]
                    ps_seq = ps_seq[args.window :]
                elif args.mode == "time":
                    t0 = row
                    t1 = row + args.window
                    xs_store = xs_seq
                    ys_store = ys_seq
                    ts_store = ts_seq
                    ps_store = ps_seq
                    xs_seq = []
                    ys_seq = []
                    ts_seq = []
                    ps_seq = []

                # store event subsequence
                ep = H5Packager(args.output_dir + filename + "_" + str(sequence_id) + ".h5")
                ep.package_events(xs_store, ys_store, ts_store, ps_store)

                # store rectification map
                rectify_map_crop = rectify_map[
                    norect_range[0][0] : norect_range[0][1], norect_range[1][0] : norect_range[1][1], :
                ]
                rectify_map_crop[:, :, 0] -= crop[1]
                rectify_map_crop[:, :, 1] -= crop[0]
                ep.package_rectification(rectify_map_crop)

                # store camera data
                intrinsics = eval(hf["calibration/intrinsics"][()])
                intrinsics["cam0"]["camera_matrix"][2] -= norect_range[1][0]
                intrinsics["cam0"]["camera_matrix"][3] -= norect_range[0][0]
                intrinsics["camRect0"]["camera_matrix"][2] -= crop[1]
                intrinsics["camRect0"]["camera_matrix"][3] -= crop[0]
                extrinsics = eval(hf["calibration/extrinsics"][()])
                disparity_to_depth = eval(hf["calibration/disparity_to_depth"][()])
                disparity_to_depth["cams_03"][0][3] = -intrinsics["camRect0"]["camera_matrix"][2]
                disparity_to_depth["cams_03"][1][3] = -intrinsics["camRect0"]["camera_matrix"][3]
                cam_to_cam = {}
                cam_to_cam["intrinsics"] = intrinsics
                cam_to_cam["extrinsics"] = extrinsics
                cam_to_cam["disparity_to_depth"] = disparity_to_depth
                ep.package_cam(cam_to_cam)

                # subsequence duration
                duration = ts_store[-1] - ts_store[0]
                if args.mode == "time":
                    duration = args.window

                tmp_ps = np.asarray(ps_store)
                ep.add_metadata(
                    len(tmp_ps[tmp_ps > 0]),
                    len(tmp_ps[tmp_ps < 0]),
                    duration,
                    ts_store[0],
                    ts_store[-1],
                    0,
                    0,
                    # hf.attrs["sensor_resolution"],
                    # hf.attrs["norect_range"],
                )

                ep.file.close()

                sequence_id += 1
                if args.random_crop:
                    crop = (
                        np.random.randint(args.original_res[0] - args.output_res[0]),
                        np.random.randint(args.original_res[1] - args.output_res[1]),
                    )
                x_min = np.argwhere(rectify_map[..., 0] >= crop[1])[:, 1].min()
                x_max = np.argwhere(rectify_map[..., 0] < crop[1] + args.output_res[1] - 1)[:, 1].max()
                y_min = np.argwhere(rectify_map[..., 1] >= crop[0])[:, 0].min()
                y_max = np.argwhere(rectify_map[..., 1] < crop[0] + args.output_res[0] - 1)[:, 0].max()
                norect_range = [(y_min, y_max + 1), (x_min, x_max + 1)]

            row += args.window
            idx0 = idx1

        hf.close()
        print("")
