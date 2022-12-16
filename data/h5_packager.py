"""
Adapted from Monash University https://github.com/TimoStoff/events_contrast_maximization
"""

import hdf5plugin
import h5py
import numpy as np


class H5Packager:
    def __init__(self, output_path):
        print("Creating file in {}".format(output_path))
        self.output_path = output_path
        self.imu_ts = None

        self.file = h5py.File(output_path, "w")
        self.event_xs = self.file.create_dataset(
            "events/xs", (0,), dtype=np.dtype(np.int16), maxshape=(None,), chunks=True, **hdf5plugin.Zstd()
        )
        self.event_ys = self.file.create_dataset(
            "events/ys", (0,), dtype=np.dtype(np.int16), maxshape=(None,), chunks=True, **hdf5plugin.Zstd()
        )
        self.event_ts = self.file.create_dataset(
            "events/ts", (0,), dtype=np.dtype(np.float64), maxshape=(None,), chunks=True, **hdf5plugin.Zstd()
        )
        self.event_ps = self.file.create_dataset(
            "events/ps", (0,), dtype=np.dtype(np.bool_), maxshape=(None,), chunks=True, **hdf5plugin.Zstd()
        )

    def append(self, dataset, data):
        dataset.resize(dataset.shape[0] + len(data), axis=0)
        if len(data) == 0:
            return
        dataset[-len(data) :] = data[:]

    def package_events(self, xs, ys, ts, ps):
        self.append(self.event_xs, xs)
        self.append(self.event_ys, ys)
        self.append(self.event_ts, ts)
        self.append(self.event_ps, ps)

    def package_flow(self, flowmap, timestamp, flow_idx):
        flowmap_dset = self.file.create_dataset(
            "flow/flowmap{:09d}".format(flow_idx), data=flowmap, dtype=np.dtype(np.float64), **hdf5plugin.Zstd()
        )
        flowmap_dset.attrs["size"] = flowmap.shape
        flowmap_dset.attrs["timestamp_from"] = timestamp[0]
        flowmap_dset.attrs["timestamp_to"] = timestamp[1]

    def package_disparity(self, dispmap, timestamp, disp_idx):
        flowmap_dset = self.file.create_dataset(
            "disp/dispmap{:09d}".format(disp_idx), data=dispmap, dtype=np.dtype(np.float64), **hdf5plugin.Zstd()
        )
        flowmap_dset.attrs["size"] = dispmap.shape
        flowmap_dset.attrs["timestamp"] = timestamp

    def package_cam(self, d):
        for k, v in d.items():
            self.file.create_dataset("calibration/" + k, data=str(v), dtype=h5py.special_dtype(vlen=str))

    def package_rectification(self, rectify_map):
        self.file.create_dataset(
            "rectification/rectify_map", data=rectify_map, dtype=np.dtype(np.float64), **hdf5plugin.Zstd()
        )

    def add_metadata(
        self,
        num_pos,
        num_neg,
        duration,
        t0,
        tk,
        num_dispmaps,
        num_flowmaps,
        # sensor_size,
        # norect_range,
    ):
        self.file.attrs["num_events"] = num_pos + num_neg
        self.file.attrs["num_pos"] = num_pos
        self.file.attrs["num_neg"] = num_neg
        self.file.attrs["duration"] = duration
        self.file.attrs["t0"] = t0
        self.file.attrs["tk"] = tk
        self.file.attrs["num_dispmaps"] = num_dispmaps
        self.file.attrs["num_flowmaps"] = num_flowmaps
        # self.file.attrs["sensor_resolution"] = sensor_size
        # self.file.attrs["norect_range"] = norect_range
