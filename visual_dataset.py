import argparse

import mlflow
import numpy as np
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader

from utils.visualization import Visualization, vis_activity




def visualize_dataset(args, config_parser):

    config = config_parser.config

     # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool
    if config["vis"]["enabled"] or config["vis"]["store"]:
        vis = Visualization(config, eval_id=-1, path_results=None)
    
    data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])   
    dataloader = torch.utils.data.DataLoader(                                                 
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )
    pass

    end_test = False
    while True:
        for inputs in dataloader:
            
            # finish loop
            if data.seq_num >= len(data.files):
                end_test = True
                break

            # visualize
            if config["vis"]["bars"]:
                for bar in data.open_files_bar:
                    bar.next()
            if config["vis"]["enabled"]:
                vis.update(inputs, None, None, None, None) 
        if end_test:
                break
    if config["vis"]["bars"]:
        for bar in data.open_files_bar:
            bar.finish()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eval_EVIMO-2.yml",
        help="config file, overwrites mlflow settings",
    )
    
    args = parser.parse_args()

    # launch visualization
    visualize_dataset(args, YAMLParser(args.config))