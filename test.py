#!/usr/bin/env python
import os
import torch
import pprint
import argparse
import importlib

import matplotlib
matplotlib.use("Agg")

from nnet.py_factory import NetworkFactory
from mmcv import Config
import pdb

torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def test(cfg, testiter, debug=False):
    result_dir = os.path.join(cfg.test_cfg.save_dir, str(testiter))

    make_dirs([result_dir])

    test_iter = cfg.train_cfg.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))

    print("building neural network...")
    nnet = NetworkFactory(cfg)
    print("loading parameters...")
    nnet.load_params(test_iter)

    test_file = "test.{}".format(cfg.test_cfg.sample_module)
    testing = importlib.import_module(test_file).testing

    nnet.cuda()
    nnet.eval_mode()
    testing(cfg, nnet, result_dir, debug=debug)

if __name__ == "__main__":
    args = parse_args()

    print("cfg_file: {}".format(args.cfg_file))

    cfg_file = Config.fromfile(args.cfg_file)
    cfg_file.test_cfg.test = True

    print("system config...")
    print(f'Config:\n{cfg_file.pretty_text}')

    test(cfg_file, args.testiter, args.debug)
