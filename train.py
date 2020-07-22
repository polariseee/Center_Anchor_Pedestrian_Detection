#!/usr/bin/env python
import os

import torch
import numpy as np
import queue
import argparse
import importlib
import threading
import traceback
import pickle
import json
import time
import random

from mmcv import Config
from mmcv.utils import get_logger
from nnet import NetworkFactory
from torch.multiprocessing import Process, Queue
import pdb
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train CAP")
    parser.add_argument("config", help="train config file path", type=str)
    parser.add_argument("--work-dir", help='the dir to save logs and models')
    parser.add_argument("--iter", dest="start_epoch",
                        help="train at epoch i",
                        default=0, type=int)

    args = parser.parse_args()
    return args


def prefetch_data(cfg, queue, sample_data, ped_data=None, emp_data=None):
    ind = 0
    n_ind = 0
    random.shuffle(ped_data)
    random.shuffle(emp_data)
    print("start prefetching data...")
    while True:
        try:
            data, ind, n_ind = sample_data(cfg.dataset, ped_data, ind, emp_data, n_ind)  # ind可以保证每个batch取得数据不相同
            queue.put(data)
        except Exception as e:
            traceback.print_exc()  # 打印出哪个文件哪个函数哪一行报的错
            raise e


def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return


def init_parallel_jobs(cfg, queue, fn, ped_data=None, emp_data=None):
    tasks = Process(target=prefetch_data, args=(cfg, queue, fn, ped_data, emp_data))
    # for task in tasks:
    #     task.daemon = True
    #     task.start()
    tasks.daemon = True
    tasks.start()
    return tasks


def train(logger, json_file, cfg, start_epoch=0):
    learning_rate    = cfg.train_cfg.learning_rate
    pretrained_model = cfg.train_cfg.pretrain
    display          = cfg.train_cfg.display
    sample_module    = cfg.train_cfg.sample_module
    iter_per_epoch   = cfg.train_cfg.iter_per_epoch
    num_epochs       = cfg.train_cfg.num_epochs
    batch_size       = cfg.dataset.batch_size

    # queues storing data for training

    training_queue   = Queue(cfg.train_cfg.prefetch_size)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(cfg.train_cfg.prefetch_size)

    # load data sampling function
    data_file   = "sample.{}".format(sample_module)
    sample_data = importlib.import_module(data_file).sample_data

    if cfg.train_cfg.cache_ped:
        with open(cfg.train_cfg.cache_ped, 'rb') as fid:
            ped_data = pickle.load(fid)
    if cfg.train_cfg.cache_emp:
        with open(cfg.train_cfg.cache_emp, 'rb') as fid:
            emp_data = pickle.load(fid)
    length_dataset = len(ped_data)+len(emp_data)
    logger.info('the length of dataset is: {}'.format(length_dataset))
    # allocating resources for parallel reading
    if cfg.train_cfg.cache_emp:
        training_tasks   = init_parallel_jobs(cfg, training_queue, sample_data, ped_data, emp_data)
    else:
        training_tasks = init_parallel_jobs(cfg, training_queue, sample_data, ped_data)
    # prefetch_data(cfg, training_queue, sample_data, ped_data, emp_data)

    training_pin_semaphore   = threading.Semaphore()
    training_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    logger.info("building model...")

    nnet = NetworkFactory(cfg)

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        logger.info("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_epoch:
        nnet.load_params(start_epoch)
        nnet.set_lr(learning_rate)
        logger.info("training starts from iteration {} with learning_rate {}".format(start_epoch, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    logger.info("training start...")
    nnet.cuda()
    nnet.train_mode()
    epoch_length = int(iter_per_epoch / batch_size)
    json_obj = open(json_file, 'w')
    loss_anchor = []
    loss_csp = []
    loss = []
    for epoch in range(start_epoch, num_epochs):
        for iteration in range(1, epoch_length + 1):
            training = training_queue.get(block=True)
            training_loss = nnet.train(**training)

            if isinstance(training_loss, tuple):
                loss_anchor.append(training_loss[0].item())
                loss_csp.append(training_loss[1].item())
                loss.append(training_loss[0].item() + training_loss[1].item())
            else:
                loss.append(training_loss.item())

            if display and iteration % display == 0:  # display = 5
                loss = np.array(loss)
                loss_anchor = np.array(loss_anchor)
                loss_csp = np.array(loss_csp)
                if loss_anchor.size > 0 and loss_csp.size > 0 and loss.size > 0:
                    logger.info("Epoch: {}/{}, loss_anchor: {:.5f}, loss_csp: {:.5f}, loss: {:.5f}".format(
                        epoch+1, num_epochs, loss_anchor.sum() / display, loss_csp.sum() / display, loss.sum() / display))
                    text = {"Epoch": epoch+1,
                            "loss_anchor": round(loss_anchor.sum() / display, 5),
                            "loss_csp": round(loss_csp.sum() / display, 5),
                            "loss": round(loss.sum() / display, 5)}
                else:
                    if 'anchor_head' in cfg:
                        logger.info("Epoch: {}/{}, loss_anchor: {:.5f}".format(epoch+1, num_epochs, loss.sum() / display))
                        text = {"Epoch": epoch+1, "loss_anchor": round(loss.sum() / display, 5)}
                    if 'kp_head' in cfg:
                        logger.info("Epoch: {}/{}, loss_csp: {:.5f}".format(epoch+1, num_epochs, loss.sum() / display))
                        text = {"Epoch": epoch+1, "loss_csp": round(loss.sum() / display, 5)}
                text = json.dumps(text)
                json_obj.write(text)
                json_obj.write('\r\n')
                loss_anchor = []
                loss_csp = []
                loss = []

            del training_loss

        nnet.save_params(epoch + 1)

    # sending signal to kill the thread
    training_pin_semaphore.release()

    # terminating data fetching processes
    # for training_task in training_tasks:
    training_tasks.terminate()


if __name__ == "__main__":
    args = parse_args()

    cfg_file = Config.fromfile(args.config)
    cfg_file.train_cfg.work_dir = args.work_dir

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg_file.train_cfg.work_dir, f'{timestamp}.log')
    logger = get_logger(name='CAP', log_file=log_file)

    json_file = os.path.join(cfg_file.train_cfg.work_dir, f'{timestamp}.json')

    # pdb.set_trace()
    logger.info("system config...")
    logger.info(f'Config:\n{cfg_file.pretty_text}')  # 打印所有的系统配置参数
    # pdb.set_trace()
    train(logger, json_file, cfg_file, args.start_epoch)
