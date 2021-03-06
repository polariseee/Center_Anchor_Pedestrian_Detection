import cv2
import numpy as np
import torch
import copy

from utils import *
import pdb


def kp_detection(cfg, ped_data, k_ind, emp_data, ngt_ind):
    batch_size = cfg.batch_size
    hyratio = cfg.hyratio
    size_train = cfg.size_train
    img_channel_mean = cfg.img_channel_mean

    batchsize_ped = int(batch_size * hyratio)
    batchsize_emp = batch_size - batchsize_ped

    img_info, x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], [], []
    # pdb.set_trace()
    for b_ind in range(0, batchsize_ped):

        img_data_aug = copy.deepcopy(ped_data[k_ind])

        assert len(img_data_aug['bboxes']) != 0

        k_ind = (k_ind + 1) % len(ped_data)
        if k_ind == len(ped_data) - 1:
            k_ind = 0
        img_meta = {}
        filepath = img_data_aug['filepath']
        image = cv2.imread(filepath)  # [h, w, c]
        img_height, img_width = image.shape[:2]
        img_meta['filename'] = filepath
        img_meta['ori_shape'] = image.shape
        img_meta['bboxes'] = []
        img_meta['ignoreareas'] = []
        # random brightness
        if np.random.randint(0, 2) == 0:
            image = brightness(image, min=0.5, max=2)
        # flipping an image randomly
        if np.random.randint(0, 2) == 0:
            image = cv2.flip(image, 1)
            if len(img_data_aug['bboxes']) > 0:
                img_data_aug['bboxes'][:, [0, 2]] = img_width - img_data_aug['bboxes'][:, [2, 0]]
            if len(img_data_aug['ignoreareas']) > 0:
                img_data_aug['ignoreareas'][:, [0, 2]] = img_width - img_data_aug['ignoreareas'][:, [2, 0]]

        # if not debug:
        #     image = image.astype(np.float32) / 255.
        #     if cfg.rand_color:
        #         color_jittering_(data_rng, image)
        #         if cfg.lighting:
        #             lighting_(data_rng, image, 0.1, np.array(cfg.eig_val, dtype=np.float32), np.array(cfg.eig_vec, dtype=np.float32))
        #
        # image[:, :, 0] *= 255.
        # image[:, :, 1] *= 255.
        # image[:, :, 2] *= 255.

        gts = np.copy(img_data_aug['bboxes'])
        igs = np.copy(img_data_aug['ignoreareas'])
        image, gts, igs = resize_image(image, gts, igs, scale=[0.4, 1.5])
        img_meta['img_shape'] = image.shape
        if image.shape[0] >= size_train[0]:
            image, gts, igs = random_crop(image, gts, igs, size_train, limit=16)
        else:
            image, gts, igs = random_pave(image, gts, igs, size_train, limit=16)
        image = image.astype(np.float32)
        image[:, :, 0] -= img_channel_mean[0]
        image[:, :, 1] -= img_channel_mean[1]
        image[:, :, 2] -= img_channel_mean[2]

        img_meta['pad_shape'] = image.shape
        image = image.transpose((2, 0, 1))
        # pdb.set_trace()
        img_data_aug['bboxes'] = gts
        img_data_aug['ignoreareas'] = igs

        img_meta['bboxes'].append(np.array(img_data_aug['bboxes'], dtype=np.float32))
        img_meta['ignoreareas'].append(np.array(img_data_aug['ignoreareas'], dtype=np.float32))

        img_data_aug['width'] = size_train[1]
        img_data_aug['height'] = size_train[0]

        y_seman, y_height, y_offset = calc_gt_center(size_train, img_data_aug)

        img_info.append(img_meta)
        x_img_batch.append(np.expand_dims(image, axis=0))
        y_seman_batch.append(np.expand_dims(y_seman, axis=0))
        y_height_batch.append(np.expand_dims(y_height, axis=0))
        y_offset_batch.append(np.expand_dims(y_offset, axis=0))

    for b_ind in range(0, batchsize_emp):

        img_data_aug = copy.deepcopy(emp_data[ngt_ind])

        assert len(img_data_aug['bboxes']) == 0

        ngt_ind = (ngt_ind + 1) % len(emp_data)
        if ngt_ind == len(emp_data) - 1:
            ngt_ind = 0
        img_meta = {}
        filepath = img_data_aug['filepath']
        image = cv2.imread(filepath)  # [h, w, c]
        img_height, img_width = image.shape[:2]
        img_meta['filename'] = filepath
        img_meta['ori_shape'] = image.shape
        img_meta['bboxes'] = []
        img_meta['ignoreareas'] = []
        # pdb.set_trace()
        # random brightness
        if np.random.randint(0, 2) == 0:
            image = brightness(image, min=0.5, max=2)
        # flipping an image randomly
        if np.random.randint(0, 2) == 0:
            image = cv2.flip(image, 1)
            if len(img_data_aug['bboxes']) > 0:
                img_data_aug['bboxes'][:, [0, 2]] = img_width - img_data_aug['bboxes'][:, [2, 0]]
            if len(img_data_aug['ignoreareas']) > 0:
                img_data_aug['ignoreareas'][:, [0, 2]] = img_width - img_data_aug['ignoreareas'][:, [2, 0]]

        # if not debug:
        #     image = image.astype(np.float32) / 255.
        #     if cfg.rand_color:
        #         color_jittering_(data_rng, image)
        #         if cfg.lighting:
        #             lighting_(data_rng, image, 0.1, np.array(cfg.eig_val, dtype=np.float32), np.array(cfg.eig_vec, dtype=np.float32))
        #
        # image[:, :, 0] *= 255.
        # image[:, :, 1] *= 255.
        # image[:, :, 2] *= 255.

        gts = np.copy(img_data_aug['bboxes'])
        igs = np.copy(img_data_aug['ignoreareas'])

        image, gts, igs = resize_image(image, gts, igs, scale=[0.4, 1.5])
        img_meta['img_shape'] = image.shape
        if image.shape[0] >= size_train[0]:
            image, gts, igs = random_crop(image, gts, igs, size_train, limit=16)
        else:
            image, gts, igs = random_pave(image, gts, igs, size_train, limit=16)
        image = image.astype(np.float32)
        image[:, :, 0] -= img_channel_mean[0]
        image[:, :, 1] -= img_channel_mean[1]
        image[:, :, 2] -= img_channel_mean[2]

        img_meta['pad_shape'] = image.shape
        image = image.transpose((2, 0, 1))
        # pdb.set_trace()
        img_data_aug['bboxes'] = gts
        img_data_aug['ignoreareas'] = igs

        assert len(img_data_aug['bboxes']) == 0

        img_meta['bboxes'].append(np.array(img_data_aug['bboxes'], dtype=np.float32))
        img_meta['ignoreareas'].append(np.array(img_data_aug['ignoreareas'], dtype=np.float32))

        img_data_aug['width'] = size_train[1]
        img_data_aug['height'] = size_train[0]

        y_seman, y_height, y_offset = calc_gt_center(size_train, img_data_aug)

        img_info.append(img_meta)
        x_img_batch.append(np.expand_dims(image, axis=0))
        y_seman_batch.append(np.expand_dims(y_seman, axis=0))
        y_height_batch.append(np.expand_dims(y_height, axis=0))
        y_offset_batch.append(np.expand_dims(y_offset, axis=0))

    x_img_batch = torch.from_numpy(np.concatenate(x_img_batch, axis=0))
    y_seman_batch = torch.from_numpy(np.concatenate(y_seman_batch, axis=0))
    y_height_batch = torch.from_numpy(np.concatenate(y_height_batch, axis=0))
    y_offset_batch = torch.from_numpy(np.concatenate(y_offset_batch, axis=0))

    return {
        "xs": [x_img_batch],
        "ys": [y_seman_batch, y_height_batch, y_offset_batch],
        "img_meta": img_info
    }, k_ind, ngt_ind


def sample_data(cfg, ped_data, k_ind, emp_data, ngt_ind):
    return globals()[cfg.sampling_function](cfg, ped_data, k_ind, emp_data, ngt_ind)
