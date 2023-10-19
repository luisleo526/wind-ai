import csv
import math

import numpy as np
import torch
from torch.utils.data import Dataset


def read_and_preprocessing(cfg):
    data = {}
    with open(cfg.data.experiment_info, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row_id, row in enumerate(reader):
            if row_id > 0:
                exp_id, exp_angle, exp_vel = row
                data[exp_id] = dict(id=exp_id, angle=float(exp_angle) * math.pi / 180.0,
                                    vel=float(exp_vel), data=[])

    with open(cfg.data.experiment_data, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row_id, row in enumerate(reader):
            if row_id > 0:
                exp_id, dev_id, x, y, power = row
                data[exp_id]['data'].append(list(map(float, [x, y, power])))

    data = list(data.values())
    for exp in data:
        exp['data'] = np.transpose(np.array(exp['data']), (1, 0))

    return data


class WindDataset(Dataset):

    def __init__(self, cfg, data, p_max, p_min):
        self.data = data

        x, y = np.meshgrid(np.linspace(-cfg.data.domain_size, cfg.data.domain_size, cfg.data.image_size),
                           np.linspace(-cfg.data.domain_size, cfg.data.domain_size, cfg.data.image_size))
        self.x = x
        self.y = y
        self.h = 2 * cfg.data.domain_size / (cfg.data.image_size - 1)

        self.r_min = cfg.data.r_min
        self.r_max = cfg.data.r_max

        self.p_max = p_max
        self.p_min = p_min

        self.image = np.zeros((cfg.data.image_size, cfg.data.image_size, 1))

        self.shift = cfg.data.domain_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.zeros_like(self.image)
        gt = np.zeros_like(self.image)
        mask = np.zeros_like(self.image)

        for device_id in range(self.data[idx]['data'].shape[1]):
            x, y, p = self.data[idx]['data'][:, device_id]
            ix, iy = int((x + self.shift) / self.h), int((y + self.shift) / self.h)
            sigma = self.r_min + (self.r_max - self.r_min) * np.random.rand()

            int_sigma = int(sigma)
            mask[iy - int_sigma:iy + int_sigma, ix - int_sigma:ix + int_sigma, 0] = 1.0
            gt[iy - int_sigma:iy + int_sigma, ix - int_sigma:ix + int_sigma, 0] = (p - self.p_min) / (
                        self.p_max - self.p_min)

            sigma = sigma * self.h
            r = (self.x - x) ** 2 + (self.y - y) ** 2

            image[:, :, 0] += np.exp(-r / (2 * sigma ** 2))

        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
        gt = torch.tensor(np.transpose(gt, (2, 0, 1)), dtype=torch.float32)
        mask = torch.tensor(np.transpose(mask, (2, 0, 1)), dtype=torch.float32)
        scalars = torch.tensor([self.data[idx]['angle'], self.data[idx]['vel']], dtype=torch.float32)

        return image, scalars, gt, mask
