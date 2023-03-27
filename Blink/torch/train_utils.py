import torch
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import albumentations as A


def numpy_image2torch_tensor(bgr, mean, std, swap_red_blue=False):
    tmp = bgr.astype(dtype=np.float32) / 255.0
    if swap_red_blue:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    tmp = np.transpose(tmp, axes=(2, 0, 1))  # HxWxC -> CxHxW
    tmp -= np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    tmp /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return torch.from_numpy(tmp)


class BlinkDataSet(Dataset):
    def __init__(self, paths, tsize, do_aug):
        """
            :param path: path to dataset
            :param tsize: target size of images
            :param do_aug: enable augmentations
        """
        self.tsize = tsize
        self.do_aug = do_aug
        self.samples = []
        self.targets = []
        self.mean = 3*[0.455]
        self.std = 3*[0.225]
        for path in paths:
            self.labels_list = [s.name for s in os.scandir(path) if s.is_dir()]
            self.labels_list.sort()
            for i, label in enumerate(self.labels_list):
                files = [(i, os.path.join(path, label, f.name)) for f in os.scandir(os.path.join(path, label))
                         if (f.is_file() and ('.jp' in f.name or '.pn' in f.name))]
                self.samples += files
                self.targets += [i]*len(files)
        self.album = A.Compose([
            A.HorizontalFlip(p=0.5),
            #A.CoarseDropout(p=0.5,
            #                max_holes=1,
            #                min_width=int(self.tsize[0] / 2.25),
            #                max_wid th=int(self.tsize[0] / 1.75),
            #                min_height=int(self.tsize[0] / 2.25),
            #                max_height=int(self.tsize[0] / 1.75)),
            A.Affine(p=1.0,
                     scale=(0.9, 1.1),
                     translate_percent=(-0.1, 0.1),
                     rotate=(-10, 10),
                     shear=(-1, 1)),
            A.RandomBrightnessContrast(p=0.25, brightness_limit=(-0.25, 0.25)),
            A.GaussNoise(p=0.5, var_limit=(1, 13)),
            A.RandomBrightnessContrast(p=0.25, brightness_limit=(-0.25, 0.25)),
            A.ToGray(p=0.5),
            A.ColorJitter(p=1.0)
        ], p=1.0)

    def labels_names(self):
        d = {}
        for i, label in enumerate(self.labels_list):
            d[i] = label
        return d

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx][1]
        mat = cv2.imread(filename, cv2.IMREAD_COLOR)

        if mat.shape[0] != self.tsize[0]:
            mat = cv2.resize(mat, self.tsize,
                             interpolation=cv2.INTER_LINEAR if mat.shape[0] * mat.shape[1] > self.tsize[0] * self.tsize[1]
                             else cv2.INTER_CUBIC)

        if self.do_aug:
            mat = self.album(image=mat)["image"]

        #cv2.putText(mat, f"{self.samples[idx][0]}", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #cv2.imshow("probe", mat)
        #cv2.waitKey(0)

        return numpy_image2torch_tensor(mat, mean=self.mean, std=self.std), self.samples[idx][0]
