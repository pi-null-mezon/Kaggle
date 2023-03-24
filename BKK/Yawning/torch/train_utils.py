import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
import cv2
import albumentations as A


class YawningDataSet(Dataset):
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
        for path in paths:
            self.labels_list = [s.name for s in os.scandir(path) if s.is_dir()]
            self.labels_list.sort()
            for i, label in enumerate(self.labels_list):
                files = [(i, os.path.join(path, label, f.name)) for f in os.scandir(os.path.join(path, label))
                         if (f.is_file() and ('.jp' in f.name or '.pn' in f.name))]
                self.samples += files
                self.targets += [i]*len(files)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.album = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(p=0.0,
                            max_holes=1,
                            min_width=int(self.tsize[0] / 2.25),
                            max_width=int(self.tsize[0] / 1.75),
                            min_height=int(self.tsize[0] / 2.25),
                            max_height=int(self.tsize[0] / 1.75)),
            A.Affine(p=0.75,
                     scale=(0.95, 1.05),
                     translate_percent=(-0.05, 0.05),
                     rotate=(-10, 10),
                     shear=(-5, 5)),
            # CutOut maybe here
            A.Blur(p=0.5, blur_limit=(3, 3)),
            A.RandomBrightnessContrast(p=0.25, brightness_limit=(-0.25, 0.25)),
            A.GaussNoise(p=0.5, var_limit=(1, 11)),
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
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        if mat.shape[0] != self.tsize[0]:
            mat = cv2.resize(mat, self.tsize,
                             interpolation=cv2.INTER_AREA if mat.shape[0] * mat.shape[1] > self.tsize[0] * self.tsize[1]
                             else cv2.INTER_CUBIC)


        if self.do_aug:
            mat = self.album(image=mat)["image"]

        cv2.imshow("probe", mat)
        cv2.waitKey(0)

        return self.transform(mat), self.samples[idx][0]
