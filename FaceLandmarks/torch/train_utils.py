import torch
from torch.utils.data import Dataset
import json
import os
import cv2
import albumentations as A
from neuralnet import numpy_image2torch_tensor


class LandmarksDataSet(Dataset):
    def __init__(self, path, tsize, do_aug):
        """
            :param path: path to dataset
            :param tsize: target size of images
            :param do_aug: enable augmentations
        """
        self.tsize = tsize
        self.do_aug = do_aug
        self.path = path
        self.mean = [0.455] * 3
        self.std = [0.225] * 3
        self.album = A.Compose([
            A.CoarseDropout(p=0.025,
                            max_holes=1,
                            min_width=int(self.tsize[1] / 3.25), max_width=int(self.tsize[1] / 1.75),
                            min_height=int(self.tsize[0] / 3.25), max_height=int(self.tsize[0] / 1.75)),
            A.RandomBrightnessContrast(p=0.25, brightness_limit=(-0.25, 0.25)),
            A.Blur(p=0.1, blur_limit=3),
            A.GaussNoise(p=0.1),
            A.ImageCompression(p=0.1, quality_lower=75, quality_upper=100),
            A.ToGray(p=0.25),
            A.Posterize(p=0.05),
            A.RandomBrightnessContrast(p=0.25, brightness_limit=(-0.25, 0.25)),
            A.ColorJitter(p=1.0),
        ], p=1.0)
        self.samples = [os.path.join(path, f.name) for f in os.scandir(path)
                        if (f.is_file() and ('.jp' in f.name or '.pn' in f.name))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        mat = cv2.imread(filename, cv2.IMREAD_COLOR)

        landmakrs = []
        json_filename = filename.rsplit('.', 1)[0] + '.json'
        with open(json_filename, 'r') as file:
            face = json.load(file)
            pitch = face['pitch']
            yaw = face['yaw']
            roll = face['roll']
            for item in face['landmarks']:
                landmakrs += [item['x'] / mat.shape[0] - 0.5, item['y'] / mat.shape[1] - 0.5]  # relative to width, rows

        if mat.shape[0] != self.tsize[0]:
            mat = cv2.resize(mat, self.tsize,
                             interpolation=cv2.INTER_LINEAR if mat.shape[0] * mat.shape[1] > self.tsize[0] * self.tsize[1]
                             else cv2.INTER_CUBIC)

        if torch.randn(1).item() > 0.5:
            mat, landmakrs, yaw, roll = lrflip(mat, landmakrs, yaw, roll)
        mat, landmakrs, roll = jitter(mat, landmakrs, roll)

        if self.do_aug:
            mat = self.album(image=mat)["image"]

        # Visual Control
        #display(mat, landmakrs, 0, "probe", False)

        # filter out confusing points
        '''cp = average(landmakrs)
        if abs(cp[0]) > 0.25 or abs(cp[1]) > 0.25:
            display(mat, landmakrs, 30, "probe", False)
            os.remove(filename)
            os.remove(json_filename)
            print(f"{filename}")
        '''
        t = numpy_image2torch_tensor(mat, self.mean, self.std, swap_red_blue=False)
        return torch.Tensor([1.2 * pitch / 90, 1.28 * yaw / 90, roll / 90]), torch.Tensor(landmakrs), t


def average(landmarks):
    x = 0
    y = 0
    num = len(landmarks) // 2
    for i in range(num):
        x += landmarks[2*i]
        y += landmarks[2*i + 1]
    x /= num
    y /= num
    return x, y


def display(img, landmarks, delay_ms, window_name="probe", show_pts=True, ):
    for i in range(len(landmarks) // 2):
        pt = (int((landmarks[2 * i] + 0.5) * img.shape[0]), int((landmarks[2 * i + 1] + 0.5) * img.shape[1]))
        cv2.circle(img, pt, round(img.shape[0]/128) if img.shape[0] > 100 else 1, (0, 255, 0), -1, cv2.LINE_AA)
        if show_pts:
            cv2.putText(img, f"{i}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow(window_name, img)
    cv2.waitKey(delay_ms)


# landmarks: [x0, y0, ..., x135, y135]
def lrflip(img, landmarks, yaw, roll):
    lpts = [1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 37, 38, 39, 40, 41, 42, 32, 33, 49, 50, 51, 61, 62, 68, 59, 60];
    rpts = [17, 16, 15, 14, 13, 12, 11, 10, 27, 26, 25, 24, 23, 46, 45, 44, 43, 48, 47, 36, 35, 55, 54, 53, 65, 64, 66,
            57, 56];
    for l, r in zip(lpts, rpts):
        landmarks[2 * (l - 1)], landmarks[2 * (r - 1)] = landmarks[2 * (r - 1)], landmarks[2 * (l - 1)]
        landmarks[2 * (l - 1) + 1], landmarks[2 * (r - 1) + 1] = landmarks[2 * (r - 1) + 1], landmarks[2 * (l - 1) + 1]
    for i in range(len(landmarks) // 2):
        landmarks[2 * i] *= -1;
    return cv2.flip(img, 1), landmarks, -yaw, -roll 


def jitter(img, landmarks, roll, tsize=(0, 0), maxscale=0.1, maxshift=0.05, maxangle=25, bordertype=cv2.BORDER_CONSTANT):
    isize = (img.shape[0], img.shape[1])
    scale = min(tsize[0] / isize[0], tsize[1] / isize[1]) if tsize[0] * tsize[1] > 0 else 1
    angle = maxangle * (2 * torch.rand(1).item() - 1)
    rm = cv2.getRotationMatrix2D((isize[0] / 2, isize[1] / 2), angle,
                                 scale * (1 + maxscale * (2 * torch.rand(1).item() - 1)))
    if tsize[0] > 0 and tsize[1] > 0:
        rm[0, 2] += -(isize[0] - tsize[0]) / 2
        rm[1, 2] += -(isize[1] - tsize[1]) / 2
    rm[0, 2] += (isize[0] * maxshift * scale * (2 * torch.rand(1).item() - 1))
    rm[1, 2] += (isize[1] * maxshift * scale * (2 * torch.rand(1).item() - 1))
    outmat = cv2.warpAffine(img, rm, tsize,
                            cv2.INTER_AREA if isize[0] * isize[1] > tsize[0] * tsize[1] else cv2.INTER_CUBIC,
                            bordertype)
    tmplbls = [0] * len(landmarks)
    for i in range(len(landmarks) // 2):
        tmplbls[2 * i] = (isize[0] * (landmarks[2 * i] + 0.5) * rm[0, 0] + isize[1] * (landmarks[2 * i + 1] + 0.5) * rm[
            0, 1] + rm[0, 2]) / isize[0] - 0.5
        tmplbls[2 * i + 1] = (isize[0] * (landmarks[2 * i] + 0.5) * rm[1, 0] + isize[1] * (landmarks[2 * i + 1] + 0.5) *
                              rm[1, 1] + rm[1, 2]) / isize[1] - 0.5
    return outmat, tmplbls, (roll - angle)  # as opencv and deep face annotations use different signs for roll
