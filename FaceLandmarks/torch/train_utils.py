import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
import cv2
import albumentations as A


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
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.album = A.Compose([
            A.CoarseDropout(p=0.5,
                            min_width=self.tsize[0]//3,
                            min_height=self.tsize[0]//3,
                            max_holes=1,
                            max_height=self.tsize[0]//2,
                            max_width=self.tsize[0]//2),
            A.RandomBrightnessContrast(p=1.0, brightness_limit=(-0.6, 0.6)),
            A.RandomGamma(p=0.5),
            A.CLAHE(p=0.5),
            A.Blur(p=0.5,),
            A.GaussNoise(p=0.5),
            A.ToGray(p=0.1),
            A.Posterize(p=0.1)
        ], p=1.0)
        self.samples = [os.path.join(path, f.name) for f in os.scandir(path)
                        if (f.is_file() and ('.jp' in f.name or '.pn' in f.name))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        mat = cv2.imread(filename, cv2.IMREAD_COLOR)
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

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
                            interpolation=cv2.INTER_AREA if mat.shape[0] * mat.shape[1] > self.tsize[0] * self.tsize[1] else cv2.INTER_CUBIC)

        if torch.randn(1).item() > 0.5:
            mat, landmakrs, yaw, roll = lrflip(mat, landmakrs, yaw, roll)
        mat, landmakrs, roll = jitter(mat, landmakrs, roll)

        if self.do_aug:
            mat = self.album(image=mat)["image"]

        #display(mat, landmakrs, 0, "probe", False)

        # filter out confusing points
        '''cp = average(landmakrs)
        if abs(cp[0]) > 0.25 or abs(cp[1]) > 0.25:
            display(mat, landmakrs, 30, "probe", False)
            os.remove(filename)
            os.remove(json_filename)
            print(f"{filename}")
        '''

        return torch.Tensor([1.2 * pitch / 90, 1.28 * yaw / 90, roll / 90]), torch.Tensor(landmakrs), self.transform(mat)


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


def jitter(img, landmarks, roll, tsize=(0, 0), maxscale=0.1, maxshift=0.05, maxangle=10, bordertype=cv2.BORDER_CONSTANT):
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
