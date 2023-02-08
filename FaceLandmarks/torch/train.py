import os

import torch
import neuralnet
import train_utils
import torch.nn as nn
from tqdm import tqdm
import sys
import cv2

isize = (100, 100)
batch_size = 64

min_test_loss_landmarks = 1E2
min_train_loss_landmarks = 1E2

min_test_loss_angles = 1E2
min_train_loss_angles = 1E2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = neuralnet.ResNet(neuralnet.BasicBlock, [2, 2, 2, 1]).to(device)

# Loss and optimizer
loss_fn_landmarks = nn.MSELoss(reduction='none')  # multi MSE
loss_fn_angles = nn.MSELoss(reduction='none')  # multi MSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=150, verbose=True)

pretrain_data = train_utils.LandmarksDataSet("/home/alex/Fastdata/HeadPose/Train/2x2", isize, do_aug=False)
train_data = train_utils.LandmarksDataSet("/home/alex/Fastdata/HeadPose/Train/2x2", isize, do_aug=True)
test_data = train_utils.LandmarksDataSet("/home/alex/Fastdata/HeadPose/Test/2x2", isize, do_aug=False)

pretrain_loader = torch.utils.data.DataLoader(pretrain_data, batch_size=batch_size, shuffle=True, num_workers=6)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=4)

#
'''
for (target, data) in pretrain_loader: 
    dummy = 0
for (target, data) in test_loader:
    dummy = 0
exit(0)
'''


def visualize(model, dirname="./test"):
    for name in [f.name for f in os.scandir(dirname) if f.is_file() and '.jp' in f.name or '.pn' in f.name]:
        test_img = cv2.imread(os.path.join(dirname, name))
        landmarks = neuralnet.predict_landmarks(test_img, isize, model, device)[1].squeeze(0).cpu().tolist()
        train_utils.display(test_img, landmarks, 30, name, False)


def train(epoch, dataloader, schedule_lr=False):
    global min_train_loss_landmarks
    model.train()
    running_loss_landmarks = 0
    running_loss_angles = 0
    print(f"Epoch: {epoch}")
    for i, (target_angles, target_landmarks, data) in enumerate(tqdm(dataloader, file=sys.stdout)):
        data = data.to(device)
        target_angles = target_angles.to(device)
        target_landmarks = target_landmarks.to(device)
        optimizer.zero_grad()
        pred_angles, pred_landmarks = model(data)
        loss_1 = loss_fn_landmarks(pred_landmarks, target_landmarks)
        loss_2 = loss_fn_angles(pred_angles, target_angles)
        loss = loss_1.sum() + loss_2.sum()
        loss.backward()
        optimizer.step()
        running_loss_landmarks += loss_1.sum().item()
        running_loss_angles += loss_2.sum().item()
    if schedule_lr:
        scheduler.step()
    running_loss_landmarks /= len(dataloader.dataset)
    running_loss_angles /= len(dataloader.dataset)
    print("Train loss:")
    print(f" - landmarks: {running_loss_landmarks:.8f}")
    print(f" - angles:    {running_loss_angles:.8f}")
    if running_loss_landmarks < min_train_loss_landmarks:
        print(f"Improvement in train loss, saving model for epoch: {epoch}")
        min_train_loss_landmarks = running_loss_landmarks
        torch.save(model, f"./build/headpose_net_train.pth")
        visualize(model)


def test(dataloader):
    global min_test_loss_landmarks
    global min_test_loss_angles
    model.eval()
    running_loss_landmarks = 0
    running_loss_angles = 0
    with torch.no_grad():
        for i, (target_angles, target_landmarks, data) in enumerate(tqdm(dataloader, file=sys.stdout)):
            data = data.to(device)
            target_angles = target_angles.to(device)
            target_landmarks = target_landmarks.to(device)
            pred_angles, pred_landmarks = model(data)
            loss_1 = loss_fn_landmarks(pred_landmarks, target_landmarks)
            loss_2 = loss_fn_angles(pred_angles, target_angles)
            running_loss_landmarks += loss_1.sum().item()
            running_loss_angles += loss_2.sum().item()
    running_loss_landmarks /= len(dataloader.dataset)
    running_loss_angles /= len(dataloader.dataset)
    print("Test loss:")
    print(f" - landmarks: {running_loss_landmarks:.8f}")
    print(f" - angles:    {running_loss_angles:.8f}")
    if running_loss_landmarks < min_test_loss_landmarks:
        print(f"Improvement in test loss, saving model for epoch: {epoch}")
        min_test_loss_landmarks = running_loss_landmarks
        torch.save(model, f"./build/headpose_net_test.pth")
        visualize(model)


print("-"*20)
print("Train several epochs without augmentations to prepare weights")
print("-"*20)
for epoch in range(5):
    train(epoch, pretrain_loader)
    print("")

print("-"*20)
print("Train with augmentations and validation")
print("-"*20)
for epoch in range(400):
    train(epoch, train_loader, schedule_lr=True)
    test(test_loader)
    print("")

cv2.waitKey()