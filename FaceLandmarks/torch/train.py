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
min_test_loss = 1
min_train_loss = 1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = neuralnet.ResNet(neuralnet.BasicBlock, [2, 2, 2, 1]).to(device)

# Loss and optimizer
loss_fn = nn.MSELoss(reduction='none')  # 'none' means minimize each difference, so it becomes multi MSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=0.00001, verbose=True)

pretrain_data = train_utils.LandmarksDataSet("/home/alex/Fastdata/HeadPose/Train/2x2", isize, do_aug=False)
train_data = train_utils.LandmarksDataSet("/home/alex/Fastdata/HeadPose/Train/2x2", isize, do_aug=True)
test_data = train_utils.LandmarksDataSet("/home/alex/Fastdata/HeadPose/Test/2x2", isize, do_aug=True)

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
        landmarks = neuralnet.predict_landmarks(test_img, isize, model, device).squeeze(0).cpu().tolist()
        train_utils.display(test_img, landmarks, 30, name, False)


def train(epoch, dataloader, schedule_lr=False):
    global min_train_loss
    model.train()
    running_loss = 0
    print(f"Epoch: {epoch}")
    for i, (target, data) in enumerate(tqdm(dataloader, file=sys.stdout)):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        predicted = model(data)
        loss = loss_fn(predicted, target)
        loss.sum().backward()
        optimizer.step()
        running_loss += loss.sum().item()
    if schedule_lr:
        scheduler.step()
    running_loss /= len(dataloader)
    print(f"Train loss: {running_loss:.8f}")
    if running_loss < min_train_loss:
        print(f"Improvement in train loss, saving model for epoch: {epoch}")
        min_train_loss = running_loss
        torch.save(model, f"./build/landmarks_net_best_train.pth")
        visualize(model)


def test(dataloader):
    global min_test_loss
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for i, (target, data) in enumerate(tqdm(dataloader, file=sys.stdout)):
            data = data.to(device)
            target = target.to(device)
            predicted = model(data)
            loss = loss_fn(predicted, target)
            running_loss += loss.sum().item()
    running_loss /= len(dataloader)
    print(f"Test loss: {running_loss:.8f}")
    if running_loss < min_test_loss:
        print(f"Improvement in test loss, saving model for epoch: {epoch}")
        min_test_loss = running_loss
        torch.save(model, f"./build/landmarks_net.pth")
        visualize(model)


print("-"*20)
print("Train several epochs without augmentations to prepare weights")
print("-"*20)
for epoch in range(1):
    train(epoch, pretrain_loader)
    print("")

print("-"*20)
print("Train with augmentations and validation")
print("-"*20)
for epoch in range(500):
    train(epoch, train_loader, schedule_lr=True)
    test(test_loader)
    print("")
    
cv2.waitKey()