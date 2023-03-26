import os
import torch
import neuralnet
import train_utils
import torch.nn as nn
from tqdm import tqdm
import sys
import cv2
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

isize = (100, 100)
batch_size = 128

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

filters = 16
layers = [1, 1, 1, 1]
model = neuralnet.ResNet(neuralnet.BasicBlock, filters, layers).to(device)
model_name = f"resnet{sum(layers) * 2 + 2}_{filters}f_{isize[0]}@200bbox"

# Loss and optimizer
loss_fn_landmarks = nn.MSELoss(reduction='none')  # multi MSE
loss_fn_angles = nn.MSELoss(reduction='none')  # multi MSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=100, verbose=True)

warmup_data = train_utils.LandmarksDataSet("/home/alex/Fastdata/FaceLandmarks/HeadPose/Train/2x2", isize, do_aug=False)
train_data = train_utils.LandmarksDataSet("/home/alex/Fastdata/FaceLandmarks/HeadPose/Train/2x2", isize, do_aug=True)
test_data = train_utils.LandmarksDataSet("/home/alex/Fastdata/FaceLandmarks/HeadPose/Test/2x2", isize, do_aug=False)

pretrain_loader = torch.utils.data.DataLoader(warmup_data, batch_size=batch_size, shuffle=True, num_workers=6)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=6)

# DB Cleaning
'''
for (target, data) in pretrain_loader: 
    dummy = 0
for (target, data) in test_loader:
    dummy = 0
exit(0)
'''


def visualize(dirname="./test"):
    for name in [f.name for f in os.scandir(dirname) if f.is_file() and '.jp' in f.name or '.pn' in f.name]:
        test_img = cv2.imread(os.path.join(dirname, name))
        landmarks = neuralnet.predict_landmarks(test_img, isize, model, device)[1].squeeze(0).cpu().tolist()
        train_utils.display(test_img, landmarks, 30, name, False)


# Metrics
metrics = {
    'train': {'loss': float('inf')},
    'test':  {'loss': float('inf')}
}


def update_metrics(mode, epoch, running_loss_angles, running_loss_landmarks):
    print(f"{mode.upper()}:")
    if not os.path.exists('./weights'):
        os.makedirs('./weights')
    writer.add_scalar(f"Loss/angles/{mode}", running_loss_angles, epoch)
    writer.add_scalar(f"Loss/landmarks/{mode}", running_loss_landmarks, epoch)
    print(f" - landmarks loss:  {running_loss_landmarks:.5f}")
    print(f" - angles loss:  {running_loss_angles:.5f}")
    running_loss = (running_loss_angles + running_loss_landmarks) / 2
    if running_loss < metrics[mode]['loss']:
        metrics[mode]['loss'] = running_loss
        torch.save(model, f"./weights/{model_name}_{mode}.pth")
        print(f" - loss:  {running_loss:.5f} - improvement")
        visualize()
    else:
        print(f" - loss:  {running_loss:.5f}")


def train(epoch, dataloader, schedule_lr=False):
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
    update_metrics('train', epoch, running_loss_angles, running_loss_landmarks)


def test(epoch, dataloader):
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
    update_metrics('test', epoch, running_loss_angles, running_loss_landmarks)


print("-" * 20)
print("Warmup training")
print("-" * 20)
for epoch in range(5):
    train(epoch, pretrain_loader)
    print("")
print("-" * 20)
print("Training with augmentations")
print("-" * 20)
for epoch in range(300):
    train(epoch, train_loader, schedule_lr=True)
    test(epoch, test_loader)
    print("")

cv2.waitKey()
