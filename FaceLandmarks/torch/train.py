import torch
import neuralnet
import train_utils
import torch.nn as nn
from tqdm import tqdm
import sys
import cv2

isize = (100, 100)
batch_size = 64

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = neuralnet.ResNet(neuralnet.BasicBlock, [1, 2, 2, 1]).to(device)

# Loss and optimizer
loss_fn = nn.MSELoss(reduction='none')  # 'none' means minimize each difference, so it becomes multi MSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=1, eta_min=0.0001, verbose=True)


train_data = train_utils.LandmarksDataSet("/home/alex/Fastdata/FaceLandmarks/300W/Crop", isize, do_aug=True)
test_data = train_utils.LandmarksDataSet("/home/alex/Fastdata/FaceLandmarks/300W/Crop", isize, do_aug=False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=2)

min_loss = 1E6


def train(epoch):
    model.train()
    running_loss = 0
    print(f"\nEpoch: {epoch}")
    for i, (target, data) in enumerate(tqdm(train_loader, file=sys.stdout)):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        predicted = model(data)
        loss = loss_fn(predicted, target)
        loss.sum().backward()
        optimizer.step()
        running_loss = +loss.sum().item()
    scheduler.step()
    running_loss /= len(train_data)
    print(f"Train loss: {running_loss:.6f}")


def test():
    global min_loss
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for i, (target, data) in enumerate(tqdm(test_loader, file=sys.stdout)):
            data = data.to(device)
            target = target.to(device)
            predicted = model(data)
            loss = loss_fn(predicted, target)
            running_loss = +loss.sum().item()
    running_loss /= len(train_data)
    print(f"Test loss: {running_loss:.6f}")
    if running_loss < min_loss:
        print(f"Improvement in test loss, saving model for epoch: {epoch}")
        min_loss = running_loss
        torch.save(model, f"./build/landmarks_net.pth")
        for name in ["./test/t1.jpg", "./test/t2.jpg", "./test/t3.jpg", "./test/t4.jpg"]:
            test_img = cv2.imread(name)
            landmarks = neuralnet.predict_landmarks(test_img, isize, model, device).squeeze(0).cpu().tolist()
            train_utils.display(test_img, landmarks, 1, name, False)


for epoch in range(1000):
    train(epoch)
    test()
