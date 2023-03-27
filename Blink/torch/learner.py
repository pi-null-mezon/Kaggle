import torch
from torch import optim
from torch import nn
from tqdm import tqdm
import sys
import os
from collections import Counter
from train_utils import BlinkDataSet
import neuralnet
from torch.utils.tensorboard import SummaryWriter

isize = (36, 36)
batch_size = 128
num_epochs = 100

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = BlinkDataSet(["/home/alex/Fastdata/Drivercontrol/Blink&Yawn/Eyes/Train"], isize, do_aug=True)
test_dataset = BlinkDataSet(["/home/alex/Fastdata/Drivercontrol/Blink&Yawn/Eyes/Test"], isize, do_aug=False)

print("Train dataset:")
print(f"  {train_dataset.labels_names()}")
num_classes = len(train_dataset.labels_names())
lbls_count = dict(Counter(train_dataset.targets))
print(f"  {lbls_count}")
class_weights = list(1 / torch.Tensor(list(lbls_count.values())))  # or maually set [1.0, 1.0, 0.01, 0.01]
samples_weights = [class_weights[lbl] for lbl in train_dataset.targets]
sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights, num_samples=len(train_dataset), replacement=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, num_workers=6)
print("Test dataset:")
print(f"  {test_dataset.labels_names()}")
print(f"  {dict(Counter(test_dataset.targets))}")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

open_lbl = None
for key in train_dataset.labels_names():
    if train_dataset.labels_names()[key] == 'Open':
        open_lbl = key
        break

# Initialize network
filters = 8
layers = [1, 1, 1]
model = neuralnet.ResNet(neuralnet.BasicBlock, filters, layers, num_classes=num_classes).to(device)
model_name = f"resnet{sum(layers)*2 + 2}_{filters}f_{isize[0]}x{isize[1]}"

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//2, gamma=0.1, verbose=True)

# Metrics
metrics = {
    'train': {'OECE': float('inf'), 'CECE': float('inf'), 'loss': float('inf'), 'AE': float('inf')},
    'test':  {'OECE': float('inf'), 'CECE': float('inf'), 'loss': float('inf'), 'AE': float('inf')}
}

writer = SummaryWriter()

def update_metrics(mode, epoch,
                   running_loss,
                   true_positive_open,
                   false_positive_open,
                   true_negative_open,
                   false_negative_open):
    print(f"{mode.upper()}:")
    if not os.path.exists('./weights'):
        os.makedirs('./weights')
    writer.add_scalar(f"Loss/{mode}", running_loss, epoch)
    if running_loss < metrics[mode]['loss']:
        metrics[mode]['loss'] = running_loss
        print(f" - loss:  {running_loss:.5f} - improvement")
    else:
        print(f" - loss:  {running_loss:.5f}")
    prob = false_positive_open / (false_positive_open + true_negative_open + 1E-6)
    writer.add_scalar(f"Closed face classification error/{mode}", prob, epoch)
    if prob < metrics[mode]['CECE']:
        metrics[mode]['CECE'] = prob
        # torch.save(model, f"./weights/{model_name}_{mode}_BPCE.pth")
        print(f" - Closed eye classification error: {prob:.5f} - improvement")
    else:
        print(f" - Closed eye classification error: {prob:.5f}")
    cece = prob
    prob = false_negative_open / (false_negative_open + true_positive_open + 1E-6)
    writer.add_scalar(f"Open face classification error/{mode}", prob, epoch)
    if prob < metrics[mode]['OECE']:
        metrics[mode]['OECE'] = prob
        # torch.save(model, f"./weights/{model_name}_{mode}_SPCE.pth")
        print(f" - Open eye classification error: {prob:.5f} - improvement")
    else:
        print(f" - Open eye classification error: {prob:.5f}")
    oece = prob
    prob = (cece + oece) / 2
    writer.add_scalar(f"Average error/{mode}", prob, epoch)
    if prob < metrics[mode]['AE']:
        metrics[mode]['AE'] = prob
        torch.save(model, f"./weights/{model_name}_{mode}_AE.pth")
        print(f" - Average error: {prob:.5f} - improvement")
    else:
        print(f" - Average error: {prob:.5f}")


def train(epoch, dataloader):
    model.train()
    running_loss = 0
    true_positive_open = 0
    false_positive_open = 0
    true_negative_open = 0
    false_negative_open = 0
    print('\nEpoch : %d' % epoch)
    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, file=sys.stdout)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        true_positive_open += (predicted[labels == open_lbl] == open_lbl).sum().item()
        false_positive_open += (predicted[labels != open_lbl] == open_lbl).sum().item()
        true_negative_open += (predicted[labels != open_lbl] != open_lbl).sum().item()
        false_negative_open += (predicted[labels == open_lbl] != open_lbl).sum().item()
    scheduler.step()
    update_metrics('train', epoch,
                   running_loss / len(dataloader),
                   true_positive_open,
                   false_positive_open,
                   true_negative_open,
                   false_negative_open)


def test(epoch, dataloader):
    model.eval()
    running_loss = 0
    true_positive_open = 0
    false_positive_open = 0
    true_negative_open = 0
    false_negative_open = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, file=sys.stdout)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            true_positive_open += (predicted[labels == open_lbl] == open_lbl).sum().item()
            false_positive_open += (predicted[labels != open_lbl] == open_lbl).sum().item()
            true_negative_open += (predicted[labels != open_lbl] != open_lbl).sum().item()
            false_negative_open += (predicted[labels == open_lbl] != open_lbl).sum().item()
    update_metrics('train', epoch,
                   running_loss / len(dataloader),
                   true_positive_open,
                   false_positive_open,
                   true_negative_open,
                   false_negative_open)


for epoch in range(num_epochs):
    train(epoch, train_dataloader)
    test(epoch, test_dataloader)

