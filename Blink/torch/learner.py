import torch
from torcheval.metrics.functional import multiclass_f1_score
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from tqdm import tqdm
import sys

from collections import Counter
from train_utils import BlinkDataSet
import neuralnet

isize = (40, 40)
batch_size = 128

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = BlinkDataSet(["/home/alex/Testdata/Blink&Yawn/Eyes/Train"], isize, do_aug=True)
test_dataset = BlinkDataSet(["/home/alex/Testdata/Blink&Yawn/Eyes/Test"], isize, do_aug=False)

print(train_dataset.labels_names())
num_classes = len(train_dataset.labels_names())

lbls_count = dict(Counter(train_dataset.targets))
print(lbls_count)
class_weights = list(1 / torch.Tensor(list(lbls_count.values())))  # or maually set [1.0, 1.0, 0.01, 0.01]
samples_weights = [class_weights[lbl] for lbl in train_dataset.targets]
sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights, num_samples=len(train_dataset), replacement=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, num_workers=6)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)


# Initialize network
model = neuralnet.ResNet(neuralnet.BasicBlock, [0, 0, 0, 0], num_classes=num_classes).to(device)
model_name = "blink_resnet4s"

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, verbose=True)
num_epochs = 300

min_test_loss = float('inf')
min_train_loss = float('inf')
max_f1_test = 0
max_f1_train = 0


def train(epoch, dataloader):
    global min_train_loss
    global max_f1_train
    model.train()
    running_loss = 0
    f1 = 0
    correct = 0
    total = 0
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
        f1 += multiclass_f1_score(predicted, labels, num_classes=num_classes, average='macro')
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    scheduler.step()
    running_loss /= len(dataloader)
    f1 /= len(dataloader)
    print("Train:")
    if running_loss < min_train_loss:
        min_train_loss = running_loss
        torch.save(model, f"./build/{model_name}_train.pth")
        print(f" - loss: {running_loss:.8f} - hooray :) improvement")
    else:
        print(f" - loss: {running_loss:.8f}")
    if f1 > max_f1_train:
        max_f1_train = f1
        torch.save(model, f"./build/{model_name}_train_f1.pth")
        print(f" - F1:   {f1:.5f} - hooray :) improvement")
    else:
        print(f" - F1:   {f1:.5f}")
    print(f" - ACC: {(100 * correct / total):.2f}")


def test(dataloader):
    global min_test_loss
    global max_f1_test
    model.eval()
    running_loss = 0
    f1 = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, file=sys.stdout)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            f1 += multiclass_f1_score(predicted, labels, num_classes=num_classes, average='macro')
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    running_loss /= len(dataloader)
    f1 /= len(dataloader)
    print("Test:")
    if running_loss < min_test_loss:
        min_test_loss = running_loss
        torch.save(model, f"./build/{model_name}_test.pth")
        print(f" - loss: {running_loss:.8f} - hooray :) improvement")
    else:
        print(f" - loss: {running_loss:.8f}")
    if f1 > max_f1_test:
        max_f1_test = f1
        torch.save(model, f"./build/{model_name}_test_f1.pth")
        print(f" - F1:   {f1:.5f} - hooray :) improvement")
    else:
        print(f" - F1:   {f1:.5f}")
    print(f" - ACC: {(100 * correct / total):.2f}")


for epoch in range(num_epochs):
    train(epoch, train_dataloader)
    test(test_dataloader)

print("\nBest metrics:")
print(f" - Train F1: {max_f1_train:.5f}")
print(f" - Test F1:  {max_f1_test:.5f}")
