import os
import kagglehub
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

#data set download
dirc_path = os.getcwd()
os.environ['KAGGLEHUB_CACHE'] = os.path.join(dirc_path, "data")

if os.path.exists(os.path.join(dirc_path, r"data\datasets\meowmeowmeowmeowmeow\gtsrb-german-traffic-sign\versions\1")):
    data_set_path = os.path.join(dirc_path, r"data\datasets\meowmeowmeowmeowmeow\gtsrb-german-traffic-sign\versions\1")
else:
    data_set_path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

print("Path to dataset files:", data_set_path)

#device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data preprocessing
data = []
labels = []
classes = 43

train_path = os.path.join(data_set_path, 'Train')

for i in range(classes):
    path = train_path
    path = os.path.join(path, str(i))
    images = os.listdir(path)

    for image in images:
        image = cv2.imread(path + "/" + image)
        image = cv2.resize(image, (30, 30))
        data.append(image)
        labels.append(i)


data = torch.from_numpy(np.array(data))
labels = torch.from_numpy(np.array(labels))

data = data.permute(0, 3, 1, 2)
data = data.to(torch.float32)
labels = labels.to(torch.float32)

dataset = TensorDataset(data, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 64)

test = pd.read_csv(os.path.join(data_set_path, "Test.csv"))

test_labels = test["ClassId"].values
test_img_paths = test["Path"].values

test_imgs = []

for img_path in test_img_paths:
    img_path = os.path.join(data_set_path, img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (30, 30))
    test_imgs.append(img)

test_imgs = torch.from_numpy(np.array(test_imgs))
test_imgs = test_imgs.permute(0,3,1,2)

test_labels = torch.from_numpy(np.array(test_labels))

test_imgs = test_imgs.to(torch.float32)
test_labels = test_labels.to(torch.float32)

test_data = TensorDataset(test_imgs, test_labels)
test_loader = DataLoader(test_data, batch_size = 64)

#CNN Architecture definition
model1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2),
            torch.nn.BatchNorm2d(32),
    
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2),
            torch.nn.BatchNorm2d(128),
    
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 4 * 4, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 43)
        )

model1 = model1.to(device)

#optimizer and loss definition
criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr = 0.001)

#training
train_losses = []
val_accuracies = []

epochs = 35
for epoch in range(epochs):
    model1.train()
    total_loss = 0

    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device).long()

        optimizer1.zero_grad()
        outputs = model1(inputs)
        loss = criterion1(outputs, labels)
        loss.backward()
        optimizer1.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

#evaluation
model1.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model1(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy of model 1: {100 * correct / total:.2f}%")


model1.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device).long()

        outputs = model1(inputs)
        _, prediction = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

print(f"Testing Accuracy of model 1: {100 * (correct / total): .2f}%")

#saving the data set
torch.save(model1.state_dict(), "model_without_data_augmentation.pth")
