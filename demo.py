import cv2
import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from sys import exit
import sys


road_sign_dict = {
    0: "Speed Limit (20km/h)",
    1: "Speed Limit (30km/h)",
    2: "Speed Limit (50km/h)",
    3: "Speed Limit (60km/h)",
    4: "Speed Limit (70km/h)",
    5: "Speed Limit (80km/h)",
    6: "End of Speed Limit (80km/h)",
    7: "Speed Limit (100km/h)",
    8: "Speed Limit (120km/h)",
    9: "No Passing",
    10: "No Passing for Vehicles Over 3.5 Metric Tons",
    11: "Right-of-Way at Intersection",
    12: "Priority Road",
    13: "Yield",
    14: "Stop",
    15: "No Vehicles",
    16: "Vehicles Over 3.5 Metric Tons Prohibited",
    17: "No Entry",
    18: "General Caution",
    19: "Dangerous Curve to the Left",
    20: "Dangerous Curve to the Right",
    21: "Double Curve",
    22: "Bumpy Road",
    23: "Slippery Road",
    24: "Road Narrows on the Right",
    25: "Construction Zone",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Children Crossing",
    29: "Bicycles Crossing",
    30: "Beware of Ice/Snow",
    31: "Wild Animals Crossing",
    32: "End of All Speed and Passing Limits",
    33: "Turn Right Ahead",
    34: "Turn Left Ahead",
    35: "Ahead Only",
    36: "Go Straight or Turn Right",
    37: "Go Straight or Turn Left",
    38: "Right Turn Yielding",
    39: "Left Turn Yielding",
    40: "Round About",
    41: "No Overtaking",
    42: "No Overtaking for Vehicles Over 3.5 Metric Tons"
}

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

if os.path.exists(r"data\datasets\meowmeowmeowmeowmeow\gtsrb-german-traffic-sign\versions\1\Test"):
    test_images_path = r"data\datasets\meowmeowmeowmeowmeow\gtsrb-german-traffic-sign\versions\1\Test"
else:
    print("Download the dataset first!")
    exit()

images = os.listdir(test_images_path)

model = torch.nn.Sequential(
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

model = model.to(device)

def eval(mode=None, model=model):
    if mode == 'augm':
        model.load_state_dict(torch.load(r"Models\model_with_data_augmentation.pth", weights_only = True))
    elif mode == 'naugm':
        model.load_state_dict(torch.load(r"Models\model_without_data_augmentation.pth", weights_only = True))
    else:
        print("No valid mode provided, falling back to model trained on augmented data!")
        model.load_state_dict(torch.load(r"Models\model_with_data_augmentation.pth", weights_only = True))

    c_c = 0

    for image in images:
        
        try:
            test_i = cv2.imread(os.path.join(test_images_path, image))
        except:
            continue
            
        
        test_i_ = cv2.cvtColor(test_i, cv2.COLOR_BGR2RGB)  
        plt.imshow(test_i_)
        plt.axis('off') 
        plt.show()
        test_i = cv2.resize(test_i, (30,30))
        test_i = torch.from_numpy(test_i)
        test_i = test_i.to(torch.float32)
        test_i = test_i.permute(2, 0, 1)
        test_i = test_i.unsqueeze(0)
        
        
        model.eval()
        model = model.to(device)
        test_i = test_i.to(device)
        
        output = model(test_i)
        _, prediction = torch.max(output, dim=1)
        
        print(road_sign_dict[prediction.item()], prediction.item())


if __name__ == '__main__':

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = None

    eval(mode)
    
