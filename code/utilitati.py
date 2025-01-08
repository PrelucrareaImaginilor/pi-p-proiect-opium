from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm

def dice_metric(predicted, target):
    # cu cat rezultatul e mai aproape de 1 cu atat este mai bun (predictie buna de valori) || 0 = opusul
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True) 
    value = 1 - dice_value(predicted, target).item()
    return value

def calculate_weights(val1, val2):
    '''
    explicatie mai tarziu ---> MATE
    val1 -> pixelii care fac parte din fundal
    val2 -> pixelii pentru ficat de ex (cei care ne intereseaza)
    '''
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count/summ
    weights = 1/weights
    summ = weights.sum()
    weights = weights/summ
    return torch.tensor(weights, dtype=torch.float32)

def plotLiver():
    
    train_loss = np.load('/home/asaf/Downloads/Liver/results/loss_train.npy')
    train_metric = np.load('/home/asaf/Downloads/Liver/results/metric_train.npy')
    test_loss = np.load('/home/asaf/Downloads/Liver/results/loss_test.npy')
    test_metric = np.load('/home/asaf/Downloads/Liver/results/metric_test.npy')

    
    plt.figure(figsize=(12, 10))

    
    plt.subplot(2, 2, 1)  
    plt.plot(train_loss, color='blue', label='Training Loss')
    plt.title('Training DICE Loss - Liver ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    
    plt.subplot(2, 2, 2)  
    plt.plot(train_metric, color='orange', label='Training Metric')
    plt.title('Training DICE Metric - Liver')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)  
    plt.plot(test_loss, color='green', label='Testing Loss')
    plt.title('Testing DICE Loss - Liver')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4) 
    plt.plot(test_metric, color='red', label='Testing Metric')
    plt.title('Testing DICE Metric - Liver')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)


    plt.show()



def plotPancreas():
    train_loss = np.load('/home/asaf/Downloads/Pancreas/results/loss_train.npy')
    train_metric = np.load('/home/asaf/Downloads/Pancreas/results/metric_train.npy')
    test_loss = np.load('/home/asaf/Downloads/Pancreas/results/loss_test.npy')
    test_metric = np.load('/home/asaf/Downloads/Pancreas/results/metric_test.npy')

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)  
    plt.plot(train_loss, color='blue', label='Training Loss')
    plt.title('Training DICE Loss - Pancreas')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)  
    plt.plot(train_metric, color='orange', label='Training Metric')
    plt.title('Training DICE Metric - Pancreas')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)  
    plt.plot(test_loss, color='green', label='Testing Loss')
    plt.title('Testing DICE Loss - Pancreas')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4) 
    plt.plot(test_metric, color='red', label='Testing Metric')
    plt.title('Testing DICE Metric - Pancreas')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)


    plt.show()




