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

def train(model, data_in, loss, optim, max_epochs, model_dir, test_interval=1 , device=torch.device("cuda")):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    train_loader, test_loader = data_in

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        for batch_data in train_loader:
            
            train_step += 1

            volume = batch_data["vol"]
            label = batch_data["seg"]
            label = label != 0
            volume, label = (volume.to(device), label.to(device))

            optim.zero_grad()
            outputs = model(volume)
            
            train_loss = loss(outputs, label)
            
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")

            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        print('-'*20)
        
        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        
        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')

        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        if (epoch + 1) % test_interval == 0:

            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                epoch_metric_test = 0
                test_step = 0

                for test_data in test_loader:

                    test_step += 1

                    test_volume = test_data["vol"]
                    test_label = test_data["seg"]
                    test_label = test_label != 0
                    test_volume, test_label = (test_volume.to(device), test_label.to(device),)
                    
                    test_outputs = model(test_volume)
                    
                    test_loss = loss(test_outputs, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(test_outputs, test_label)
                    epoch_metric_test += test_metric
                    
                
                test_epoch_loss /= test_step
                print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                save_loss_test.append(test_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

                epoch_metric_test /= test_step
                print(f'test_dice_epoch: {epoch_metric_test:.4f}')
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth"))
                
                print(
                    f"current epoch: {epoch + 1} current mean dice: {test_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )


    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")

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
    plt.title('Training DICE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)  
    plt.plot(train_metric, color='orange', label='Training Metric')
    plt.title('Training DICE Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)  
    plt.plot(test_loss, color='green', label='Testing Loss')
    plt.title('Testing DICE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4) 
    plt.plot(test_metric, color='red', label='Testing Metric')
    plt.title('Testing DICE Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)


    plt.show()




