import sys
sys.path.insert(1,'/code')
import os
from glob import glob
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset

from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from Organ import *
from monai.inferers import sliding_window_inference
from matplotlib.colors import ListedColormap
from utilitati import *

class Ficat(Organ):

    def __init__(self, data_dir, model_dir, model_path):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_path = model_path
            
    def prepare_train(self, in_dir, pixdim=(1.0, 1.0, 1.0), spatial_size=[128, 128, 64], cache=True):
        in_dir = self.data_dir
        path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii")))
        path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentation", "*.nii")))

        train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]

        train_transforms = Compose([
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstD(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),
        ])

        if cache:
            train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
            train_loader = DataLoader(train_ds, batch_size=1)
            return train_loader
        else:
            train_ds = Dataset(data=train_files, transform=train_transforms)
            train_loader = DataLoader(train_ds, batch_size=1)
            return train_loader
        
    def prepare_test(self, pixdim=(1.0, 1.0, 1.0), spatial_size=[128, 128, 64]):
        try:
            in_dir = self.data_dir
            path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii")))
            path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii")))
        except Exception as e:
            print(f"Error {e}")
        test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_test_volumes, path_test_segmentation)]

        test_transforms = Compose([
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstD(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),
        ])

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)
        return test_loader
    
    def train(self,model, data_in, loss, optim, max_epochs, model_dir, test_interval=1 , device=torch.device("cpu" if not torch.cuda.is_available() else "cuda")):
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
    
    def results(self,test_loader,model,device):

        sw_batch_size = 4 # impartim imaginea in bucati mai mici de 4
        roi_size = (128, 128, 64)
        
        with torch.no_grad():

            test_patient = first(test_loader)
            t_volume = test_patient['vol']    

            test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
            sigmoid_activation = torch.nn.Sigmoid()
            test_outputs = sigmoid_activation(test_outputs)
            test_outputs = test_outputs > 0.53
            

            for i in range(46):
                image = test_patient["vol"][0, 0, :, :, i].cpu().numpy()
                label = test_patient["seg"][0, 0, :, :, i] != 0  

        
                plt.figure("check", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title(f"Imaginea {i} ")
                plt.imshow(image, cmap="gray")

                plt.subplot(1, 2, 2)
                plt.title(f" Segmentarea ficatului la imaginea {i} ")
                plt.imshow(np.zeros_like(label), cmap="gray")  
                plt.imshow(label, cmap=ListedColormap(["black", "yellow"]), alpha=0.7)  

                plt.show()