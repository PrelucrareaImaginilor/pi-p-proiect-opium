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
import os
from monai.inferers import sliding_window_inference
from matplotlib.colors import ListedColormap
from Organ import *
from utilitati import *
from Organ import *
from Train import *

class Pancreas(Organ, Train):
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
                plt.title(f" Segmentarea pancreasului la imaginea {i} ")
                plt.imshow(np.zeros_like(label), cmap="gray")  
                plt.imshow(label, cmap=ListedColormap(["black", "yellow"]), alpha=0.7)  

                plt.show()