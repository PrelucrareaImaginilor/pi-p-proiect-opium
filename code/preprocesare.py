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
from monai.utils import set_determinism


def prepare(in_dir, 
            pixdim=(1.0, 1.0, 1.0), 
            a_min=-200, a_max=200, 
            spatial_size=[128, 128, 64],  
            cache=True,
            num_workers=4):
    """
    Function to prepare training and testing data loaders
    """
    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentation", "*.nii")))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii")))

    train_files = [{"vol": image_name, "seg": label_name} 
                   for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} 
                  for image_name, label_name in zip(path_test_volumes, path_test_segmentation)]

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

    test_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstD(keys=["vol", "seg"]),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),
        ToTensord(keys=["vol", "seg"]),
    ])

    # Load data
    dataset_type = CacheDataset if cache else Dataset
    train_ds = dataset_type(data=train_files, transform=train_transforms, cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=1, num_workers=num_workers)

    test_ds = dataset_type(data=test_files, transform=test_transforms, cache_rate=1.0)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers)

    return train_loader, test_loader

def prepare_train(in_dir, pixdim=(1.0, 1.0, 1.0), spatial_size=[128, 128, 64], cache=True):
    
    
    # Get paths for training volumes and segmentations
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


def prepare_test(in_dir, pixdim=(1.0, 1.0, 1.0), spatial_size=[128, 128, 64]):
   
    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii")))

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