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

""""
Trebuie create cele 4 foldere -> TrainVolumes, TrainSegmentation, TestVolumes si TestSegmentation !!!!!!!
Pentru adaugare de transformari mari, acestea trebuie realizate in partea de *train*, nu de testing!
"""



def prepare(in_dir, 
            pixdim=(1.0, 1.0, 1.0), 
            a_min=-200, a_max=200, 
            spatial_size=[128, 128, 64] # al treilea parametru ar trebuie sa fie numarul de slice-uri CORECT !!!! (minimul dintre toate imaginile)
            , cache=True):
    
    """
    functie de preparare conform https://monai.io/docs.html
    """

    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentation", "*.nii")))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in  #crearea de dictionar pentru antrenare
                   zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in  #crearea de dictionar pentru testare
                  zip(path_test_volumes, path_test_segmentation)]                       

    train_transforms = Compose(    # functie de transformare -> "Compose" ofera disponibilitatea pentru transformare a mai multor transformari
        [
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstD(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")), # Spacingd folosit pentru a regla corect dimensiunile la imagini
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True), # variabilele a si b sunt folosite pentru a schimba contrastul imaginilior pacientilor
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]), # convertire in tensor

        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]), # d -> de la dictionar 
            EnsureChannelFirstD(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'), # transformare importanta
            Resized(keys=["vol", "seg"], spatial_size=spatial_size), 
            ToTensord(keys=["vol", "seg"]),

        ]
    )


    #load la date
    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0) # load in GPU
        train_loader = DataLoader(train_ds, batch_size=1) # pune transformarile in RAM si le aplica pe imagini
        
        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader