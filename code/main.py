import sys
sys.path.insert(1,'/code')
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
import torch
from preprocesare import *
from utilitati import *
#from monai.inferers import sliding_window_inference
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
from Ficat import *
from Pancreas import *

'''
PATH-URI trebuie setate in functie de direatoarele utilizatorului !!!!!!! 
'''
def main():

    while True:
        ok = input("Do you want to train the model?y/n\n")
        if ok == "y" or ok == "n" or ok == "yes" or ok == "no":
            break
        else:
            print("Enter a valid answer")

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    liver = Ficat(
        data_dir = 'D:/AC/An III/PI/Project/Liver/Task03_Liver/Data_Train_Test_Kaggle',
        model_dir = 'D:/AC/An III/PI/Project/Liver/Task03_Liver/results',
        model_path = 'D:/AC/An III/PI/Project/Liver/Task03_Liver/results/best_metric_model.pth'
    )
    pancreas = Pancreas(
        data_dir = 'D:/AC/An III/PI/Project/Pancreas/Pancreas/Data_Train_Test_Kaggle',
        model_dir = 'D:/AC/An III/PI/Project/Pancreas/Pancreas/results',
        model_path = 'D:/AC/An III/PI/Project/Pancreas/Pancreas/results/best_metric_model.pth'
    )

    if ok == "y" or ok == "yes":
        check = input('1 - Ficat || 2 - Pancreas\n')
        if check == "1":
            data_in = (liver.prepare_train(liver.data_dir, cache=True), liver.prepare_test(spatial_size=[128,128,64]))
            loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
            optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)
            liver.train(model, data_in, loss_function, optimizer, 50, liver.model_dir)
        elif check == "2":
            data_in =(pancreas.prepare_train(pancreas.data_dir, cache=True), pancreas.prepare_test(spatial_size=[128,128,64]))
            loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
            optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)
            pancreas.train(model, data_in, loss_function, optimizer, 50, pancreas.model_dir)

    if ok == "n" or ok == "no":
        check = input('1 - Ficat || 2 - Pancreas\n')
        if check == "1":
            test_loader = liver.prepare_test(spatial_size=[128,128,64])
            model.load_state_dict(torch.load(liver.model_path, map_location=device))
            model.eval() # setam modelul pe evaluare, nu il antrenam fiecare data
            liver.results(test_loader,model,device)
        elif check == "2":
            test_loader = pancreas.prepare_test(spatial_size=[128,128,64])
            model.load_state_dict(torch.load(pancreas.model_path, map_location=device))
            model.eval() # setam modelul pe evaluare, nu il antrenam fiecare data
            pancreas.results(test_loader,model,device)


if __name__ == '__main__':
    main()