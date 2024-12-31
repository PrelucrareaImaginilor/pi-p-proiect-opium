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
        data_dir = '/home/asaf/Downloads/Liver/Data_Train_Test_Kaggle',
        model_dir = '/home/asaf/Downloads/Liver/results',
        model_path = '/home/asaf/Downloads/Liver/results/best_metric_model.pth'
    )
    pancreas = Pancreas(
        data_dir = '/home/asaf/Downloads/Pancreas/Data_Train_Test_Kaggle',
        model_dir = '/home/asaf/Downloads/Pancreas/results',
        model_path = '/home/asaf/Downloads/Pancreas/results/best_metric_model.pth'
    )

    while True:
        train = input('1 - Train Liver || 2 - Train Pancreas || 3 - PASS ==> ')
        if not train.isdigit() or int(train) not in (1,2,3):
            print('Not a valid choice!')
        elif int(train) == 1:
            data_in = (liver.prepare_train(liver.data_dir, cache=True), liver.prepare_test(spatial_size=[128,128,64]))
            loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
            optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)
            liver.train(model, data_in, loss_function, optimizer, 50, liver.model_dir)
            break
        elif int(train) == 2:
            data_in = (pancreas.prepare_train(pancreas.data_dir, cache=True), pancreas.prepare_test(spatial_size=[128,128,64]))
            loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
            optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)
            pancreas.train(model, data_in, loss_function, optimizer, 50, pancreas.model_dir)
            break
        elif int(train) == 3:
            break

    while True:
        segment = input('1 - Liver Results || 2 - Pancreas Results ==> ')
        if not segment.isdigit() or int(segment) not in (1,2):
            print('Not a valid choice!')
        elif int(segment) == 1:
            test_loader = liver.prepare_test(spatial_size=[128,128,64])
            model.load_state_dict(torch.load(liver.model_path, map_location=device))
            model.eval() # setam modelul pe evaluare, nu il antrenam fiecare data
            liver.results(test_loader,model,device)
            break
        elif int(segment) == 2:
            test_loader = pancreas.prepare_test(spatial_size=[128,128,64])
            model.load_state_dict(torch.load(pancreas.model_path, map_location=device))
            model.eval() # setam modelul pe evaluare, nu il antrenam fiecare data
            pancreas.results(test_loader,model,device)
            break
        
if __name__ == '__main__':
    main()