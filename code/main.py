import sys
sys.path.insert(1,'/code')
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss
import torch
from preprocesare import *
from utilitati import *
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from Ficat import *

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

    device = torch.device("cuda")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    liver = Ficat()

    if ok == "y" or ok == "yes":
        data_in =liver.prepare(cache=True)
   
        loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

        liver.train(model, data_in, loss_function, optimizer, 50,liver.model_dir)
    else:

        test_loader = liver.prepare_test(spatial_size=[128,128,64])
        model.load_state_dict(torch.load(liver.model_path, map_location=device))
        model.eval() # setam modelul pe evaluare, nu il antrenam fiecare data
        
        liver.results(test_loader,model,device)
    

if __name__ == '__main__':
    main()