
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

'''
PATH-URI trebuie setate in functie de direatoarele utilizatorului !!!!!!! 
'''
def main():
    
    data_dir = 'D:/AC/An III/PI/Project/Liver/Task03_Liver/Data_Train_Test_Kaggle'
    model_dir = 'D:/AC/An III/PI/Project/Liver/Task03_Liver/results' 
    model_path = 'D:/AC/An III/PI/Project/Liver/Task03_Liver/results/best_metric_model.pth'

    # data_in = prepare(data_dir, cache=True)
    #train_loader = prepare_train(data_dir, cache=True)
    test_loader = prepare_test(data_dir, spatial_size=[128,128,64])
    device = torch.device("cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    #loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
    #loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    #optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # setam modelul pe evaluare, nu il antrenam fiecare data
    
    #train(model, data_in, loss_function, optimizer, 50, model_dir)
    
    sw_batch_size = 4 # impartim imaginea in bucati mai mici de 4
    roi_size = (128, 128, 64)
    with torch.no_grad():
        test_patient = first(test_loader)
        t_volume = test_patient['vol']    

        test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
        sigmoid_activation = torch.nn.Sigmoid()
        test_outputs = sigmoid_activation(test_outputs)
        test_outputs = test_outputs > 0.53
        

        for i in range(45):
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

if __name__ == '__main__':
    main()
   
    
            
    