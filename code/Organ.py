from abc import ABC
import torch

class Organ(ABC):
    def __init__(self):
        pass
    def prepare(self, pixdim=(1.0, 1.0, 1.0),a_min = -200, a_max = 200, spatial_size=[128, 128, 64], cache=True, num_workers = 4):
        pass

    def prepare_test(self, pixdim=(1.0, 1.0, 1.0), spatial_size=[128, 128, 64]):
        pass

    def train(self,model, data_in, loss, optim, max_epochs, model_dir, test_interval=1 , device = torch.device("cuda")):
        pass

    def results(self,test_loader,model,device):
        pass
