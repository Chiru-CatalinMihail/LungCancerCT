### COUNTS THE NUMBER OF TRAINABLE HYPERPARAMETERS

from prettytable import PrettyTable
import torch
import torch.nn as nn
from monai.networks.nets import UNet
import yaml

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    # Get the total number of parameters as millions
    print(f"Total Trainable Params: {total_params/1e6:.2f} M params")
    return total_params
    


# SEND THIS TO A CONFIG FILE




if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    print('Start main')
    
    with open("config.yaml", "r") as file:
        UNet_metadata = yaml.load(file, Loader=yaml.FullLoader)["UNet_metadata"]

    print('Loaded config file')


    net = nn.Sequential(nn.BatchNorm3d(1, affine=True), UNet(**UNet_metadata), nn.InstanceNorm3d(2, affine=True)).to(device)

    random_input = torch.rand(16, 1, 512, 512, 6).to(device)

    print(f"Output shape: {net(random_input).shape}")

    count_parameters(net)