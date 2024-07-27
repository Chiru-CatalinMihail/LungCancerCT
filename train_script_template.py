import os
import warnings
warnings.filterwarnings("ignore") # remove some scikit-image warnings

# import monai
# # monai.config.print_config()

from monai.apps import DecathlonDataset
from monai.data import DataLoader, CacheDataset, decollate_batch
# # from monai.data import decollate_patient_batch
# from monai.utils import first, set_determinism
from monai.networks.nets import UNet, DynUNet, AttentionUnet, ViTAutoEnc, UNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, MeanIoU, compute_average_surface_distance
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImage,
    LoadImageD,
    EnsureChannelFirstD,
#     AddChannelD,
    ScaleIntensityD,
    ToTensorD,
    Compose,
    AsDiscreteD,
    SpacingD,
    OrientationD,
    ResizeD,
    RandAffineD,
    AsDiscrete,
    AsDiscreted,
    EnsureTyped,
    EnsureType,
    LoadImageD,
    EnsureChannelFirstD,
    OrientationD,
    SpacingD,
    ScaleIntensityD,
    ResizeD,
    RandAffineD,
    RandFlipD,
    RandRotateD,
    RandZoomD,
#     RandDeformD,
    ToTensorD,
)

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import sys
from tqdm import tqdm
import pandas as pd
import pickle as pkl

from hyperparams import *

from torch.utils.tensorboard import SummaryWriter


### HYPERPARAMS ###
crt_dir = os.getcwd()
datasets_path = f'{crt_dir}/datasets/MedicalDecathlon/'
checkpoints_path = f'{crt_dir}/checkpoints/UNET'

best_model_name = checkpoints_path +  "best_dice_unet.pth"
DEBUG_MODE = True


### PREPROCESSING DATA ###
train_transform = Compose([
    LoadImageD(keys=KEYS),
    # EnsureChannelFirstD(keys=KEYS),
    # OrientationD(keys=KEYS, axcodes='RAS'),
    # SpacingD(keys=KEYS, pixdim=(1., 1., 1.), mode=('bilinear', 'nearest')),
    # ScaleIntensityD(keys="image"),
    # ResizeD(keys=KEYS, spatial_size=(IMG_HEIGHT, IMG_HEIGHT, NO_STACKED_IMGS), mode=('trilinear', 'nearest')),
    # # ResizeD(keys=KEYS, spatial_size=(128, 128, 64), mode=('trilinear', 'nearest')),

    # RandAffineD(
    #     keys=KEYS,
    #     spatial_size= (IMG_HEIGHT, IMG_HEIGHT, NO_STACKED_IMGS),

    #     # spatial_size=(128, 128, 64),
    #     rotate_range=(0, 0, np.pi/12),
    #     scale_range=(0.1, 0.1, 0.1),
    #     mode=('bilinear', 'nearest'),
    #     prob=0.5
    # ),
    # RandFlipD(keys=KEYS, spatial_axis=[0,1], prob=0.5),
    # RandRotateD(keys=KEYS, range_x=np.pi/12, range_y=np.pi/12, range_z=np.pi/12, prob=0.5),
    # RandZoomD(keys=KEYS, min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ToTensorD(keys=KEYS),
])

val_transform = Compose([
    LoadImageD(keys = KEYS),
    EnsureChannelFirstD(keys = KEYS),
    OrientationD(KEYS, axcodes='RAS'),
    SpacingD(keys = KEYS, pixdim = (1., 1., 1.), mode = ('bilinear', 'nearest')),
    ScaleIntensityD(keys = "image"),
    ResizeD(KEYS, (IMG_HEIGHT, IMG_HEIGHT, NO_STACKED_IMGS), mode=('trilinear', 'nearest')),
    ToTensorD(KEYS),
])


### LOSS ###
loss_function = DiceLoss(to_onehot_y = True, softmax = True)


### MODEL HYPERPARAMS ###
UNet_metadata = dict(
    spatial_dims = 3,
    in_channels = 1,
    out_channels = 2,
    channels = (64, 128, 256, 512),
    strides = (2, 2, 2),
    num_res_units = 2,
    norm = Norm.BATCH,
    # act = torch.nn.ReLU,
    dropout = 0.1
)

### PERFORMANCE METRICS ###
dice_metric = DiceMetric(include_background = False, reduction = "mean") # include_background = False,
iou_metric = MeanIoU(include_background=False, reduction = "mean", get_not_nans=False, ignore_empty=True)


### TRAINING SETUP ###
def train(model, train_dataset, train_loader, val_loader, loss_function, lr_scheduler, MAX_EPOCHS=20, VALIDATION_INTERVAL=2):

    # Variables to get the best model
    best_dice = -1
    best_metrics = None
    best_metric_epoch = -1

    # Evaluation metrics per epoch
    dice_values = []
    iou_values = []

    epoch_loss_values = []

    post_pred = Compose([AsDiscrete(argmax = True, to_onehot = 2)])
    post_label = Compose([AsDiscrete(to_onehot = 2)])

    for epoch in range(MAX_EPOCHS):
        print("-" * 12)
        print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")

        # Turn model to "train" mode
        model.train()

        epoch_loss = 0
        for step, batch_data in enumerate(train_loader):
            step += 1

            input, label = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )

            # # A common pytorch Deep Learning format to train model
            # optimizer.zero_grad()
            output = model(input)

            loss = loss_function(output, label)
            loss.backward() # Compute gradient
            optimizer.step() # Update model's parameters

            epoch_loss += loss.item()
            print(f"{step}/{len(train_dataset) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            # Save current checkpoint of the network

            print(f"Saving checkpoint: {epoch//VALIDATION_INTERVAL + 1} / {MAX_EPOCHS//VALIDATION_INTERVAL}!!!")
            name = checkpoints_path + f'unet_lr{LEARNING_RATE}_epoch{epoch}.pth'
            torch.save(model.state_dict(), name)

            # Turn model to "eval" mode
            model.eval()
            lr_scheduler.step()

            # Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward().
            # It will reduce memory consumption for computations that would otherwise have requires_grad=True
            with torch.no_grad():
                iteration_ious = []
                iteration_pixel_accuracies = []
                iteration_rvds = []

                for val_data in val_loader:
                    val_input, val_label = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )

                    roi_size = (128, 128, 96)
                    sw_batch_size = 1

                    # Set AMP for MONAI validation
                    val_output = sliding_window_inference(
                        val_input, roi_size, sw_batch_size, model
                    )
                    val_output = [post_pred(i) for i in decollate_batch(val_output)]
                    val_label = [post_label(i) for i in decollate_batch(val_label)]


                    # Compute metrics for current iteration
                    dice_metric(y_pred = val_output, y = val_label)
                    iou_metric(y_pred=val_output, y=val_label)

            # Aggregate the final mean results
            dice_score = dice_metric.aggregate().item()
            mean_iou = iou_metric.aggregate().item()

            # Reset the status for the next epoch
            dice_metric.reset()
            iou_metric.reset()

            dice_values.append(dice_score)
            iou_values.append(mean_iou)

            if dice_score > best_dice:
                best_dice = dice_score
                best_metrics = (dice_score, mean_iou)
                best_metric_epoch = epoch + 1
                print("saved new best metric model!!!")

                torch.save(model.state_dict(), best_model_name)

        print(
            f"current epoch: {epoch + 1},"
            f" current mean dice: {dice_score:.4f},"
            f" current mean iou: {mean_iou:.4f},"
            f" best mean dice: {best_dice:.4f},"
            f" at epoch: {best_metric_epoch}"
        )

    print(
        f"train completed, metrics correspondic to best dice are: dice: {best_metrics[0]:.4f}, iou: {best_metrics[1]:.4f}, acc: {best_metrics[2]:.4f}, rvd: {best_metrics[3]:.4f}"
        f" at epoch: {best_metric_epoch}"
    )

    with open(checkpoints_path + 'metrics_evolution.pkl', 'wb') as f:
        pkl.dump((dice_values, iou_values, epoch_loss_values), f)

if __name__ == "__main__":
    # Initialize torch and cuda
    cuda = torch.cuda.is_available()

    device = torch.device("cuda" if cuda else "cpu")
    num_workers = 4 if cuda else 1

    print(f'You are using {device}')

    print(f'Number of images in a stack: {NO_STACKED_IMGS}')

    # Set random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load data

    ### TRAINING DATA ###
    train_name = 'validation' # From Monai: ['training', 'validation', 'test']
    train_dataset = DecathlonDataset(root_dir = f'{datasets_path}{train_name}/',
                            task = "Task06_Lung", section = train_name,
                            transform = train_transform, download = False)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = 1 - DEBUG_MODE, num_workers = num_workers)


    ### VALIDATION DATA ###
    val_name = 'validation'
    val_dataset = DecathlonDataset(root_dir = f'{datasets_path}{val_name}/',
                            task = "Task06_Lung", section = val_name,
                            transform = val_transform, download = False)

    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = 1 - DEBUG_MODE, num_workers = num_workers)
    
    # Instantiate model
    model = UNet(**UNet_metadata).to(device)


    # Instantiate optimizer
    optimizer = torch.optim.NAdam(model.parameters(), lr = LEARNING_RATE)

    # Instantiate learning rate scheduler
    # TODO - add lr_scheduler as part of training
    decayRate = 0.96
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    # Train model
    train(model, train_dataset, train_loader, val_loader, loss_function, lr_scheduler, MAX_EPOCHS=MAX_EPOCHS, VALIDATION_INTERVAL=VALIDATION_INTERVAL)

