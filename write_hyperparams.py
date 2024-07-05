### Hyperparameters for the models ###

from monai.networks.layers import Norm
import yaml


if __name__ == "__main__":

    BATCH_SIZE = 16 # 40 maxim pentru reteaua mea, pentru reteaua de 8M hyperparametrii e 16

    # IMG_HEIGHT = 128
    # IMG_WIDTH = 128
    COLOUR_CHANNELS = 1
    NO_STACKED_IMGS = 6

    KEYS = ["image", "label"]

    LEARNING_RATE = 3e-2
    decayRate = 0.96

    MAX_EPOCHS = 10
    VALIDATION_INTERVAL = 2

    # UNet_metadata_8M = dict(
#     spatial_dims = 3,
#     in_channels = 1,
#     out_channels = 2,
#     channels = (16, 32, 32, 128),
#     kernel_size = (5, 5, 5),
#     strides = (1, 2, 1, 2),
#     num_res_units = 4,
#     norm = Norm.BATCH,
#     # act = torch.nn.ReLU,
#     dropout = 0.1
# )


    UNet_metadata = dict(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = 2,
        channels = (16, 32, 32, 128),
        kernel_size = (5, 5, 5),
        strides = (1, 1, 1),
        num_res_units = 4,
        norm = Norm.BATCH,
        # act = torch.nn.ReLU,
        dropout = 0.1
    )

    hyperparams = {"BATCH_SIZE": BATCH_SIZE,
                #    "IMG_HEIGHT": IMG_HEIGHT,
                #    "IMG_WIDTH": IMG_WIDTH,
                   "COLOUR_CHANNELS": COLOUR_CHANNELS,
                   "NO_STACKED_IMGS": NO_STACKED_IMGS,
                   "KEYS": KEYS,
                   "LEARNING_RATE": LEARNING_RATE,
                   "decayRate": decayRate,
                   "MAX_EPOCHS": MAX_EPOCHS,
                   "VALIDATION_INTERVAL": VALIDATION_INTERVAL,
                   "UNet_metadata": UNet_metadata}

    with open("config.yaml", "w") as file:
        yaml.dump(hyperparams, file)
