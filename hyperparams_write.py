### Hyperparameters for the models ###

BATCH_SIZE = 16 # 40 maxim pentru reteaua mea, o sa vad care e maximul pentru MONAI U-Net

IMG_HEIGHT = 128
IMG_WIDTH = 128
COLOUR_CHANNELS = 1
NO_STACKED_IMGS = 64 # FOLOSESC DIRECT MINIBATCH-URI DE CATE 6 IMAGINI

KEYS = ["image", "label"]

LEARNING_RATE = 3e-2
decayRate = 0.96

MAX_EPOCHS = 10
VALIDATION_INTERVAL = 2
