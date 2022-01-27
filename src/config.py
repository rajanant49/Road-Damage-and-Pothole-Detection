import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 256 # resize the image for training and transforms
NUM_EPOCHS = 3 # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = '../Pothole Dataset/potholes'
# validation images and XML files directory
# VALID_DIR = '../input/underwater-trash-detection/val'

# classes: 0 index is reserved for background
CLASSES = [
    'background','crack' ,'damage' ,'pothole' ,'pothole_water' ,'pothole_water_m'
]
NUM_CLASSES = 6

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 1 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 1 # save model after these many epochs