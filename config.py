import torch

DATA_DIR = 'C:/Users/ASUS/OneDrive/Documents/emotion-detector/data'

EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 32
VAL_SPLIT = 0.2
IMG_SIZE = 64

MODEL_PATH = './model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')