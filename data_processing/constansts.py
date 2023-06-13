import os

RAW_CVS_PATH = "../data/airbus-ship-detection/train_ship_segmentations_v2.csv"

CLEANED_CSV_PATH = "../data/cleaned_csv.csv"

BALANCED_CSV_PATH = "../data/balanced.csv"

TRAIN_DF = "../data/train_df.csv"

VALIDATION_DF = "../data/val_df.csv"

IMAGES_PATH = "../data/airbus-ship-detection/train_v2/"

IMAGES_PATH_TEST = "../data/airbus-ship-detection/test_v2/"

SAVE_WEIGHTS_PATH = "weights_unet.h5"

RANDOM_STATE = 42 # default value

BATCH_SIZE = 64

EPOCHS = 1

LEARNING_RATE = 0.001

FACTOR = 0.33

PAT = 2

INP_IMG_SIZE = (256, 256, 3)

INP_IMG_SIZE_FULL_RES = (768, 768, 3)
