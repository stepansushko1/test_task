
import os
import random
import cv2
import numpy as np
import sys
from create_model import U_NET_fullresolution

import matplotlib.pyplot as plt
# os.chdir("../../")
current_dir = os.path.dirname(os.path.abspath(__file__))

preprocessing_dir = os.path.join(current_dir, "../data_processing")

sys.path.append(preprocessing_dir)

os.chdir("../..") # need to be in root directory test_task if you then don't need this

from constansts import *

def read_img(filename: str) -> np.ndarray:
    """Reads image from file, and clips its' values to be in [0, 1] range"""
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.float32(image / 255)

def select_random_files(directory, n:int):
    file_list = []
    curr_dir = os.path.join(current_dir, IMAGES_PATH_TEST)
    print('sjnhdfjshdfojsjdofs', curr_dir)
    entries = os.listdir(curr_dir)
    random_entries = random.sample(entries, n)

    for entry in random_entries:
        file_list.append(read_img(os.path.join(curr_dir, entry)))


    return np.stack(file_list)

def check_results():

    model = U_NET_fullresolution(1,True)
    model.load_weights("models/weights_unet.h5")


    test_imgs = select_random_files(IMAGES_PATH_TEST, 10)
    masks = model(test_imgs, training=False)
    

    for i, img in enumerate(test_imgs):
        fig, ax = plt.subplots(1, 3, figsize=(12, 8))
        fig.suptitle('Prediction results', fontsize=18)

        # plot orig image
        ax[0].imshow(img)
        ax[0].set_title('Original image', fontsize=14)
        ax[0].axis('off')
        ax[0].grid(False)
        
        # plot predicted mask
        mask = masks[i].numpy()
        print(mask)
        ax[1].imshow(mask, 'binary_r')
        ax[1].set_title('Predicted mask', fontsize=14)
        ax[1].axis('off')
        ax[1].grid(False)

        # plot both image and mask
        ax[2].imshow(img)
        ax[2].imshow(mask, 'binary_r', alpha=0.4)
        ax[2].set_title('Image and mask', fontsize=14)
        ax[2].axis('off')
        ax[2].grid(False)
        fig.tight_layout()

        plt.savefig(os.path.join(current_dir, f"results/res{i}.jpg"), bbox_inches='tight')


check_results()


