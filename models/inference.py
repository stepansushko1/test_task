
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

from constansts import *

# os.chdir("../..") # need to be in root directory test_task if you then don't need this

def read_img(filename: str) -> np.ndarray:
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.float32(image / 255)

def select_random_files(n:int):
    file_list = []
    curr_dir = os.path.join(current_dir, IMAGES_PATH_TEST)
    entries = os.listdir(curr_dir)
    random_entries = random.sample(entries, n)

    for entry in random_entries:
        file_list.append(read_img(os.path.join(curr_dir, entry)))

    return np.stack(file_list)

def plot_helper(ax:any, indx:int, title_name:str):
    ax[indx].set_title(title_name, fontsize=14)
    ax[indx].axis('off')
    ax[indx].grid(False)

def check_results(number_of_examples:int):

    model = U_NET_fullresolution(1,True)
    model.load_weights("models/weights_unet.h5")

    test_imgs = select_random_files(number_of_examples)
    masks = model(test_imgs, training=False)
    

    for i, img in enumerate(test_imgs):
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle('Prediction results', fontsize=18)

        # plot orig image
        ax[0].imshow(img)
        plot_helper(ax, 0, "Image")
        
        # plot predicted mask
        mask = masks[i].numpy()
        ax[1].imshow(mask, 'binary_r')
        plot_helper(ax, 1, 'UNET mask')
        fig.tight_layout()

        plt.savefig(os.path.join(current_dir, f"results/res{i}.jpg"), bbox_inches='tight')

if __name__ == "__main__":
    check_results(10)


