import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.utils import Sequence

class DataSet(Sequence):
    def __init__(self,
        raw_data_csv: str,
        target_dir: str,
        batch_size: int = 64,
        image_size: int = 768,

    ):

        self.raw_data_csv = pd.read_csv(raw_data_csv)
        self.target_dir = target_dir
        self.batch_size= batch_size
        self.image_size = image_size

    def read_one_image(self, filename: str):
        img = cv2.imread(self.target_dir + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        return np.float32(img / 255)
    
    def read_mask(self, rle_str):
        mask = rle_decode(rle_str)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        return mask.astype(np.float32)

    def __len__(self):
        return len(self.raw_data_csv) // self.batch_size
    
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            filename = self.raw_data_csv['ImageId'].iloc[i]
            rle_str = self.raw_data_csv['EncodedPixels'].iloc[i]

            img = self.read_one_image(filename)
            mask = self.read_mask(rle_str)

            batch_x.append(img)
            batch_y.append(mask)

        return np.array(batch_x), np.array(batch_y).reshape(-1, self.image_size, self.image_size, 1)

def rle_decode(mask_rle: str, shape=(768, 768)):
    """
    Creates a binary mask from run-length-encoded string
    """
    if not isinstance(mask_rle, str):
        return np.zeros(shape, dtype=np.uint8)
    
    values = mask_rle.split()
    starts, lengths = [], []
    
    for i in range(0, len(values), 2):
        starts.append(int(values[i]) - 1)
        lengths.append(int(values[i + 1]))
    
    ends = []
    for start, length in zip(starts, lengths):
        ends.append(start + length)
    
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    for start, end in zip(starts, ends):
        img[start:end] = 1
    
    return img.reshape(shape).T