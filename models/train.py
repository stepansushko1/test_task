import os
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

preprocessing_dir = os.path.join(current_dir, "../data_processing")

sys.path.append(preprocessing_dir)

os.chdir("../..") # need to be in root directory test_task if you then don't need this

from dataset_airship import DataSet
from constansts import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove tensorflow warning messages

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from create_model import U_NET, U_NET_fullresolution
from metrics import cross_entropy_dice_loss, dice_coef


class Trainer:
    def __init__(self) -> None:
        self.train, self.validation = self.init_datasets()
        self.u_net = self.get_and_train_model()
        self.make_full_resolution()


    def init_datasets(self):
        train_dataset = DataSet(
        os.path.join(current_dir, TRAIN_DF),
        os.path.join(current_dir, IMAGES_PATH),
        BATCH_SIZE,
        INP_IMG_SIZE[0],
        )

        val_dataset = DataSet(
            os.path.join(current_dir, VALIDATION_DF),
            os.path.join(current_dir, IMAGES_PATH), 
            BATCH_SIZE, 
            INP_IMG_SIZE[0]
        )
        return train_dataset, val_dataset
    
    def get_and_train_model(self):

        model = U_NET(INP_IMG_SIZE)
        optimizer = Adam(LEARNING_RATE)

        model.compile(optimizer, cross_entropy_dice_loss, metrics=[dice_coef])


        scheduler = ReduceLROnPlateau(
            factor=FACTOR, patience=PAT, verbose=1, mode='min'
        )

        checkpoint = ModelCheckpoint(
            '/checkpoint/', monitor='val_loss', mode='min', verbose=1
        )
        callbacks = [scheduler, checkpoint]

        history = model.fit(
            self.train,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=self.validation,
        )
        r, dice = model.evaluate(self.validation)
        print('Dice: ', dice)

        return model
    
    def make_full_resolution(self):
 
        unet_full_resolution = U_NET_fullresolution(self.u_net, False)

        unet_full_resolution.save(os.path.join(SAVE_WEIGHTS_PATH))



if __name__ == "__main__":
    trainer_unet = Trainer()

