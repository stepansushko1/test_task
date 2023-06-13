# test_task

Create directory data/airbus-ship-detection. Here should be stored the data the same way as dataset in kaggle.

Contains

data_processing:

* constansts.py - here all path needed for project
* dataset_airship.py - dataset class for training for keras
* preprocess.py - adding new row counting ships, balancing dataset

models:
  results:
    * file.jpg - visulisation of working
* create_model.py - creates UNET and Full Resolution UNET (for efficiancy)
* inference.py - prediction of mask and showing results of model
* metrics.py - dice coefficient and loss
* train.py - for training model
* weights_unet.h5 - trained model

test_jupiter.ipynb - basic version of eda

test_tensorflow.py - test if tensorflow installed and gpu enabled

Accuracy: app. 69%

Possible improvements:
  * add data augmentation 
  * use pretrained models (VGG, ImageNet)
  * check other architectures and improved versions of UNET (UNET++, V-NET, maskRCNN)
