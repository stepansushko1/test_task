# test_task
How to run?

Note! Some packages in requirements.txt are for MacOS!

Create conda env <code> conda activate tensorflow </code>

1. Create directory data/airbus-ship-detection. Here should be stored the data the same way as dataset in kaggle.

2. Set current path to test_task

3. Run <code> python preprocess.py </code>

4. Run <code> train.py </code>

5. Run <code> inference.py </code>


Contains:

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

eda.ipynb - data investigation + visualisation + showing results on test imgs

test_tensorflow.py - test if tensorflow installed and gpu enabled

Accuracy: app. 69%

Possible improvements:
  * add data augmentation 
  * use pretrained models (VGG, ImageNet)
  * check other architectures and improved versions of UNET (UNET++, V-NET, maskRCNN)
