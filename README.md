# Histopathologic-Cancer-Detection-using-DeepLearning
This was my capstone project for udacity's Machine Learning Engineer Nanodegree
For all the detailed description of this project please read the report.

This project was implemented using python 3 in a google colaboratory notebook with GPU as a hardware accelerator, the deep learning library Keras with TensorFlow as a backend and Kaggle
API to download the dataset.
______________________________________________________________

libraries used in this project: all libraries are imported in the first cell of the ipynb file
________________________________
import pandas as pd

import os

import numpy as np



from keras.preprocessing.image import ImageDataGenerator

from keras.engine.input_layer import Input

from keras.layers import Dense, Concatenate

from keras.applications.inception_v3 import InceptionV3

from keras.applications.resnet import ResNet152

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.initializers import RandomNormal

from keras.models import Model

from keras.utils import plot_model

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.optimizers import RMSprop


import random

random.seed(42)


%matplotlib inline


import matplotlib.pyplot as plt

import matplotlib.image as mpimg


from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
______________________________________________________________

download the dataset from kaggle: this step is done in the second cell of the ipynb file, when you run the cell an Upload widget will appear you have to upload kaggle.json.

get the kaggle.json file (Create an API key in Kaggle):
My account -> Create New API Tokens. This will download a kaggle.json file to your computer.  
________________________________
#to use kaggle api

from google.colab import files

files.upload() #upload kaggle.json token file in ~/.kaggle/

!pip install kaggle

!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

# Download the data

!kaggle competitions download -c histopathologic-cancer-detection --force
________________________________

The dataset I'm using in this project can be found here:
https://www.kaggle.com/c/histopathologic-cancer-detection/data
