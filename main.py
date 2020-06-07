"""
main file
"""

import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, Activation
from keras.callbacks import EarlyStopping, TensorBoard
from keras import activations
from keras.models import Sequential
from keras import optimizers
from keras.models import Model
import keras
import time
import tqdm
from PIL import Image

# predefined variables

Image_Height , Image_width = 75 , 75
Batch_size = 32
num_classes = 6
train_class_names = ['Charlock','Common Chickweed','Fat Hen','Loose Silky-bent','Scentless Mayweed','Small-flowered Cranesbill']
feature_space_size = 1024

# this is the path in my google drive
train_path = '/content/drive/My Drive/Python 3/AI_Seedling_train/Labeled'
model_path = '/content/drive/My Drive/Python 3/AI_Seedling_train/Model'

#train_path = '/content/drive/My Drive/Python 3/AI_Seedling_train/To-be-Labeled'

if __name__ == '__main__':

    print('u')
