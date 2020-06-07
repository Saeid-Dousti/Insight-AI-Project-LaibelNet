"""
main file
"""
# ---------------------------------------------------------------

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

# ---------------------------------------------------------------

# predefined variables

Image_Height, Image_width = 75, 75
Batch_size = 32
num_classes = 6
train_class_names = ['Charlock', 'Common Chickweed',
                     'Fat Hen', 'Loose Silky-bent',
                     'Scentless Mayweed',
                     'Small-flowered Cranesbill']
feature_space_size = 1024

# this is the path in my google drive
train_path = 'C:/Users/Saeid/Documents/AI_project/input/train'


# model_path = '/content/drive/My Drive/Python 3/AI_Seedling_train/Model'

# ---------------------------------------------------------------

# time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def predic_gen(path, image_size, batch_size, data_classes=None):
    if data_classes is None:
        data_classes = train_class_names

    datagen = ImageDataGenerator(
        # set rescaling factor (applied before any other transformation)
        rescale=1. / 255)

    generator = datagen.flow_from_directory(
        directory=path,
        target_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        classes=data_classes,
        shuffle=True,
        subset='prediction')

    return generator


train_generator = datagen.flow_from_directory(
    directory=train_path,
    target_size=(Image_width, Image_Height),
    color_mode="rgb",
    batch_size=Batch_size,
    class_mode="categorical",
    classes=train_class_names,
    shuffle=True,
    subset='training')

val_generator = datagen.flow_from_directory(
    directory=train_path,
    target_size=(Image_width, Image_Height),
    color_mode="rgb",
    batch_size=Batch_size,
    class_mode="categorical",
    classes=train_class_names,
    subset='validation')

if __name__ == '__main__':
    # Transfer Learning: pretrained Resnet

    base_model = ResNet50(include_top=False, weights='imagenet',
                          input_shape=(Image_Height, Image_width, 3))


