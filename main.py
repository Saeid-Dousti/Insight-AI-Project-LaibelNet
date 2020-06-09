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
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from keras import activations
from keras.models import Sequential
from keras import optimizers
from keras.models import Model
from sklearn.mixture import GaussianMixture as GMM
#from sklearn.mixture import bayesianGaussianMixture as BGMM
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import keras
import time
import tqdm
import pickle
from PIL import Image

# ---------------------------------------------------------------

# predefined variables

Image_Height, Image_width = 250, 250
Batch_size = 100
num_classes = 12
class_names = ['Charlock', 'Common Chickweed', 'Black-grass',
               'Fat Hen', 'Loose Silky-bent', 'Sugar beet', 'Maize',
               'Scentless Mayweed', 'Shepherds Purse', 'Cleavers',
               'Small-flowered Cranesbill', 'Common wheat']
feature_space_size = 2048

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


def predic_gen(path, batch_size, image_size=None, data_classes=None):
    '''
    this function creates data generator
    :param path:
    :param image_size:
    :param batch_size:
    :param data_classes:
    :return:
    '''

    if data_classes is None:
        data_classes = class_names

    datagen = ImageDataGenerator(
        # set rescaling factor (applied before any other transformation)
        rescale=1. / 255)

    if image_size is None:

        generator = datagen.flow_from_directory(
            directory=path,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            classes=data_classes,
            shuffle=True,
            subset='training')
    else:

        generator = datagen.flow_from_directory(
            directory=path,
            target_size=image_size,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            classes=data_classes,
            shuffle=True,
            subset='training')

    return generator


def my_model():

    model = ResNet50(include_top=False, weights='imagenet',
                     input_shape=(Image_Height, Image_width, 3))

    out_lay = GlobalAveragePooling2D()(model.output)

    model = Model(model.input, out_lay)

    return model


def read_images():
    """
    - this function loads data and their labels from the original input folder
        or the associated pickle file

    :return: images of plant seedling and their labels of 12 classes or 10 fashion
    """


    if os.path.isfile('input/image_n_label.pickle'):

        with open('input/image_n_label.pickle', 'rb') as f:
            image_n_label = pickle.load(f)

    else:

        img_size = (Image_Height, Image_width)

        pred_gen = predic_gen(train_path, Batch_size, img_size)

        image_n_label = next(pred_gen)

        with open('./input/image_n_label.pickle', 'wb') as f:
                pickle.dump(image_n_label, f)

    return image_n_label


def df_maker(feature, labels):
    '''
    create data frame to store clustering info
    :return: data frame
    '''
    df = pd.DataFrame()
    df['img_ftrs'] = [feature[i, :] for i in range(len(feature[:, 0]))]
    df['Real_Labls'] = [labels[i].argmax() for i in range(len(feature[:, 0]))]

    return df


def main():
    # Transfer Learning: pretrained Resnet

    imgs, labels = read_images(1) # any input can indicated fashion-mnist dataset

    model = my_model()

    features = model.predict(imgs)

    df = df_maker(features, labels)

    kmeans = KMeans(n_clusters=12, random_state=0).fit(features)

    gmm = GMM(n_components = 12).fit(features)

    #bgmm = BGMM(n_components=12).fit(features)

    df['KMN_Labls'] = kmeans.labels_

    df['gmm'] = gmm.predict(features)

    #df['bgmm'] = bgmm.predict(features)

if __name__ == '__main__':

    main()