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

print('hello world')

if __name__ == '__main__':

    pass
