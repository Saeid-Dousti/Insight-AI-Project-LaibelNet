import os
import time
import streamlit as st
from load_image import load_image
import pandas as pd
import numpy as np
from sklearn import preprocessing


@st.cache
class Image_set():
    def __init__(self, path, image_size, num_image=None):
        self.path = path
        self.image_size = image_size
        self.num_image = num_image
        self.image_name = []
        self.image_path = []
        self.image_label = []
        self.image_nparray = []
        self.image_frame()

    @st.cache
    def image_frame(self):

        if self.num_image is None:
            for root, _, files in os.walk(self.path):
                for file in files:
                    try:
                        temp_path = os.path.join(root, file)
                        temp = load_image(temp_path, self.image_size)
                        self.image_nparray.append(np.array(temp) / 255.0)
                        self.image_name.append(file)
                        self.image_path.append(temp_path)
                        self.image_label.append(root.split(os.sep)[-1])

                    except:
                        print(f'--- {temp_path} is a none image file ---')

        else:
            loop = 0
            for root, _, files in os.walk(self.path):
                if self.num_image is not None:
                    if loop >= self.num_image:
                        break
                for file in files:

                    try:
                        temp_path = os.path.join(root, file)
                        temp = load_image(temp_path, self.image_size)
                        self.image_nparray.append(np.array(temp) / 255.0)
                        self.image_name.append(file)
                        self.image_path.append(temp_path)
                        self.image_label.append(root.split(os.sep)[-1])

                    except:
                        print(f'--- {temp_path} is a none image file ---')

                    if self.num_image is not None:
                        loop += 1
                        if loop >= self.num_image:
                            break


        self.image_nparray = np.asarray(self.image_nparray)


