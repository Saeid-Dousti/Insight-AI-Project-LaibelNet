import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


def load_image(image_path, image_size):
    im = Image.open(image_path)
    w, h = im.size
    res = min(h, w)
    h0, w0 = (h - res) // 2, (w - res) // 2
    im = im.resize(image_size, box=(w0, h0, w0 + res, h0 + res))
    im = im.convert('RGB')
    return im


@st.cache
class Image_set():
    def __init__(self, path, image_size, num_image=None):
        self.path = path
        self.image_size = image_size
        self.num_image = num_image
        self.df = pd.DataFrame()
        self.image_df()

    @st.cache
    def image_df(self):

        self.df = pd.DataFrame([
                    [file,
                    os.path.join(root, file),
                    np.array(load_image(os.path.join(root, file), self.image_size)) / 255.0,
                    root.split(os.sep)[-1]]
                         for root, _, files in os.walk(self.path) for file in files],
                    columns=['name', 'Path', 'Image_np', 'Sub-directory']).sample(n = self.num_image)


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
