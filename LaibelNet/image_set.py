import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn import preprocessing


def load_image(image_path, image_size):
    im = Image.open(image_path)
    w, h = im.size
    res = min(h, w)
    h0, w0 = (h - res) // 2, (w - res) // 2
    im = im.resize(image_size, box=(w0, h0, w0 + res, h0 + res))
    im = im.convert('RGB')
    return im


def label_encoding(list_):
    le = preprocessing.LabelEncoder()

    return le.fit_transform(list_)

@st.cache
def imageset_dataframe(path, image_size, num_image):
    df = pd.DataFrame([
        [file,
         os.path.join(root, file),
         np.array(load_image(os.path.join(root, file), image_size)) / 255.0,
         root.split(os.sep)[-1]]
        for root, _, files in os.walk(path) for file in files],
        columns=['Image', 'Path', 'Image_np', 'Sub-directory']).sample(n=num_image).reset_index(drop=True)

    #st.write(list(df['Sub-directory']))
    #print(label_encoding(list(df['Sub-directory'])))
    df['Encoded Sub-directory'] = label_encoding(list(df['Sub-directory']))

    return df
