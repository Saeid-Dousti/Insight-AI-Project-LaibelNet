import os
from load_image import load_image
from PIL import Image
import pandas as pd
import numpy as np
from sklearn import preprocessing


class Image_set():

    def __init__(self, path, image_size, num_image=None):
        self.path = path
        self.image_size = image_size
        self.num_image = num_image
        self.image_frame()

    def image_frame(self):

        image_name = []
        image_label = []
        image_path = []
        image_nparray = []

        if self.num_image is None:
            for root, _, files in os.walk(self.path):
                for file in files:
                    try:
                        temp_path = os.path.join(root, file)
                        temp = load_image(temp_path, self.image_size)
                        image_nparray.append(np.array(temp) / 255.0)
                        image_name.append(file)
                        image_path.append(temp_path)
                        image_label.append(root.split(os.sep)[-1])

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
                        image_nparray.append(np.array(temp) / 255.0)
                        image_name.append(file)
                        image_path.append(temp_path)
                        image_label.append(root.split(os.sep)[-1])

                    except:
                        print(f'--- {temp_path} is a none image file ---')

                    if self.num_image is not None:
                        loop += 1
                        if loop >= self.num_image:
                            break

        le = preprocessing.LabelEncoder()
        le.fit(image_label)
        print(le.transform(image_label))

        df = pd.DataFrame(list(zip(np.arange(len(image_name)), image_name, image_label, image_path, image_nparray,
                                   le.transform(image_label))),
                          columns=['Index', 'Name', 'Label', 'Path', 'NP Array', 'Label Code'])

        self.image_df = df

