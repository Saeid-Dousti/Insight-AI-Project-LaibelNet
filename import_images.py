import os
import numpy as np
from PIL import Image
from numpy.core._multiarray_umath import ndarray
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_image(image_path, image_size):
    im = Image.open(image_path)
    w, h = im.size
    res = min(h, w)
    h0, w0 = (h - res) // 2, (w - res) // 2
    im = im.resize(image_size, box=(w0, h0, w0 + res, h0 + res))
    im = im.convert('RGB')
    return im


def read_images(path, image_size, task=None, n_images=None):
    if task is None:
        path = os.path.join(path, 'Labeled')

        class_names = os.listdir(path)

        datagen = ImageDataGenerator(
            # set rescaling factor (applied before any other transformation)
            rescale=1. / 255)

        generator = datagen.flow_from_directory(
            directory=path,
            target_size=image_size,
            color_mode="rgb",
            batch_size=32,
            classes=class_names,
            shuffle=True)

        return generator

    else:
        path = os.path.join(path, 'Unlabeled')

        if n_images is None:
            n_images = -1

        image_list = os.listdir(path)

        unlabeled_images: ndarray = np.array(
            [np.array(load_image(os.path.join(path, image), image_size)) / 255.0 for image in image_list[0: n_images]]
        )

        return unlabeled_images
