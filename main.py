import argparse
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from matplotlib import rcParams
import numpy as np
from import_images import read_images
from cnn_model import cnn_model
from cluster import clustering
from keras.preprocessing.image import ImageDataGenerator
import streamlit as st
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import keras
import time
import tqdm
import pickle
from PIL import Image


# ---------------------------------------------------------------

# predefined variables

def pars_arg():
    parser = argparse.ArgumentParser(description='lAIbelNet: an automatic labeling tool using unsupervised clustering')

    parser.add_argument('--res', type=int, help='Image Resolution', default=224)
    parser.add_argument('--mode', type=int, help='0:Labeled, 1:Unlabeled', default=1)
    parser.add_argument('--data_path', type=str, help='Data Path', default='data')
    parser.add_argument('--n_images', type=int, help='Number of Images to Label', default=None)
    parser.add_argument('--ftr_ext', type=int, help='0:MobileNetV2, 1:ResNet50, 2:InceptionResNetV2', default=0)
    parser.add_argument('--min_clustr', type=int, help='Min Number of Clusters', default=2)
    parser.add_argument('--max_clustr', type=int, help='Max Number of Clusters', default=10)

    args = parser.parse_args()
    return args


def df_maker(imgs, features, labels):
    '''
    create data frame to store clustering info
    :return: data frame
    '''
    df = pd.DataFrame(df)
    df['img', 'Real_Labls', 'img_ftrs'] = imgs
    df['img_ftrs'] = features
    df['Real_Labls'] = labels
    print(df)
    # df['img_ftrs'] = [feature[i, :] for i in range(len(feature[:, 0]))]
    # df['Real_Labls'] = [labels[i].argmax() for i in range(len(feature[:, 0]))]
    df = pd.DataFrame(df)

    df.head()

    return df


def plot_():
    rcParams['figuer.figsize'] = 16, 5
    _ = plt.plot(range(2, 10), silhout, "bo-", color='blue', linewith=3, markersize=8,
                 label='Silhoutee curve')
    _ = plt.xlabel("$k$", fontsize=14, family='Arial')
    _ = plt.ylabel("Silhoutte score", fontsize=14, family='Arial')
    _ = plt.grid(which='major', color='#cccccc', linestyle='--')
    _ = plt.title('Silhoutee curve for predict optimal number of clusters',
                  family='Arial', fontsize=14)

    k = np.argmax(silhout) + 2

    _ = plt.axvline(x=k, linestyle='--', c='green', linewith=3,
                    label=f'Optimal number of clusters({k})')
    _ = plt.scatter(k, silhout[k - 2], c='red', s=400)
    _ = plt.legend(shadow=True)
    _ = plt.show()

    print(f'The optimal number of clusters is {k}')


def main():
    args = pars_arg()
    st.write('hello')
    st.title("L`ai'belNet: ")

    image_size = (args.res, args.res)

    model = cnn_model(args.ftr_ext, image_size)

    images, labels = read_images(args.data_path, image_size, args.mode, args.n_images)

    features = model.predict(images)

    print(images.shape, labels, model, features.shape)

    # df_maker(images, features, labels)

    silhout, opt_clustr = clustering(features, args.min_clustr, args.max_clustr)

    print(silhout)
    print(opt_clustr)

    plt.figure()
    plt.plot(np.arange(args.min_clustr, args.max_clustr), silhout['KMeans'], linestyle='-')
    plt.plot(np.arange(args.min_clustr, args.max_clustr), silhout['GMM'], linestyle='--')
    plt.show()



if __name__ == '__main__':
    main()
