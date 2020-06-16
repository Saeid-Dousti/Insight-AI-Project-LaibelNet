import os
import argparse

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import rcParams
import seaborn as sns
import numpy as np
from load_image import load_image
from image_set import Image_set
from cnn_model import cnn_model
from cluster import clustering2UNknown, clustering2known
from keras.applications import MobileNetV2
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import keras
import time
import tqdm
import pickle
from PIL import Image
import keras.backend.tensorflow_backend as tb


@st.cache(persist=True)
# ---------------------------------------------------------------

# predefined variables

def pars_arg():
    parser = argparse.ArgumentParser(description='lAIbelNet: an automatic labeling tool using unsupervised clustering')

    parser.add_argument('--res', type=int, help='Image Resolution', default=224)
    parser.add_argument('--mode', type=int, help='0:Labeled, 1:Unlabeled', default=1)
    parser.add_argument('--data_path', type=str, help='Data Path', default='data')
    parser.add_argument('--n_images', type=int, help='Number of Images to Label', default=None)
    # parser.add_argument('--ftr_ext', type=int, help='0:MobileNetV2, 1:ResNet50, 2:InceptionResNetV2', default=0)
    parser.add_argument('--min_clustr', type=int, help='Min Number of Clusters', default=3)
    parser.add_argument('--max_clustr', type=int, help='Max Number of Clusters', default=10)

    args = parser.parse_args()
    return args


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


def total_img_nums(path):
    nums = 0
    for root, _, files in os.walk(path):
        nums += len(files)
    return nums


def tsne_plot(tsne_features, labels):
    matplotlib.rc('image', cmap='jet')

    plt.figure(figsize=(8, 6), dpi=100)
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, edgecolors='none')
    # plt.title(os.path.splitext(tsne_features_path)[0])
    plt.title('t-SNE plot')
    plt.colorbar()
    plt.show()

    st.pyplot()


@st.cache(suppress_st_warning=True)
def plot_df(df):
    st.dataframe(df[['Name', 'sub-directory', 'Path']])


def main():
    tb._SYMBOLIC_SCOPE.value = True

    args = pars_arg()

    # sidebar title and logo
    st.sidebar.title("L`ai'belNet\n An AI-powered Image Labeling Tool")

    st.sidebar.image(Image.open('label.jpg').resize((240, 106)))

    path_name = st.sidebar.text_input('1) Enter imageset path (Ex. data\Labled):', args.data_path)

    img_num = st.sidebar.slider('2) Number of images to analyze:', 2,
                                total_img_nums(path_name), total_img_nums(path_name))

    if img_num == total_img_nums(path_name):
        img_num = None

    img_res = st.sidebar.slider('3) Image size to resize (224 recommended):', 30, 400, args.res)

    image_size = (img_res, img_res)

    my_imageset = Image_set(path_name, image_size, img_num)

    # display
    st.markdown('Sample images from imageset **"' + path_name + '"** :')

    st.image([Image.open(img).resize(image_size)
              for img in my_imageset.image_df['Path'].sample(n=3, random_state=1)])

    st.markdown('Imageset Information Table:')

    plot_df(my_imageset.image_df)

    img_sel_index = st.selectbox('Select an image index to display:', my_imageset.image_df.index)

    img, sub_dir = my_imageset.image_df[['Path', 'sub-directory']].iloc[img_sel_index]

    st.image(Image.open(img).resize(image_size), caption=sub_dir)

    st.markdown('Imageset label counts:')

    fg = sns.countplot(my_imageset.image_df['sub-directory'])

    fg.set(xlabel='sub-directory', ylabel='image counts')

    st.pyplot()

    grnd_trth_label = st.sidebar.checkbox('4) Are the sub-directories (in bar chart) GROUND TRUTH image labels?')

    num_clstrs_known = st.sidebar.checkbox('5) Do you want imageset to cluster to a specific'
                                           ' number of clusters (labels)? \n (if not optimum number of'
                                           ' clusters will be discovered)')

    if num_clstrs_known:
        number_clstrs = st.sidebar.text_input( '5*) Enter number of clusters (label (Ex. data\Labled):',
                                               len(set(my_imageset.image_df['Label Code']))  )


    # analysis section
    cnn_name = st.sidebar.selectbox('Select CNN Feature Extractor Model:', ['MobileNetV2', 'ResNet50',
                                                                            'InceptionResNetV2'])

    cnn_model_ = cnn_model(cnn_name, image_size)

    features = cnn_model_.predict(my_imageset.image_nparray)

    #if num_clstrs_known:



    # tsne_features = TSNE().fit_transform(features)

    # tsne_plot(tsne_features, list(my_imageset.image_df['sub-directory']))

    img_res = st.sidebar.slider('Number of clusters:', args.min_clustr, args.max_clustr, 6)

    silhout, opt_clustr, optimized_model = clustering(features, args.min_clustr, args.max_clustr)

    '''
    # images, labels = read_images(, image_size, args.mode)

    features = cnn_model_.predict(my_imageset.image_df['images'])

    print(images.shape, labels, cnn_model_, features.shape)

    # df_maker(images, features, labels)

    silhout, opt_clustr, optimized_model = clustering(features, args.min_clustr, args.max_clustr)

    print(silhout)
    print(opt_clustr)

    plt.figure()
    plt.plot(np.arange(args.min_clustr, args.max_clustr), silhout['KMeans'], linestyle='-')
    plt.plot(np.arange(args.min_clustr, args.max_clustr), silhout['GMM'], linestyle='--')
    plt.show()
    '''


if __name__ == '__main__':
    main()
