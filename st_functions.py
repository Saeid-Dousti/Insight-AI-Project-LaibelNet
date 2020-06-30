import os
import argparse
from random import sample
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
import pickle
from PIL import Image
from LaibelNet.image_set import imageset_dataframe
from LaibelNet.feature_extraction import feature_extraction
from LaibelNet.cluster import imageset_cluster


# ---------------------------------------------------------------

# default variables

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


def total_img_nums(path):
    nums = 0
    for root, _, files in os.walk(path):
        nums += len(files)
    return nums


def tsne_plot(tsne_features, labels):
    matplotlib.rc('image', cmap='jet')

    plt.figure(figsize=(8, 6), dpi=100)
    sns.scatterplot(x='t-SNE one', y='t-SNE two', hue=labels, data=tsne_features,
                    palette=sns.color_palette("hls", len(list(set(labels)))), alpha=1, s=55)

    st.pyplot()


def silhouette_plot(cluster):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(cluster.min_clustr, cluster.max_clustr), cluster.kmns_silhout_range, linestyle='-')
    plt.plot(np.arange(cluster.min_clustr, cluster.max_clustr), cluster.gmm_silhout_range, linestyle='--')
    plt.legend(shadow=True, labels=['KMeans', 'GMM'])
    k = cluster.kmns_num_clstrs
    plt.axvline(x=k, linestyle='--', c='green', label=f'Optimal number of clusters({k})')
    plt.scatter(k, cluster.kmns_silhout, c='red', s=200)
    k = cluster.gmm_num_clstrs
    plt.axvline(x=k, linestyle='--', c='green', label=f'Optimal number of clusters({k})')
    plt.scatter(k, cluster.gmm_silhout, c='red', s=200)
    plt.xlabel("$Num. of Clusters$", fontsize=14)
    plt.ylabel("$ave.~Silhoutte score$", fontsize=14, family='Arial')
    plt.grid(True)

    st.pyplot()


def introduction():
    st.markdown(open('README.md').read())


def section_zero():
    # sidebar title and logo
    st.sidebar.title("L`ai'belNet\n _An AI-powered Image Labeling Tool_")
    try:
        st.sidebar.image(Image.open(os.path.join('config','logo.jpg')).resize((240, 106)))
    except:
        pass


def section_one(args):
    st.subheader('Load Imageset')

    if not os.path.exists('pickledir'):
        os.makedirs('pickledir')

    if os.path.exists(os.path.join('pickledir', 'image_path.pickle')):
        with open(os.path.join('pickledir', 'image_path.pickle'), 'rb') as f:
            tmp = pickle.load(f)
        path_name = st.text_input('Enter imageset path (Ex. data/Labled):', tmp)
    else:
        path_name = st.text_input('Enter imageset path (Ex. data/Labled):', args.data_path)

    with open(os.path.join('pickledir', 'image_path.pickle'), 'wb') as f:
        pickle.dump(path_name, f)

    img_num = st.slider('Number of images to analyze:', 2,
                        total_img_nums(path_name), total_img_nums(path_name))

    img_res = st.slider('Image size to resize (224 recommended):', 30, 400, args.res)

    image_size = (img_res, img_res)

    Load_imageset_button = st.button('Load Imageset', key=None)

    if Load_imageset_button:

        if os.path.exists('pickledir/label_dict.pickle'):
            os.remove('pickledir/label_dict.pickle')

        st.markdown('Imageset summary table:')

        imageset_df = imageset_dataframe(path_name, image_size, img_num)

        # save dataframe
        imageset_df.to_pickle(os.path.join('pickledir', 'imageset_df.pickle'))

        with open(os.path.join('pickledir', 'args.pickle'), 'wb') as f:
            pickle.dump({'image_size': image_size, 'img_num': img_num}, f)

        st.dataframe(imageset_df)


def section_two():
    st.subheader('Imageset Visualization')
    # loading
    imageset_df = pd.read_pickle(os.path.join('pickledir', 'imageset_df.pickle'))

    st.markdown('Imageset summary table:')
    st.dataframe(imageset_df)

    st.markdown('Imageset samples:')
    for i in range(3):
        st.image([Image.open(img).resize((150, 150))
                  for img in sample(list(imageset_df['Path']), 3)])

    st.markdown('Imageset Information Table:')

    st.markdown('Imageset (sub-)directory count bar chart:')

    fg = sns.countplot(imageset_df['Sub-directory'])

    fg.set(xlabel='sub-directory', ylabel='image counts')

    st.pyplot()

    gt = st.checkbox('Are the sub-directories Ground Truth Labels?')

    with open(os.path.join('pickledir', 'ground_truth_labels.pickle'), 'wb') as f:
        pickle.dump(list(imageset_df['Sub-directory'].unique()), f)

    img_sel = st.selectbox('Select an image name to display:', list(imageset_df['Image']))

    img_path, cap = imageset_df[['Path', 'Sub-directory']][imageset_df['Image'] == img_sel].iloc[0]

    st.image(Image.open(img_path).resize((150, 150)), caption=cap)


def section_three():
    st.subheader('Cluster Imageset')

    imageset_df = pd.read_pickle(os.path.join('pickledir', 'imageset_df.pickle'))

    with open(os.path.join('pickledir', 'ground_truth_labels.pickle'), 'rb') as f:
        known_gt_labels = pickle.load(f)

    with open(os.path.join('pickledir', 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    image_size = args['image_size']
    img_num = args['img_num']

    num_clstrs_known = st.markdown('If a desired number of clusters is not known a priori'
                                   ' an optimum number of clusters will be discovered automatically')

    if st.checkbox('known number of clusters'):
        num_clstrs = int(st.text_input('number of clusters', str(len(known_gt_labels))))
        min_num_clstrs, max_num_clstrs = None, None
    else:
        num_clstrs = None
        st.markdown('Range of min and Max cluster numbers to search optimum number of clusters:'
                    '\n (a wide search range can be computationally expensive and time consuming)')
        min_num_clstrs = int(st.text_input('min number of clusters', '4'))
        max_num_clstrs = int(st.text_input('max number of clusters', '7'))

    # analysis section
    cnn_name = st.selectbox('Select CNN Feature Extractor Model:', ['MobileNetV2', 'ResNet50',
                                                                    'InceptionResNetV2'])

    cluster_button = st.button('Run Clustering', key=None)

    if cluster_button:
        features = feature_extraction(cnn_name, image_size, np.array(list(imageset_df['Image_np'])))

        my_cluster = imageset_cluster(features, num_clstrs, min_num_clstrs, max_num_clstrs)

        with open(os.path.join('pickledir', 'cluster_class.pickle'), 'wb') as f:
            pickle.dump(my_cluster, f)


def section_four():
    st.subheader('Cluster Visualization and Imageset Labeling')

    # load data
    with open(os.path.join('pickledir', 'cluster_class.pickle'), 'rb') as f:
        my_cluster = pickle.load(f)
    imageset_df = pd.read_pickle(os.path.join('pickledir', 'imageset_df.pickle'))

    if not os.path.exists(os.path.join('pickledir', 'label_dict.pickle')):
        label_dict = dict()
    else:
        with open(os.path.join('pickledir', 'label_dict.pickle'), 'rb') as f:
            label_dict = pickle.load(f)

    st.markdown(f'Number of clusters based on **KMeans** method: {my_cluster.kmns_num_clstrs}')
    st.markdown(f'Number of clusters based on **GMM** method: {my_cluster.gmm_num_clstrs}')

    labeled_cluster_df = imageset_df[['Image', 'Path']]
    labeled_cluster_df['KMean_Clusters'] = my_cluster.kmns_clstrs
    labeled_cluster_df['GMM_Clusters'] = my_cluster.gmm_clstrs

    cluster_method = st.selectbox('Select Clustering Method to Label Imageset:', ['KMeans', 'Gaussian Mixture Model'])

    if cluster_method == 'KMeans':
        labeled_cluster_df['Cluster'] = labeled_cluster_df['KMean_Clusters']
    elif cluster_method == 'Gaussian Mixture Model':
        labeled_cluster_df['Cluster'] = labeled_cluster_df['GMM_Clusters']

    labeled_cluster_df['Label'] = labeled_cluster_df['Cluster']

    num_sample_cluster = st.slider('Number of images from each cluster to display:', 1, 60, 3)

    cluster_choice = st.selectbox('cluster to visualize:', list(set(labeled_cluster_df['Cluster'])))

    cluster_img_path = list(labeled_cluster_df[labeled_cluster_df['Cluster'] == cluster_choice]['Path'])

    st.image([Image.open(img).resize((150, 150))
              for img in sample(cluster_img_path, num_sample_cluster)])

    try:
        label = st.text_input(f'You may label cluster **{cluster_choice}** as:', label_dict[cluster_choice])
    except:
        label = st.text_input(f'You may label cluster **{cluster_choice}** as:', None)

    if label != 'None':
        label_dict[cluster_choice] = label
        with open(os.path.join('pickledir', 'label_dict.pickle'), 'wb') as f:
            pickle.dump(label_dict, f)

    st.write(label_dict)

    label_button = st.button('label Imageset')

    if label_button:
        for key, label_name in label_dict.items():
            labeled_cluster_df['Label'][labeled_cluster_df['Cluster'] == key] = label_name

        st.markdown(f'Labeled based on {cluster_method}:')
        st.dataframe(labeled_cluster_df[['Image', 'Path', 'Cluster', 'Label']])

        st.markdown(f'Imageset clusters based on both approaches:')
        st.dataframe(labeled_cluster_df)

        with open(os.path.join('pickledir', 'labeled_cluster_df.pickle'), 'wb') as f:
            pickle.dump(labeled_cluster_df, f)

        with open(os.path.join('pickledir', 'cluster_method.pickle'), 'wb') as f:
            pickle.dump(cluster_method, f)


def section_five():
    st.subheader('Cluster Visualization and Imageset Labeling')

    # loading
    with open(os.path.join('pickledir', 'cluster_method.pickle'), 'rb') as f:
        cluster_method = pickle.load(f)
    with open(os.path.join('pickledir', 'cluster_class.pickle'), 'rb') as f:
        my_cluster = pickle.load(f)
    with open(os.path.join('pickledir', 'labeled_cluster_df.pickle'), 'rb') as f:
        labeled_cluster_df = pickle.load(f)
    imageset_df = pd.read_pickle(os.path.join('pickledir', 'imageset_df.pickle'))

    st.markdown(f'Labeled based on {cluster_method}:')
    st.write(labeled_cluster_df)

    features = my_cluster.features

    gt_checkbox = st.checkbox('Are the sub-directories(see "Visualize Imageset" sec.) Ground Truth Labels?')
    # loading
    with open(os.path.join('pickledir', 'ground_truth_labels.pickle'), 'rb') as f:
        known_gt_labels = pickle.load(f)

    features_embedded = TSNE(n_components=2, random_state=1).fit_transform(features)

    if my_cluster.kmns_silhout_range:
        silhouette_plot(my_cluster)

    if gt_checkbox:
        comp_label_df = pd.DataFrame()
        comp_label_df['Image'] = imageset_df['Image']
        comp_label_df['Ground Truth Label'] = imageset_df['Sub-directory']
        comp_label_df['Discovered Label'] = labeled_cluster_df['Label']

        st.markdown("L`ai'belNet labels vs. Ground Truth labels:")
        st.write(comp_label_df)

        st.markdown('Clustering quality measures compared to Ground Truth labels:')

        st.markdown('**- H (homogeneity)**: _A clustering result satisfies homogeneity if all of'
                    ' its clusters contain only data points which are members of a single class._')
        st.markdown('**- C (completeness)**: _A clustering result satisfies completeness if all the data points that '
                    'are members of a given class are elements of the same cluster._')
        st.markdown('**- V**: v_measure score is the harmonic mean between homogeneity and completeness')
        st.latex(r'''\frac{1}{V} = \frac{1}{2}\left(\frac{1}{C} + \frac{1}{H}\right)''')

        measures_df = st.write(pd.DataFrame([[homogeneity_score(comp_label_df['Ground Truth Label'],
                                                                labeled_cluster_df['KMean_Clusters']),
                                              completeness_score(comp_label_df['Ground Truth Label'],
                                                                 labeled_cluster_df['KMean_Clusters']),
                                              v_measure_score(comp_label_df['Ground Truth Label'],
                                                              labeled_cluster_df['KMean_Clusters'])],
                                             [homogeneity_score(comp_label_df['Ground Truth Label'],
                                                                labeled_cluster_df['GMM_Clusters']),
                                              completeness_score(comp_label_df['Ground Truth Label'],
                                                                 labeled_cluster_df['GMM_Clusters']),
                                              v_measure_score(comp_label_df['Ground Truth Label'],
                                                              labeled_cluster_df['GMM_Clusters'])]],
                                            columns=['Homogeneity', 'Completeness', 'V_measure'],
                                            index=['KMeans', 'GMM']))

    st.markdown('t-SNE plot based on discovered labels:')
    tsne_plot(pd.DataFrame(features_embedded, columns=['t-SNE one', 't-SNE two']), labeled_cluster_df['Label'])

    if gt_checkbox:
        st.markdown('t-SNE plot based on Ground Truth labels:')
        tsne_plot(pd.DataFrame(features_embedded, columns=['t-SNE one', 't-SNE two']),
                  imageset_df['Sub-directory'])


def main():
    tb._SYMBOLIC_SCOPE.value = True

    section_zero()
    args = pars_arg()

    introduction_button = st.sidebar.checkbox('1) Introduction', key=None)

    if introduction_button:
        introduction()

    load_select = st.sidebar.checkbox('2) Load Imageset', key=None)

    if load_select:
        section_one(args)

    vis_select = st.sidebar.checkbox('3) Visualize Imageset', key=None)

    if vis_select:
        section_two()

    cluster_select = st.sidebar.checkbox('4) Cluster Imageset', key=None)

    if cluster_select:
        section_three()

    vis_cluster_select = st.sidebar.checkbox('5) Vis. Clusters & Label Imageset', key=None)

    if vis_cluster_select:
        section_four()

    performance_select = st.sidebar.checkbox('6) Clustering Performance Analytics', key=None)

    if performance_select:
        section_five()

    st.sidebar.markdown('**_For optimum performance keep '
                        'only one section active/visible at a time_**')


if __name__ == '__main__':
    main()
