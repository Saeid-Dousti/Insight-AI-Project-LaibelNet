# L`ai'belNet 

<p align="left">  <img src="config\logo.jpg" width="75" height="38"> </p>

An AI-powered Image Labeling Tool

<p align="left">
  <img src="config\LaibelNet.gif">
</p>

*Presentation accompanying this project can be found [here](https://docs.google.com/presentation/d/e/2PACX-1vQN673DnLNkzx0vkuFhmstOFfeqxI_0uv_7NMElC8WDfKQI1X61qqYLGZY_sQ5k3mxDe67u5fVKYIFW/pub?start=false&loop=false&delayms=3000).
**You can watch the application video demo [here](https://www.youtube.com/watch?v=TNdMeh0DcHQ).

## Introduction
Labeling images is an initial and essential step in training computer vision algorithms.  This has offered great business opportunities. Image labeling is currently conducted mostly manually, and companies are actively looking for methods to accelerate this process and make it faster, cheaper, and more profitable. 
In this project, I am proposing an unsupervised approach to label image data for computer vision based on common clustering methods.

**_L`ai'belNet_** is a Steamlit app and uses either KMeans or Gaussian Mixture Model based on user's choice to cluster imageset into groups of more similar images. User can specify the number of clusters or have the app find the optimum number of clusters. Next a few samples from each cluster is randomly selected so user can discover their labels. Finally, all the images of each cluster are labeled with the cluster's respective label discovered via sampling. 

## Data
**_L`ai'belNet_** allows user to provide path to the image directory, e.g., _"\data"_ in this repository. All the image type files contained in the path directory and its sub-directories will be imported and their paths are saved. The immediate directory where an image is found can be considered as Ground Truth label of the image by user's choice and will be used (if availabel) for the application's performance evaluation. A sample of [Intel Kaggle competition](https://www.kaggle.com/puneet6060/intel-image-classification) is included in this repository.

```
--\data
  |    |  
  |    --\class 1
  |    |  (images)
  |    --\class 2
  |    |  (images)
  |    |... 
  |    |...  
  |    |...  
  |    --\class n
  |       (images)
  -- (images)
```

The application's algorithm entails muliple steps broken into sections. The results of each section is saved in pickle files and stored  interacively in *pickldir* directory. DO NOT commit these files since they are temporary and are very large.

## Dependencies and Installation

Inclusion of Keras and Tensorflow via Anaconda env is required.

- [Anaconda](https://docs.anaconda.com/anaconda/install/) 
- [Streamlit](streamlit.io)

To run locally, clone this repository:
```
git clone https://github.com/Saeid-Dousti/Insight-AI-Project-LaibelNet.git
cd Insight-AI-Project-LaibelNet
pip install -r requirements.txt
python setup.py install
```
To run **_L`ai'belNet_** use
```
streamlit run app.py
```

While runing the application make sure to run sections in order otherwise the required pickle file to run the next section won't be 
available which raises error(s).

Docker image build(Optional):
```
docker build -t laibelnet:v1 -f Dockerfile.app .
docker run -p 80:80 laibelnet:v1
```

Docker image of **_L`ai'belNet_** application can be downloaded from [here](https://hub.docker.com/repository/registry-1.docker.io/saeiddousti86/laibelnet/tags?page=1).

## Analysis

The feature extraction of the imageset is conducted by CNN algorithms pretrained on imagenet. Three CNN architectures MobileNetV2, ResNet50, and InceptionResNetV2 are included. For faster operation, MobileNetV2 should be tried first and it is observed that it can outperform ResNet50 in the feature extraction. In general, it is expected for the application to perform better on imagesets similar to the imagenet classes. The t-SNE graph can be consulted to evaluate the overal performance of the application.  

<p align="left">
  <img src="config\t_SNE.jpg" width="600" >
</p>
