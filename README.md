# LAIBELNet: An Automatic Image Labeling Tool
Framework for machine learning projects at Insight Atrificial Intelegence summer of 2020.

## Motivation for this project:
Labeling new data can be a tedious and challenging task. This project aims to create a framework for discovering new labels using already labeled data within a particular taxonomy. As a concrete example, let’s say the user has a dataset of bird pictures, partially annotated with labels for eagle, parrot and swallow, however the dataset also contains pictures of hummingbirds and owls. The goal is to leverage the structure in the data annotated as eagle/parrot/swallow to discover the new concepts of hummingbird and owl in the unlabeled data.

## Data
This tool can process both labled and unlabeled image sets. Labeled images can be used as ground truth performance prediction of model. Unlabeled data are used for performiing the labeling/clustering task.
