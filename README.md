# Flower Classification with PyTorch

This project implements a binary classification model to classify flower images using PyTorch. The model is built on the pre-trained ResNet-18 architecture and fine-tuned for the task of identifying whether an image contains a flower or not.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Inference](#inference)
- [Results](#results)


## Overview

This project uses transfer learning with the ResNet-18 model to classify flower images. The model is trained on a custom dataset, where the last fully connected layer is modified for binary classification. We use techniques like data augmentation, pre-trained weights, and stochastic gradient descent for optimization.

## Requirements

To install the dependencies, run the following command:

pip install torch torchvision matplotlib pillow

## Dataset Preparation

The dataset used for training and validation should be organized in the following structure:

## Training the Model

To train the model, run the train.py script.

The train.py script performs the following tasks:

Loads the dataset from the specified directory.

Defines transformations like resizing, cropping, and normalization.

Initializes a pre-trained ResNet-18 model, modifying the last layer for binary classification (flower or not flower).

Trains the model over a specified number of epochs, prints the loss, and saves the model as flower_model.pth.

## Inference
To classify a new image using the trained model, run the classify.py script.

This script:

Loads the pre-trained model from flower_model.pth.

Preprocesses the image similarly to the training pipeline.

Predicts whether the image contains a flower or not and displays the result.

## Results
After training the model, you can expect to achieve a reasonably high accuracy for the binary classification of flower images. The saved model can be used for further inference tasks.

