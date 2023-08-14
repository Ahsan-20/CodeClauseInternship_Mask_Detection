# Mask Detection for codeClause Internship

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Importing Necessary Libraries](#importing-necessary-libraries)
- [Loading and Preprocessing the Dataset](#loading-and-preprocessing-the-dataset)
- [Building the Model](#building-the-model)
- [Compiling and Training the Model](#compiling-and-training-the-model)
- [Evaluating the Model and Saving it](#evaluating-the-model-and-saving-it)
- [Conclusion](#conclusion)

## Introduction

This project focuses on building a mask detection system that can determine whether a person is wearing a mask or not using a combination of Haar Cascade for face detection and a custom-trained TensorFlow model for mask classification. The project was developed as part of an internship at Code Clause.

## Dataset

The dataset used for training and evaluating the mask detection model is sourced from Kaggle: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset). It contains a collection of images with individuals wearing masks and without masks. This dataset serves as the foundation for training a robust mask detection model.

## Getting Started

### Prerequisites

Make sure you have the following prerequisites installed:

- Python 3.x
- TensorFlow
- Keras
- Other required libraries (list them here)

### Installation

1. Clone or download the repository to your local machine.
2. Install the required libraries by running: `pip install -r requirements.txt`
3. Preprocess the dataset using the provided data augmentation techniques (details in [Loading and Preprocessing the Dataset](#loading-and-preprocessing-the-dataset)).
4. Build, train, and evaluate the mask detection model (steps in [Building the Model](#building-the-model), [Compiling and Training the Model](#compiling-and-training-the-model), and [Evaluating the Model and Saving it](#evaluating-the-model-and-saving-it)).

## Importing Necessary Libraries

At the beginning of the code, we import essential libraries, including TensorFlow and Keras for building and training neural networks. Additional utilities for data augmentation and evaluation are also imported to ensure the model's robustness and generalization.

## Loading and Preprocessing the Dataset

The dataset is loaded using the `ImageDataGenerator` class, which includes data augmentation settings to enhance the model's ability to handle diverse images. Images are resized to a common size (224x224 pixels) and batched for efficient training. The dataset is divided into two classes: "with mask" and "without mask."

## Building the Model

We load a pre-trained MobileNetV2 model as the base model. MobileNetV2 is a popular convolutional neural network architecture that is often used as a feature extractor. The base model's top (fully connected) layers are removed since we'll replace them with our custom head layers. We construct the custom head layers that are specific to our mask detection task. These layers include average pooling, flattening, dense (fully connected) layers, and a dropout layer to prevent overfitting. The final output layer has a single neuron with a sigmoid activation function, which is suitable for binary classification tasks like mask detection. The base model's layers are frozen to prevent their weights from being updated during training. We'll fine-tune these layers later if necessary.

## Compiling and Training the Model

The model is compiled using the Adam optimizer with a low learning rate and weight decay. We use binary cross-entropy as the loss function since we have a binary classification task. The fit function is used to train the model. We provide the training generator and specify the number of steps per epoch. The number of steps is calculated as samples // batch_size, ensuring that each image is used once per epoch. The model is trained for a fixed number of epochs (in this example, 20). During each epoch, the training data is iterated through multiple times, and the model's weights are updated to minimize the loss.

## Evaluating the Model and Saving it

After training, the model's performance is evaluated on the training dataset. The model's predictions are compared to the ground truth labels, and a classification report is generated, showcasing metrics such as accuracy, precision, recall, and F1-score for each class. The trained model is saved to a file named "mask_detection_model.h5" for future use.

## Conclusion

This project presents a mask detection system that combines Haar Cascade for face detection and a custom-trained TensorFlow model for mask classification. By accurately identifying whether individuals are wearing masks...
