# Mask Detection using Haar Cascade and TensorFlow


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
- [Further Enhancements](#further-enhancements)

## Introduction

This project focuses on building a mask detection system that can determine whether a person is wearing a mask or not using a combination of Haar Cascade for face detection and a custom-trained TensorFlow model for mask classification. The goal is to contribute to public health and safety measures by automating the process of mask detection in images.

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

The model architecture is based on a pre-trained MobileNetV2 model, which acts as a feature extractor. The custom head layers are constructed to suit the mask detection task, including average pooling, flattening, dense layers, and dropout to prevent overfitting. The final output layer employs a sigmoid activation function for binary classification.

## Compiling and Training the Model

The model is compiled using the Adam optimizer with a low learning rate and weight decay. Binary cross-entropy is chosen as the loss function for the binary classification task. The training process involves iterating through the training data for a fixed number of epochs while updating the model's weights to minimize the loss.

## Evaluating the Model and Saving it

After training, the model's performance is evaluated on the training dataset. The model's predictions are compared to the ground truth labels, and a classification report is generated, showcasing metrics such as accuracy, precision, recall, and F1-score for each class. The trained model is saved to a file named "mask_detection_model.h5" for future use.

## Conclusion

This project presents a mask detection system that combines Haar Cascade for face detection and a custom-trained TensorFlow model for mask classification. By accurately identifying whether individuals are wearing masks, the system can contribute to public health measures and safety protocols.

## Further Enhancements

While the provided code offers a solid foundation for mask detection, further enhancements can be considered:

- Fine-tuning the model: Experiment with unfreezing some layers of the base model for fine-tuning to improve performance.
- Hyperparameter tuning: Optimize learning rates, batch sizes, and other hyperparameters for better results.
- Real-time detection: Implement real-time mask detection using the trained model on video streams or webcam feeds.
- Deploy the model: Convert the trained model to a deployable format (such as TensorFlow Lite) for easy integration into applications or devices.

Remember that this readme provides a comprehensive overview of the project, but additional documentation and code comments are essential for others to understand and contribute effectively to the project.

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

