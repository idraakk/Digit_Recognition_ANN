# Handwritten Digit Recognition

This project showcases a neural network-based system for recognizing handwritten digits using the MNIST benchmark dataset.

## Table of Contents

- [Introduction](#introduction)
- [Libraries Used](#libraries-used)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Compilation](#model-compilation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Predictions and Visualization](#predictions-and-visualization)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Introduction

This project demonstrates a digit recognition system implemented using a simple neural network. The system is designed to recognize handwritten digits from the MNIST dataset, which is a standard benchmark dataset in the field of machine learning and image processing.

## Libraries Used

- **NumPy**: For numerical computations and array manipulations.
- **Matplotlib**: For data visualization, particularly for displaying the digits and the predictions.
- **TensorFlow/Keras**: For building and training the neural network model.

## Dataset

The MNIST dataset is used in this project. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is a grayscale image of size 28x28 pixels.

## Data Preprocessing

Before feeding the data into the neural network, it is necessary to preprocess it. The preprocessing steps include:

1. **Normalization**: The pixel values of the images are normalized to the range [0, 1] by dividing by 255.0.
2. **One-Hot Encoding**: The labels are converted to one-hot encoded vectors, which is a common format for classification tasks.

## Model Architecture

The neural network model is built using the Keras Sequential API. The architecture consists of:

1. **Flatten Layer**: Converts the 28x28 pixel input images into a flat vector of 784 elements.
2. **Dense Layer**: A fully connected layer with 128 neurons and ReLU activation function.
3. **Output Layer**: A fully connected layer with 10 neurons (one for each digit) and softmax activation function to output probabilities for each class.

## Model Compilation

The model is compiled using the following configurations:

- **Optimizer**: Adam, which is an adaptive learning rate optimization algorithm.
- **Loss Function**: Categorical Crossentropy, which is suitable for multi-class classification problems.
- **Metrics**: Accuracy, to evaluate the performance of the model.

## Model Training

The model is trained on the training dataset for 5 epochs with a batch size of 32. Additionally, 20% of the training data is used as a validation set to monitor the model's performance during training.

## Model Evaluation

After training, the model is evaluated on the test dataset to determine its accuracy. The test accuracy is printed to provide an indication of how well the model generalizes to unseen data.

## Predictions and Visualization

The model's predictions on the test dataset are visualized by plotting the first 25 test images along with the predicted digit labels. 
This provides a visual confirmation of the model's performance.

<img src="https://github.com/idraakk/Digit_Recognition_ANN/assets/73667258/ade26de9-6717-413d-9108-1e8208774e8c" alt="test_result" width="300"/>

## Results

The model achieved an accuracy of approximately **97.31%** on the test dataset. The visualization of the first 25 test images along with the predicted labels confirms that the model is able to correctly identify the majority of the digits.

![Test Accuracy](https://github.com/idraakk/Digit_Recognition_ANN/assets/73667258/8d22076f-6446-41b2-ac44-33291b0cab9f)


## Conclusion

This project successfully demonstrates the implementation of a neural network for digit recognition using the MNIST dataset. The simplicity of the model makes it a good starting point for beginners in machine learning, while still providing a solid understanding of key concepts such as data preprocessing, model building, training, evaluation, and visualization.

## Future Work

Possible improvements and extensions of this project include:

- Increasing the model complexity by adding more layers or neurons.
- Experimenting with different activation functions and optimizers.
- Implementing data augmentation techniques to improve model generalization.
- Exploring more advanced architectures such as Convolutional Neural Networks (CNNs) for improved performance.
