# Convolutional Neural Network for Handwritten Digit Recognition

This project demonstrates the implementation of a Convolutional Neural Network (CNN) using Keras and TensorFlow for handwritten digit recognition on the MNIST dataset.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## Introduction
Handwritten digit recognition is a fundamental task in computer vision and has numerous applications in areas such as document processing, postal automation, and form data entry. This project aims to build a CNN model that can accurately classify handwritten digits from the MNIST dataset.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- scikit-learn

## Usage
Clone the repository:
```bash
git clone https://github.com/your-username/handwritten-digit-recognition.git

Navigate to the project directory:
```bash
cd handwritten-digit-recognition

Run the script:
```bash
python digit_recognition.py

The script will train the CNN model on the MNIST dataset, evaluate its performance, and save the trained model to a file named ocr.h5.
Code Explanation


## Code Explanation

- The necessary packages are imported, including Keras layers, optimizers, and the MNIST dataset.
- The `build_model` function is defined to create the CNN architecture. It consists of two sets of convolutional, activation, and max-pooling layers, followed by two sets of fully connected, activation, and dropout layers. Finally, a softmax classifier is added.
- The MNIST dataset is loaded, and the data is reshaped and scaled to the range of .
- The labels are converted from integers to vectors using `LabelBinarizer`.
- The CNN model is compiled with the Adam optimizer and categorical cross-entropy loss.
- The model is trained using the `fit` method, with the training and validation data, batch size, and number of epochs specified.
- The trained model is evaluated on the test data, and the classification report is printed.
- The trained model is serialized to disk using `model.save`.

## Results

The trained CNN model achieves an accuracy of approximately 99% on the MNIST test set. The classification report provides detailed performance metrics for each digit class.

## Future Improvements

- Experiment with different CNN architectures and hyperparameters to further improve the model's performance.
- Implement data augmentation techniques to increase the diversity of the training data and improve generalization.
- Explore transfer learning by using pre-trained models as a starting point for fine-tuning on the MNIST dataset.

## License

This project is licensed under the MIT License.

