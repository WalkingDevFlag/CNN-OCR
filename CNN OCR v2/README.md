# Digit Recognizer with CNN

This project involves building a Convolutional Neural Network (CNN) to recognize handwritten digits. The dataset used is the Chars74k dataset. The model is trained using TensorFlow/Keras.

## Table of Contents
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Saving](#model-saving)
- [Usage](#usage)
- [License](#license)

## Dataset
The dataset used for this project is the Chars74k dataset. The dataset contains images of handwritten digits (0-9), each stored in a separate folder.

- Dataset path: `/content/unzipped_data/Data/` (for Colab) or `path_to_your_local_data/Data/` (for local setup)

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV
- scikit-learn

Install the required packages using:
```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn
```
Data Preparation
The data preparation involves loading the images, preprocessing them, and splitting them into training, validation, and test sets.

## Preprocessing Steps
- Convert images to grayscale.
- Apply histogram equalization.
- Apply binary thresholding.
- Normalize pixel values to the range [0, 1].

## Model Training
A Convolutional Neural Network (CNN) with the following architecture is used for training:

- Conv2D layer with 32 filters, kernel size (3,3), ReLU activation
- MaxPooling2D layer with pool size (2,2)
- Dropout layer with rate 0.5
- Conv2D layer with 64 filters, kernel size (3,3), ReLU activation
- MaxPooling2D layer with pool size (2,2)
- Dropout layer with rate 0.5
- Conv2D layer with 64 filters, kernel size (3,3), ReLU activation
- Flatten layer
- Dense layer with 128 units, ReLU activation
- Dropout layer with rate 0.5
- Dense layer with 10 units, softmax activation
- Early Stopping
- Training stops early if the validation accuracy reaches 99.5%.

## Model Evaluation
The model's performance is evaluated using the test dataset. Metrics such as loss and accuracy are recorded.
