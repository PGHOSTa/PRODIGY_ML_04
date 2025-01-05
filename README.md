# PRODIGY_ML_04

### README for Hand Sign Recognition Using CNN

---

## Description

This project implements a Convolutional Neural Network (CNN) to recognize hand signs from the **Sign Language MNIST** dataset. The dataset includes grayscale images of hand gestures representing letters `A-Z` (excluding `J` and `Z`), and the model is trained to classify these gestures accurately.

---

## Features

- **Data Preprocessing**: 
  - Normalization of pixel values to the range `[0, 1]`.
  - Reshaping of input data for compatibility with TensorFlow's Conv2D layer.
  - Dynamic one-hot encoding for labels to accommodate all possible classes.

- **Model Architecture**:
  - Convolutional layers with ReLU activation.
  - MaxPooling layers to reduce spatial dimensions.
  - Dense layers for classification.
  - Dropout layer to prevent overfitting.
  - Output layer with `softmax` activation for multi-class classification.

- **Visualization**:
  - Training and validation accuracy/loss curves.
  - Sample test predictions with true labels displayed.

---

## Prerequisites

1. Python 3.6+
2. Libraries:
   - `numpy`
   - `pandas`
   - `tensorflow`
   - `matplotlib`
   - `scikit-learn`

Install dependencies using:
```bash
pip install numpy pandas tensorflow matplotlib scikit-learn
```

---

## Dataset

The dataset can be downloaded from the following link:  
[Sign Language MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

Ensure the dataset is extracted and follows this file structure:
```
Desktop/
└── prodigy/
    └── PRODIGY_ML_04/
        ├── sign_mnist_train/
        │   └── sign_mnist_train.csv
        └── sign_mnist_test/
            └── sign_mnist_test.csv
```

---

## Usage

### 1. Load Dataset
The dataset is loaded from CSV files. Labels are extracted from the `label` column, and the remaining columns represent pixel data.

### 2. Preprocessing
- The pixel values are reshaped to 28x28 grayscale images and normalized.
- Labels are one-hot encoded for multi-class classification.

### 3. Training
Run the CNN model for 10 epochs using a batch size of 64. The training set is split into 80% training and 20% validation.

### 4. Visualization
- **Training Progress**: Plots for training/validation accuracy and loss.
- **Predictions**: Visualize 10 randomly selected test samples, showing the true label and the predicted label.


## Output

1. **Test Accuracy**: Printed on the console after evaluation.
2. **Training History**: Accuracy and loss plots.
3. **Predictions**: A grid of 10 test images with true and predicted labels.

---

## Results

The model achieves high accuracy on the test set, demonstrating effective learning of hand sign gestures.

---

## Future Work

- Add data augmentation to improve model robustness.
- Experiment with deeper architectures like ResNet or MobileNet.
- Extend the dataset to include gestures for `J` and `Z`.

---


