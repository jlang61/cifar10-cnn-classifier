# CIFAR-10 CNN Classifier

This project implements a Convolutional Neural Network (CNN) in Python using PyTorch to classify images in the CIFAR-10 dataset.

## Project Overview

The `cifar_cnn.ipynb` notebook includes:

- **Introduction**: Explains the motivation behind using Convolutional Neural Networks for image classification tasks, particularly on the CIFAR-10 dataset. Discusses the challenges in classifying small images with low resolution and the importance of deep learning approaches.

- **Data Preparation**: Details the loading and preprocessing steps for the CIFAR-10 dataset. Includes normalization techniques, data augmentation strategies applied to improve model generalization, and the reasoning behind splitting the dataset into training, validation, and test sets.

- **Data Visualization**: Presents visual examples of the dataset by displaying sample images from each class. This helps in understanding the variety and complexity of the images the model will be trained on.

- **Model Definition**: Defines the architecture of the CNN used for image classification. Provides a layer-by-layer explanation, including the number of convolutional layers, kernel sizes, activation functions, pooling layers, and any dropout layers to prevent overfitting.

- **Training the Model**: Describes the training process in detail, specifying the loss function (e.g., CrossEntropyLoss), optimization algorithm (e.g., Adam optimizer), learning rate, batch size, and the number of epochs. Includes explanations of how hyperparameters were chosen and any challenges faced during training.

- **Evaluation**: Evaluates the trained model's performance on the test set. Includes accuracy metrics, confusion matrix, and loss and accuracy curves over epochs. Discusses the results, highlights any misclassified images, and suggests potential improvements for future work.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Torchvision

## Usage

1. **Clone the Repository**

Navigate to the directory where you want to clone the repository and run:

```bash
git clone https://github.com/jlang61/cifar10-cnn-classifier.git
cd cifar10-cnn-classifier
```

2. **Install Dependencies**

Navigate to the project directory and install the required packages:

```bash
pip install -r requirements.txt
```

3. **Run the Notebook**

Launch Jupyter Notebook and open `cifar_cnn.ipynb`:

```bash
jupyter notebook
```

Execute the cells in the notebook to train and evaluate the CNN model.

4. **Viewing Results**

The notebook includes visualizations of the training process and evaluation metrics. Review the plots and outputs to understand the model's performance.
