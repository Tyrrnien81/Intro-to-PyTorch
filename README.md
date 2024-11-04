# Neural Network Project - Intro to PyTorch

This project is a Python implementation of a neural network using PyTorch, designed to train, evaluate, and make predictions on the Fashion-MNIST dataset. This README provides a detailed overview of the project, including installation steps, descriptions of each function, and instructions on how to run the project.

## Table of Contents

-   [Introduction](#introduction)
-   [Project Structure](#project-structure)
-   [Getting Started](#getting-started)
-   [Usage](#usage)
-   [Functions Overview](#functions-overview)
-   [Dataset Information](#dataset-information)
-   [Example Outputs](#example-outputs)
-   [Contributing](#contributing)
-   [License](#license)

## Introduction

This project demonstrates how to build and train a simple neural network using PyTorch to classify images from the Fashion-MNIST dataset. The goal is to help beginners understand the steps involved in building, training, evaluating, and making predictions using a neural network in PyTorch. The Fashion-MNIST dataset consists of 28x28 grayscale images of 10 different categories of clothing items, such as T-shirts, trousers, and coats.

## Project Structure

-   **`intro_pytorch.py`**: Main Python script containing all the functions to load data, build the model, train, evaluate, and predict labels.
-   **`README.md`**: Documentation file that provides an overview of the project.
-   **`/data`**: Directory to store the Fashion-MNIST dataset (automatically downloaded).

## Getting Started

To run this project locally, follow the instructions below:

### Prerequisites

-   **Python** (v3.10.12 or >= v3.8)

1. **Clone the repository**:

    ```sh
    git clone https://github.com/Tyrrnien81/Intro-to-PyTorch
    cd <repository-folder>
    ```

2. **Create a virtual environment**:

    ```sh
    python3 -m venv Pytorch
    source Pytorch/bin/activate  # For Mac and Linux
    Pytorch/Scripts/activate  # For Windows
    ```

3. **Install the dependencies**:

    ```sh
    pip install -r requirements.txt

    or

    pip install --upgrade pip
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
    pip install numpy==1.26.4
    ```

    Make sure `torch`, `torchvision`, and other required libraries are properly installed.

## Usage

After installing the necessary dependencies, you can run the script by executing:

```sh
python intro_pytorch.py
```

The script includes functions for training the model, evaluating its performance, and making predictions on test images.

You can modify the `__main__` section at the bottom of `intro_pytorch.py` to test specific parts of the functionality.

## Functions Overview

The main script (`intro_pytorch.py`) contains the following functions:

1. **`get_data_loader(training=True)`**: Loads the Fashion-MNIST dataset and returns a data loader for either training or testing, depending on the argument.

    - **Parameters**: `training` (bool) - whether to load the training set.
    - **Returns**: DataLoader for the specified dataset.
    - **Concepts Involved**: DataLoader is used to handle batch processing of the dataset, making it easier to manage and shuffle data during training. The `transforms` applied to the dataset, such as `ToTensor()` and `Normalize()`, convert image data into a format suitable for neural networks.

2. **`build_model()`**: Constructs and returns an untrained neural network model with fully connected layers.

    - **Returns**: An untrained PyTorch neural network model.
    - **Concepts Involved**: The model is a simple feedforward neural network built with `torch.nn.Module`. It consists of input, hidden, and output layers. Layers are connected using linear transformations (`nn.Linear`), and ReLU activation functions (`nn.ReLU`) are applied to introduce non-linearity.

3. **`train_model(model, train_loader, criterion, T)`**: Trains the given model on the training dataset.

    - **Parameters**:
        - `model` (PyTorch model): The neural network model to be trained.
        - `train_loader` (DataLoader): The training data loader to provide batches of images and labels.
        - `criterion` (loss function): The function used to calculate the error (e.g., CrossEntropyLoss).
        - `T` (int): Number of epochs to train the model.
    - **Returns**: None.
    - **Concepts Involved**: The training loop involves several steps:
        - **Forward Pass**: The model makes predictions on the input data.
        - **Loss Calculation**: The error between predicted labels and actual labels is calculated using the `criterion` (typically Cross-Entropy loss for classification tasks).
        - **Backpropagation**: Using `loss.backward()`, gradients are computed for each parameter.
        - **Optimizer Step**: The optimizer (`optim.SGD` or similar) updates the weights to reduce the error. This process is repeated for `T` epochs.

4. **`evaluate_model(model, test_loader, criterion, show_loss=True)`**: Evaluates the model using the test dataset and prints accuracy and loss.

    - **Parameters**:
        - `model` (PyTorch model): The trained model to evaluate.
        - `test_loader` (DataLoader): The test data loader to provide batches of images and labels.
        - `criterion` (loss function): The function used to calculate the loss on the test set.
        - `show_loss` (bool, optional): Whether to print the loss value.
    - **Returns**: None.
    - **Concepts Involved**: Evaluation involves putting the model in evaluation mode (`model.eval()`), disabling gradient calculations (`torch.no_grad()`), and looping through the test dataset to compute predictions. Accuracy is calculated by comparing predicted labels with the true labels, and loss is computed similarly to training but without updating the weights.

5. **`predict_label(model, test_images, index)`**: Predicts the label for a specific image in the test set.
    - **Parameters**:
        - `model` (PyTorch model): The trained model used for prediction.
        - `test_images` (tensor): The set of test images.
        - `index` (int): The index of the image to predict.
    - **Returns**: None. Prints the top 3 class predictions with probabilities.
    - **Concepts Involved**: The function sets the model to evaluation mode and uses `torch.no_grad()` to disable gradient computation. The logits for the specified image are obtained, and `Softmax` is applied to convert these logits to probabilities. Then `torch.topk()` is used to find the top 3 predicted classes, which are then printed along with their probabilities. This helps to understand the model's confidence and uncertainty for each prediction.

## Dataset Information

The **FashionMNIST** dataset is a collection of grayscale images, each of size 28x28 pixels, representing 10 categories of clothing items:

-   T-shirt/top
-   Trouser
-   Pullover
-   Dress
-   Coat
-   Sandal
-   Shirt
-   Sneaker
-   Bag
-   Ankle Boot

The dataset is automatically downloaded by the `torchvision.datasets.FashionMNIST` class, and is divided into training and test sets. The dataset is often used as a beginner-friendly introduction to image classification tasks, and provides a more complex challenge than the classic MNIST digits dataset.

## Example Outputs

The following is an example of the output generated by `predict_label()` function:

```
Pullover: 87.62%
Shirt: 11.26%
Coat: 1.02%
```

This output shows the top 3 predictions with their respective probabilities for a given image in the test dataset. This type of output allows you to see which classes the model is most confident about and helps assess the quality of the model.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any bugs, feel free to open an issue or create a pull request.

1. **Fork the repository**.
2. **Create a new branch** (`git checkout -b feature-branch`).
3. **Commit your changes** (`git commit -m 'Add some feature'`).
4. **Push to the branch** (`git push origin feature-branch`).
5. **Open a Pull Request**.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

Feel free to customize the details to fit your needs or include additional information if necessary.
