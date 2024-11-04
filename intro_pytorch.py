import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    # Define a transform to normalize the data
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the training data
    dataset = datasets.FashionMNIST('./data', train=training, download=True, transform=transform)
    # Create a dataloader
    loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    return loader



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential (
        # Convert 2D image to 1D vector, because fully connected layers 
        # (dense layers) expect 1D vector
        nn.Flatten(),

        # Input: 28 * 28 image, Output: 128 Neurons
        # Linear Layer roles that the input is connected to the next layer
        # as fully connected layer (dense layer)
        nn.Linear(28 * 28, 128),

        # Activation Function (Rectified Linear Unit)
        # ReLU is a non-linear activation function that allows 
        # the model to learn complex patterns in the data
        nn.ReLU(),

        # Input: 128 Neurons, Output: 64 Neurons
        nn.Linear(128, 64),

        # Activation Function (ReLU)
        nn.ReLU(),

        # Input: 64 Neurons, Output: 10 Classes (Class number of FashionMNIST)
        nn.Linear(64, 10)
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
 



def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """

    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    # Loss function measures the difference between 
    # what the neural network predicts and the actual correct answer
    criterion = nn.CrossEntropyLoss()
