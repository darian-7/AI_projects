# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size, h=128):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        
        super(NeuralNet, self).__init__()
        
        # CNN and pooling
        self.func = nn.Sequential(
            nn.Conv2d(3, 31, kernel_size=3, padding=1),     # Takes in 3 channels cuz RGB images
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(31, 62, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Activation func
        self.activator = nn.LeakyReLU()
        self.loss_fn = nn.CrossEntropyLoss()

        self.flat_size = 62*7*7     # Flatten the image before feeding into linear (hidden) layers. dimensions reduced to 7x7 after CNN layers
        self.lin1 = nn.Linear(self.flat_size, h)
        self.lin2 = nn.Linear(h, out_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lrate, weight_decay=0.0)

        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        # Kernel Size: Affects the receptive field and focus on local/global features.
        # Stride: Impacts the reduction of spatial dimensions and the field of view.
        # Padding: Determines how border information is handled and controls output size.
        # Pool size: influences how much the spatial dimensions of the feature maps are reduced.
        

    def forward(self, x):
        
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """

        if x.dim() == 2:        # Condition that manages cases that are not in the 4D format required
            x = x.view(-1, 3, 31, 31)  
        
        x = self.func(x)
        x = x.view(x.size(0), -1)       # Flatten before feeding into linear (hidden) layers
        x = self.activator(self.lin1(x))
        x = self.lin2(x)
        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        self.optimizer.zero_grad()      # Reset gradient collection before backward pass
        output = self.forward(x)
        loss_func = self.loss_fn(output, y)
        loss_func.backward()        # Ensures the gradient update performs only through one batch of data
        self.optimizer.step()       # Single optim step

        return loss_func.detach().cpu().item()     # Convert to a plain number (scalar val)



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """

#    lrate = 0.003
    in_size = 31 * 31 * 3       # RBG image with 31x31 pixel dimensions
    out_size = 4                # 4 classes
    loss_fn = nn.CrossEntropyLoss()  # multi-class classification.
    net = NeuralNet(lrate=0.0001, loss_fn=loss_fn, in_size=in_size, out_size=out_size)

    # Standardize the training and dev data with mean and std
    mean = train_set.mean()
    std = train_set.std()
    train_set = (train_set - mean) / std
    dev_set = (dev_set - mean) / std  

    train_dataset = get_dataset_from_arrays(train_set, train_labels)  # Convert to torch dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # DataLoader for batching

    losses = []  # Store loss after each epoch
    for epoch in range(epochs):
        overall_loss = 0.0
        for batch in train_dataloader:
            x = batch['features']
            y = batch['labels']
            loss = net.step(x, y)
            overall_loss += loss
        losses.append(overall_loss)  # Append the total loss after each epoch.

    # Evaluate on dev_set
    yhats = []
    with torch.no_grad():
        for x in DataLoader(dev_set, batch_size=batch_size):
            x = x.view(x.size(0), -1)
            yhats.extend(torch.argmax(net(x), dim=1).cpu().numpy().astype(np.int64))
    yhats = np.array(yhats)

    return losses, yhats, net
