## USING MSE LOSS FUNCTION

# Generating a dataset
import numpy as np

# create dummy data for training
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


# Model architecture
import torch
from torch.autograd import Variable
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


# Creating a linear regression model
inputDim = 1        # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.01 
epochs = 100

model = linearRegression(inputDim, outputDim)
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()


# Initialize loss function
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)


# Training the model
for epoch in range(epochs):
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))


# Testing the model
with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)

import matplotlib.pyplot as plt
plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()



## USING LASSO LOSS FUNCTION
## LASSO LOSS FUNCTION COMBINES MSE LOSS FUNCTION AND L1 REGULARIZATION

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generating a dataset
# create dummy data for training
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32).reshape(-1, 1)
y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32).reshape(-1, 1)

# Model architecture
class linearRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

# Define the custom Lasso loss function
class LassoLoss(nn.Module):
    def __init__(self, base_loss, alpha):
        super(LassoLoss, self).__init__()
        self.base_loss = base_loss
        self.alpha = alpha  # Regularization strength

    def forward(self, y_pred, y_true, model):
        # Calculate the base loss (e.g., MSE)
        base_loss_value = self.base_loss(y_pred, y_true)
        # Calculate the L1 regularization term
        l1_regularization = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        # Combine the base loss with the L1 regularization term
        return base_loss_value + self.alpha * l1_regularization

# Creating a linear regression model
inputDim = 1  # takes variable 'x' 
outputDim = 1  # takes variable 'y'
learningRate = 0.01 
epochs = 100

model = linearRegression(inputDim, outputDim)

# For GPU
if torch.cuda.is_available():
    model.cuda()

# Initialize loss function and optimizer
base_loss = nn.MSELoss()
alpha = 0.001  # Regularization strength
criterion = LassoLoss(base_loss, alpha)
optimizer = optim.SGD(model.parameters(), lr=learningRate)

# Training the model
for epoch in range(epochs):
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, don't want to accumulate gradients
    optimizer.zero_grad()

    # Get output from the model, given the inputs
    outputs = model(inputs)

    # Get loss for the predicted output
    loss = criterion(outputs, labels, model)
    
    # Get gradients w.r.t to parameters
    loss.backward()

    # Update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

# Testing the model
with torch.no_grad():  # We don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()

# Plotting the results
plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
