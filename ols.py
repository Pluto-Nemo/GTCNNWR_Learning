# pycharm professional 2022.2.3,Anaconda 3.8 python 3.8,torch 3.8
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
from sklearn.datasets import fetch_california_housing
import numpy as np


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


x_values = [i for i in range(99)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

inputDim = 1  # variable 'x' for Linear Regression
outputDim = 1  # variable 'y' for Linear Regression
learningRate = 0.05
epochs = 100
model = linearRegression(inputDim, outputDim)
criterion = torch.nn.MSELoss()  # set loss
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)  # Stochastic Gradient Descent

for epoch in range(epochs):
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)  # calculate loss of x and y
    loss.backward()
    optimizer.step()
    # print(inputs)
    # print(labels)
    print("epoch is {}, loss is {}".format(epoch, loss.item()))
with torch.no_grad():
    predict = model(Variable(torch.from_numpy(x_train))).data.numpy()

plt.clf()
# plt.plot(x_train, y_train, '', label='True data', alpha=0.5)
plt.plot(x_train, predict, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
