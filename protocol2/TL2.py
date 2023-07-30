#!/usr/bin/env python 
# -*- coding:utf-8 -*-
__author__ = 'Yan Yu @ Jilin University'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
from sklearn.metrics import mean_squared_error, r2_score

# loading data
file = 'superLIBStr.mat'
data = sio.loadmat(file)
X = data['NIRcha0']
Y = data['octane0']

# divide dataset
k = np.random.permutation(X.shape[0])
print(k)
X_train = X[k[:120], :]
Y_train = Y[k[:120],0:1]

X_test = X[k[611:], :]
Y_test = Y[k[611:],0:1]

# normalization
mms = preprocessing.MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

Y_train = mms.fit_transform(Y_train)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# Convert to tensor 
xtrain = torch.unsqueeze(torch.tensor(X_train).float(), dim=1)  
xtest = torch.unsqueeze(torch.tensor(X_test).float(), dim=1)
print(xtrain.shape)
ytrain = torch.tensor(Y_train).float()
ytest = torch.tensor(Y_test).float()


class DatasetXY(Dataset):
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._len = len(x)

    def __getitem__(self, item):  
        return self._x[item], self._y[item]

    def __len__(self):
        return self._len

batchsize = 5
train_loader = DataLoader(DatasetXY(xtrain, ytrain), batch_size=batchsize, shuffle=False, drop_last=True, num_workers=0)
test_loader = DataLoader(DatasetXY(xtest, ytest), batch_size=batchsize, shuffle=False, drop_last=True, num_workers=0)


# Import pre-trained models
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(  
            nn.Conv1d(1, 16, kernel_size=201, stride=100),   # output[16, 201]
            nn.ReLU(inplace=True), 
            nn.MaxPool1d(kernel_size=3, stride=1),                  # output[16, 100]
            # nn.Conv1d(16, 32, kernel_size=3, padding=2),           # output[32, 100]
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=2, stride=1),                  # output[32, 49]
            # nn.Conv1d(32, 64, kernel_size=3, padding=1),          # output[64, 49]
            # nn.ReLU(inplace=True),
            # nn.Conv1d(64, 128, kernel_size=3, padding=1),          # output[128, 49]
            # nn.ReLU(inplace=True),
            # nn.Conv1d(128, 32, kernel_size=3, padding=1),          # output[32, 49]
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=3, stride=2),                  # output[32, 24]
        )
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.5),
            #FC
            nn.Linear(16 * 146, 1),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Linear(32, 16),
            # nn.ReLU(inplace=True),
            # nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  
        x = self.regressor(x)
        return x


# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = ConvNet()
# load model weights
model_weight_path = "./ConvNet.pth"
model.load_state_dict(torch.load(model_weight_path))

# TL2
# for parma in model.parameters():
#     parma.requires_grad = False

model.regressor = nn.Sequential(
            nn.Dropout(p=0.5),
            #FC
            nn.Linear(16 * 146, 1),
            #\
                nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Linear(32, 16),
            # nn.ReLU(inplace=True),
            # nn.Linear(32, 1),
        )
print(model)


# Setting up the optimizer and loss function
#optimizer = torch.optim.Adam(model.regressor.parameters(), lr=0.01)  # 梯度下降方法
#optimizer = torch.optim.SGD(model.regressor.parameters(), lr=0.01) #  # 梯度下降方法
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_function = torch.nn.MSELoss()  

model.to(device)
#  Training
for epoch in range(50):
    with torch.no_grad():
        for step, data in enumerate(train_loader, start=0):
            x, y = data
            # feature extraction
           # features = net.features[0:1](x.to(device))
            features = model.features[0:3](xtrain.to(device))
            print(features.size())
    # train
    model.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        x, y = data
        optimizer.zero_grad()
        outputs = model(x.to(device))
        loss = loss_function(outputs, y.to(device))
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        print(len(train_loader))
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 10)
        b = "." * int((1 - rate) * 10)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter()-t1)


# Test Set Prediction
model.eval()
with torch.no_grad():
    y_sim = torch.squeeze(model(xtest.to(device)))
    print(y_sim.shape)

    y_sim = mms.inverse_transform(y_sim.cpu().numpy().reshape(30, -1))



# error analysis
error = abs(ytest.data.numpy()-y_sim) / ytest.data.numpy()

results = np.hstack((ytest.data.numpy(), y_sim, error))
print(results)


# R2
def compute_correlation(x,y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    ssr = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(0,len(x)):
        diff_xbar = x[i] - xbar
        dif_ybar = y[i] - ybar
        ssr += (diff_xbar * dif_ybar)
        var_x += diff_xbar**2
        var_y += dif_ybar**2
    sst = np.sqrt(var_x * var_y)
    return ssr/sst


R = compute_correlation(y_sim, ytest.data.numpy())
print("R2 = ", R**2)
print("Mean squared error: %.4f" % mean_squared_error(Y_test, y_sim)**0.5)
#print("Coefficient of determination: %.4f" % r2_score(Y_test, y_sim))
# plot
plt.title('Prediction Results')
plt.scatter(ytest.data.numpy(), y_sim)
plt.plot(ytest.data.numpy(), ytest.data.numpy(),'r--')
plt.show()

