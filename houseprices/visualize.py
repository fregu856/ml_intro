from datasets import DatasetTrain, DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from model import LineModel

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# NOTE! change this to not overwrite all log data when you train the model:
model_id = "visualize"

num_epochs = 100
batch_size = 137

train_dataset = DatasetTrain()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=False)

model = LineModel(model_id, project_dir="/home/fredrik/ml_intro/houseprices")

k_values = np.linspace(-0.1, 0.1, 300)
m_values = np.linspace(0.0, 1.0, 300)
loss_values = np.zeros((300, 300))
for i_k in range(len(k_values)):
    for i_m in range(len(m_values)):
        for step, (x, y) in enumerate(train_loader):
            k_value = k_values[i_k]
            m_value = m_values[i_m]

            model.k.data.fill_(k_value)
            model.m.data.fill_(m_value)

            x = Variable(x).unsqueeze(1) # (shape: (batch_size, 1))
            y = Variable(y).unsqueeze(1) # (shape: (batch_size, 1))

            y_hat = model(x) # (shape: (batch_size, 1))

            loss = torch.mean(torch.pow(y - y_hat, 2))

            loss_value = loss.data.cpu().numpy()
            loss_values[i_k, i_m] = loss_value

print (loss_values)
print (loss_values.shape)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

k_values, m_values = np.meshgrid(k_values, m_values)
surf = ax.plot_surface(k_values, m_values, loss_values, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig("%s/test.png" % model.model_dir)
