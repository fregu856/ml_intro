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
model_id = "1"

num_epochs = 100
batch_size = 16
learning_rate = 0.001

train_dataset = DatasetTrain()
val_dataset = DatasetEval()

num_train_batches = int(len(train_dataset)/batch_size)
print ("num_train_batches: %d" % num_train_batches)

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches: %d" % num_val_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size, shuffle=False)

model = LineModel(model_id, project_dir="/home/fredrik/ml_intro/houseprices")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch_losses_train = []
epoch_losses_val = []
k_values = []
m_values = []
for epoch in range(num_epochs):
    print ("###########################")
    print ("######## NEW EPOCH ########")
    print ("###########################")
    print ("epoch: %d/%d" % (epoch+1, num_epochs))

    model.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (x, y) in enumerate(train_loader):
        x = Variable(x).unsqueeze(1) # (shape: (batch_size, 1))
        y = Variable(y).unsqueeze(1) # (shape: (batch_size, 1))

        y_hat = model(x) # (shape: (batch_size, 1))

        ########################################################################
        # compute the loss:
        ########################################################################
        loss = torch.mean(torch.pow(y - y_hat, 2))

        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        ########################################################################
        # optimization step:
        ########################################################################
        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % model.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print ("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % model.model_dir)
    plt.close(1)

    k_value = model.k.data.numpy()[0]
    k_values.append(k_value)
    with open("%s/k_values.pkl" % model.model_dir, "wb") as file:
        pickle.dump(k_values, file)
    print ("k value: %g" % k_value)
    plt.figure(1)
    plt.plot(k_values, "k^")
    plt.plot(k_values, "k")
    plt.ylabel("k")
    plt.xlabel("epoch")
    plt.title("k value per epoch")
    plt.savefig("%s/k_values.png" % model.model_dir)
    plt.close(1)

    m_value = model.m.data.numpy()[0]
    m_values.append(m_value)
    with open("%s/m_values.pkl" % model.model_dir, "wb") as file:
        pickle.dump(m_values, file)
    print ("m value: %g" % m_value)
    plt.figure(1)
    plt.plot(m_values, "k^")
    plt.plot(m_values, "k")
    plt.ylabel("m")
    plt.xlabel("epoch")
    plt.title("m value per epoch")
    plt.savefig("%s/m_values.png" % model.model_dir)
    plt.close(1)

    ############################################################################
    # val:
    ############################################################################
    model.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (x, y) in enumerate(val_loader):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            x = Variable(x).unsqueeze(1) # (shape: (batch_size, 1))
            y = Variable(y).unsqueeze(1) # (shape: (batch_size, 1))

            y_hat = model(x) # (shape: (batch_size, 1))

            ####################################################################
            # compute the loss:
            ####################################################################
            loss = torch.mean(torch.pow(y - y_hat, 2))

            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % model.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print ("val loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_val, "k^")
    plt.plot(epoch_losses_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val loss per epoch")
    plt.savefig("%s/epoch_losses_val.png" % model.model_dir)
    plt.close(1)

    # save the model weights to disk:
    checkpoint_path = model.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    torch.save(model.state_dict(), checkpoint_path)
