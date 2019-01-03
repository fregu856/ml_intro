import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, dataset="train"):

        x_values = []
        y_values = []
        with open("houseprices.txt") as file:
            # line format: area  price
            for line in file:
                values = line.split()

                x = float(values[0])
                y = float(values[1])

                x_values.append(x)
                y_values.append(y)

        if dataset == "train":
            x_values = x_values[0:int(0.8*len(x_values))]
            y_values = y_values[0:int(0.8*len(y_values))]
        elif dataset == "val":
            x_values = x_values[int(0.8*len(x_values)):len(x_values)]
            y_values = y_values[int(0.8*len(y_values)):len(y_values)]
        else:
            raise Exception("dataset must be either 'train' or 'val'")

        x_values = np.array(x_values, dtype=np.float32)
        y_values = np.array(y_values, dtype=np.float32)

        y_values = y_values/1000000.0

        plt.figure(1)
        plt.plot(x_values, y_values, "k.")
        plt.ylabel("y")
        plt.xlabel("x")
        plt.xlim([0, 60])
        plt.ylim([0, 2.5])
        plt.title("training data\nx: area (m^2), y: price (MSEK)")
        plt.savefig("training_data.png")
        plt.close(1)

        self.examples = []
        for i in range(x_values.shape[0]):
            example = {}
            example["x"] = x_values[i]
            example["y"] = y_values[i]
            self.examples.append(example)

        self.num_examples = len(self.examples)

        print ("num training examples: %d" % self.num_examples)

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]
        y = example["y"]

        return (x, y)

    def __len__(self):
        return self.num_examples

#_ = DatasetTrain()

class DatasetEval(torch.utils.data.Dataset):
    def __init__(self, dataset="val"):

        x_values = []
        y_values = []
        with open("houseprices.txt") as file:
            # line format: area  price
            for line in file:
                values = line.split()

                x = float(values[0])
                y = float(values[1])

                x_values.append(x)
                y_values.append(y)

        if dataset == "train":
            x_values = x_values[0:int(0.8*len(x_values))]
            y_values = y_values[0:int(0.8*len(y_values))]
        elif dataset == "val":
            x_values = x_values[int(0.8*len(x_values)):len(x_values)]
            y_values = y_values[int(0.8*len(y_values)):len(y_values)]
        else:
            raise Exception("dataset must be either 'train' or 'val'")

        x_values = np.array(x_values, dtype=np.float32)
        y_values = np.array(y_values, dtype=np.float32)

        y_values = y_values/1000000.0

        plt.figure(1)
        plt.plot(x_values, y_values, "k.")
        plt.ylabel("y")
        plt.xlabel("x")
        plt.xlim([0, 60])
        plt.ylim([0, 2.5])
        plt.title("evaluation data\nx: area (m^2), y: price (MSEK)")
        plt.savefig("evaluation_data.png")
        plt.close(1)

        self.examples = []
        for i in range(x_values.shape[0]):
            example = {}
            example["x"] = x_values[i]
            example["y"] = y_values[i]
            self.examples.append(example)

        self.num_examples = len(self.examples)

        print ("num evaluation examples: %d" % self.num_examples)

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]
        y = example["y"]

        return (x, y)

    def __len__(self):
        return self.num_examples

#_ = DatasetEval()
