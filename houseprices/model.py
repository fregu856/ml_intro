import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

import os

class LineModel(nn.Module):
    def __init__(self, model_id, project_dir):
        super(LineModel, self).__init__()

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.k = nn.Parameter(torch.Tensor([0.0]))
        self.m = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        # (x has shape (batch_size, 1))

        y = self.k*x + self.m # (shape: (batch_size, 1))

        return y

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

# model = LineModel("test", "/home/fredrik/ml_intro/houseprices")
# x = Variable(torch.ones(8, 1))
# print (x)
# print (x.size())
# y = model(x)
# print (y)
# print (y.size())
