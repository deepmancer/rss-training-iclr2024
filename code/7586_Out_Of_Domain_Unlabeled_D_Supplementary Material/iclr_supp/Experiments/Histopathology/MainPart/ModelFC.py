import torch
import torch.nn as nn


class myFC(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=[2048, 2048], class_num=2):
        super(myFC, self).__init__()
        self.fc1 = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim[1], class_num)
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.softmax(x)
        return x