import torch
import torch.nn as nn

class CNNController(nn.Module):

    def __init__(self, d):
        super(CNNController, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 128, 5,dilation=2),nn.ReLU(),nn.Conv2d(128, 128, 5,dilation=2),nn.ReLU())
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.block2 = nn.Sequential(nn.Conv2d(128, 128, 3),nn.ReLU(),nn.Conv2d(128, 128, 3),nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear(512, d)

    def forward(self, inputs):
        x = self.block1(inputs)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = torch.flatten(x,start_dim=1)
        return self.fc(x)
