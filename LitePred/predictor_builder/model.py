# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self,input_features=None):
        super().__init__()
        self.fc1 = nn.Linear(input_features,16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64,128)
        self.fc5 = nn.Linear(128,128)
        self.fc6 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 256)
        self.fc10 = nn.Linear(256, 256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 128)
        self.fc14 = nn.Linear(128, 64)
        self.fc15 = nn.Linear(64, 16)
        self.fc16 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x))
        x = self.relu(self.fc11(x))
        x = self.relu(self.fc12(x))
        x = self.leakyrelu(self.fc13(x))
        x = self.leakyrelu(self.fc14(x))
        x = self.relu(self.fc15(x))
        out = self.fc16(x)

        return out.squeeze(-1)

