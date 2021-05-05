import torch
import torch.nn as nn


class ClfImg(nn.Module):
    def __init__(self):
        super(ClfImg, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2);
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2);
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2);
        self.relu = nn.ReLU();
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=128, out_features=10, bias=True)  # 10 is the number of classes (=digits)
        self.sigmoid = nn.Sigmoid();

    def forward(self, x):
        h = self.conv1(x);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.relu(h);
        h = self.dropout(h);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h);
        return out;

    def get_activations(self, x):
        h = self.conv1(x);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.relu(h);
        h = self.dropout(h);
        h = h.view(h.size(0), -1);
        return h;