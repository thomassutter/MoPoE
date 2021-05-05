import torch
import torch.nn as nn

class ClfImgSVHN(nn.Module):
    def __init__(self):
        super(ClfImgSVHN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, dilation=1);
        self.bn1 = nn.BatchNorm2d(32);
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.bn2 = nn.BatchNorm2d(64);
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.bn3 = nn.BatchNorm2d(64);
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, dilation=1);
        self.bn4 = nn.BatchNorm2d(128);
        self.relu = nn.ReLU();
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=128, out_features=10, bias=True)  # 10 is the number of classes (=digits)
        self.sigmoid = nn.Sigmoid();

    def forward(self, x):
        h = self.conv1(x);
        h = self.dropout(h);
        h = self.bn1(h);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.dropout(h);
        h = self.bn2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.dropout(h);
        h = self.bn3(h);
        h = self.relu(h);
        h = self.conv4(h);
        h = self.dropout(h);
        h = self.bn4(h);
        h = self.relu(h);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h);
        return out;

    def get_activations(self, x):
        h = self.conv1(x);
        h = self.dropout(h);
        h = self.bn1(h);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.dropout(h);
        h = self.bn2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.dropout(h);
        h = self.bn3(h);
        h = self.relu(h);
        h = self.conv4(h);
        h = self.dropout(h);
        h = self.bn4(h);
        h = self.relu(h);
        h = h.view(h.size(0), -1);
        return h;