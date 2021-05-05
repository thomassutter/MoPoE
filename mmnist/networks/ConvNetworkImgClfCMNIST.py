import torch
import torch.nn as nn

from utils.utils import Flatten, Unflatten


# class ClfImg(nn.Module):
#     def __init__(self):
#         super(ClfImg, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2);
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2);
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2);
#         self.relu = nn.ReLU();
#         self.dropout = nn.Dropout(p=0.5, inplace=False);
#         self.linear = nn.Linear(in_features=128, out_features=10, bias=True)  # 10 is the number of classes (=digits)
#         self.sigmoid = nn.Sigmoid();

#     def forward(self, x):
#         h = self.conv1(x);
#         h = self.relu(h);
#         h = self.conv2(h);
#         h = self.relu(h);
#         h = self.conv3(h);
#         h = self.relu(h);
#         h = self.dropout(h);
#         h = h.view(h.size(0), -1);
#         h = self.linear(h);
#         out = self.sigmoid(h);
#         return out;

#     def get_activations(self, x):
#         h = self.conv1(x);
#         h = self.relu(h);
#         h = self.conv2(h);
#         h = self.relu(h);
#         h = self.conv3(h);
#         h = self.relu(h);
#         h = self.dropout(h);
#         h = h.view(h.size(0), -1);
#         return h;

class ClfImg(nn.Module):
    """
    MNIST image-to-digit classifier. Roughly based on the encoder from:
    https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1),     # -> (10, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),    # -> (20, 7, 7)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            Flatten(),                                                # -> (980)
            nn.Linear(980, 128),                                      # -> (128)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10)                                        # -> (10)
        )

    def forward(self, x):
        h = self.encoder(x)
        # return F.log_softmax(h, dim=-1)
        return h
