import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# NN for Image modality
class IMG_NN(nn.Module):
    def __init__(self, input_dim=4096, output_dim=1024):
        super(IMG_NN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out

# NN for Text modality
class TEXT_NN(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024):
        super(TEXT_NN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out

class CC_NN(nn.Module):
    def __init__(self, img_input_dim=4096, img_output_dim=2048,
                 text_input_dim=1024, text_output_dim=2048, minus_one_dim=1024, output_dim=10):
        super(CC_NN, self).__init__()
        self.img_net = IMG_NN(img_input_dim, img_output_dim)
        self.text_net = TEXT_NN(text_input_dim, text_output_dim)
        self.linearLayer = nn.Linear(img_output_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(minus_one_dim, output_dim)

    def forward(self, img, text):
        feature1 = self.img_net(img)
        feature2 = self.text_net(text)
        feature1 = self.linearLayer(feature1)
        feature2 = self.linearLayer(feature2)
        predict1 = self.linearLayer2(feature1)
        predict2 = self.linearLayer2(feature2)
        return feature1, feature2, predict1, predict2

