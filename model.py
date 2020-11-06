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
    # def __init__(self, img_input_dim=4096, img_output_dim=2048,
    #              text_input_dim=1024, text_output_dim=2048, minus_one_dim=1024, output_dim=10):
    #     super(CC_NN, self).__init__()
    #     self.img_net = IMG_NN(img_input_dim, img_output_dim)
    #     self.text_net = TEXT_NN(text_input_dim, text_output_dim)
    #     self.linearLayer = nn.Linear(img_output_dim, minus_one_dim)
    #     self.linearLayer2 = nn.Linear(minus_one_dim, output_dim)

    # def forward(self, img, text):
    #     feature1 = self.img_net(img)
    #     feature2 = self.text_net(text)
    #     feature1 = self.linearLayer(feature1)
    #     feature2 = self.linearLayer(feature2)
    #     predict1 = self.linearLayer2(feature1)
    #     predict2 = self.linearLayer2(feature2)
    #     return feature1, feature2, predict1, predict2
     def __init__(self, img_enc_size, sent_enc_size, img_layer_sizes=[250, 200], \
        sent_layer_sizes=[2000, 500, 200], dropout=0.2):

        super().__init__()
        img_layer_sizes = [img_enc_size] + img_layer_sizes
        sent_layer_sizes = [sent_enc_size] + sent_layer_sizes
        self.img_MLP = nn.Sequential()
        self.sent_MLP = nn.Sequential()
        self.prob = nn.Sigmoid() 

        for i, (in_size, out_size) in enumerate( zip(img_layer_sizes[:-1], img_layer_sizes[1:]) ):
            self.img_MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            self.img_MLP.add_module(name="D%i"%(i), module=nn.Dropout(dropout)) # added
            self.img_MLP.add_module(name="A%i"%(i), module=nn.Tanh())

        for i, (in_size, out_size) in enumerate( zip(sent_layer_sizes[:-1], sent_layer_sizes[1:]) ):
            self.sent_MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            self.sent_MLP.add_module(name="D%i"%(i), module=nn.Dropout(dropout)) # added
            self.sent_MLP.add_module(name="A%i"%(i), module=nn.Tanh())
        
    # (neg_)img: bs x id, sent: bs x sd
    def forward(self, img, sent, neg_img=None):
        img_feat = self.img_MLP(img) # bs x dim
        sent_feat = self.sent_MLP(sent) # bs x dim
        dots = (img_feat * sent_feat).sum(dim=1) # bs
        probs = self.prob(dots) # bs
        if neg_img is not None:
            neg_img_feat = self.img_MLP(neg_img)
            neg_dots = (neg_img_feat * sent_feat).sum(dim=1) # bs
            neg_probs = self.prob(neg_dots) # bs
        else:
            neg_probs = torch.zeros(probs.shape)

        return probs, neg_probs

