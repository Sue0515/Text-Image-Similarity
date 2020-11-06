import torch
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
#from data_load import get_loader
#from train import train_model
import torch.nn as nn
import argparse
import numpy as np
import pdb
from collections import OrderedDict, defaultdict
from torchvision import transforms
from torchvision.datasets import CocoCaptions #MNIST
from torch.utils.data import DataLoader
from torchvision import models
from InferSent.models import InferSent

class Embed(nn.Module):
    def __init__(self):
        super(Embed, self).__init__()
        
def init_sentemb():
    params = {'bsize': 16, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    model = InferSent(params)
    model.load_state_dict(torch.load('../milestone2/ProfessorCode/sent_emb/encoder/infersent2.pkl'))
    model = model.cuda()
    model.set_w2v_path('../milestone2/ProfessorCode/sent_emb/fastText/crawl-300d-2M.vec')
    model.build_vocab_k_words(K=100000)
    for p in model.parameters(): p.requires_grad = False
    return model

def load_resnet():
    model = models.resnet34(pretrained=True) # 34: 512, rest: 2048
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules).cuda()
    for p in model.parameters(): p.requires_grad = False
    return model # output size: bs x 512