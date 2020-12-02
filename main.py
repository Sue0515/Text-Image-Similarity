import torch
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from model import CC_NN
from embed import load_sentemb
from embed import load_resnet
from train import train_model
import torch.nn as nn
import argparse
import numpy as np
import pdb
from collections import OrderedDict, defaultdict
from torchvision import transforms
from torchvision.datasets import CocoCaptions #MNIST
from torch.utils.data import DataLoader
from torchvision import models

def main(args):
    ts = time.time()
    datasets = OrderedDict()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MAX_EPOCH = 100
    k = 5
    common_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalization 
    ])
    datasets['train'] = CocoCaptions(root='../train2014', 
        annFile='../annotations/captions_train2014.json', transform=common_trans)
    datasets['val'] = CocoCaptions(root='../val2014/val2014', 
        annFile='../annotations/captions_val2014.json', transform=common_trans)
    #datasets['test'] = CocoCaptions(root='../test2014', 
     #    annFile='../annotations/captions_test2014.json', transform=common_trans)

    print('Loading sentence encoder...')
    stcencoder = load_sentemb()

    print('Loading ResNet...')
    resnet = load_resnet()

    img_size = 512
    stc_size = 4096

    model = CC_NN(img_size, stc_size)
    model.train() # set to training mode 
    model.cuda()    # use gpu
    #model.to(device)
    next(model.parameters()).is_cuda

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) # 최적화 알고리즘 
    model_loss = nn.BCELoss() ### binary cross entropy - measuring the error of a reconstruction in for example an auto-encoder.
    #model_loss2 = nn.MSELoss()
    batch = args.batch_size
    #batch = 32
    save_path = './models/' + time.strftime('%Y%m%d%H%M') + '-model.pth'
    min_loss = np.Inf
    cap_per_img = 5 # number of captions per image in MSCOCO set
    target = torch.ones(batch * cap_per_img).cuda()


    # train model 
    print('Start training...')
    model = train_model(args, datasets, stcencoder, resnet, model, optimizer, model_loss, batch, save_path, min_loss, cap_per_img, target, MAX_EPOCH, k)
    print('Finished training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--print_every", type=int, default=10)
    args = parser.parse_args()
    main(args)

