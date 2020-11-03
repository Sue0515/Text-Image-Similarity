import torch
from data_load import get_loader
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = 'train2014'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    DATA_DIR = 'data/' + dataset + '/'
    MAX_EPOCH = 500
    batch_size = 100
    alpha = 1e-3
    beta = 1e-1
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 0

    print('1. Initiate the data loading ...')
    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)
    print('Data loading is completed!')

