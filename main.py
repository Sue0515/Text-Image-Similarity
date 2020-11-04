import torch
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
from data_load import get_loader
from model import CC_NN
from train import train_model

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

    print('2. Initiate the training')
    model = CC_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'], 
    output_dim=input_data_par['num_class']).to(device)
    params_to_update = list(model_ft.parameters())
    model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model, data_loader, optimizer, alpha, beta, MAX_EPOCH)
    print('Training is completed!')
