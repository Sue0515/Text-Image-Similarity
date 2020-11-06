import torch
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
from data_load import get_loader
from model import CC_NN
from embed import Embed
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

# if __name__ == '__main__':
#     dataset = 'train2014'
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     DATA_DIR = 'data/' + dataset + '/'
#     MAX_EPOCH = 500
#     batch_size = 100
#     alpha = 1e-3
#     beta = 1e-1
#     lr = 1e-4
#     betas = (0.5, 0.999)
#     weight_decay = 0

#     print('1. Initiate the data loading ...')
#     data_loader, input_data_par = get_loader(DATA_DIR, batch_size)
#     print('Data loading is completed!')

#     print('2. Initiate the training')
#     model = CC_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'], 
#     output_dim=input_data_par['num_class']).to(device)
#     params_to_update = list(model_ft.parameters())
#     model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model, data_loader, optimizer, alpha, beta, MAX_EPOCH)
#     print('Training is completed!')

def main(args):
    ts = time.time()
    datasets = OrderedDict()
    common_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    datasets['train'] = CocoCaptions(root='../train2014', 
        annFile='../annotations/captions_train2014.json', transform=common_trans)
    datasets['val'] = CocoCaptions(root='../val2014', 
        annFile='../annotations/captions_val2014.json', transform=common_trans)

    emb = Embed()
    print('Loading sentence encoder...')
    sentenc = emb.init_sentemb()
    print('Loading ResNet...')
    resnet = emb.load_resnet()
    img_enc_size = 512
    sent_enc_size = 4096

    sim = CC_NN(img_enc_size, sent_enc_size)
    sim.train()
    sim.cuda()

    optimizer = torch.optim.Adam(sim.parameters(), lr=args.learning_rate) # 최적화 알고리즘 
    sim_loss = nn.BCELoss() ### binary cross entropy - measuring the error of a reconstruction in for example an auto-encoder.
    bs = args.batch_size
    save_path = './models/' + time.strftime('%Y%m%d%H%M') + '-sim.pth'
    min_loss = np.Inf
    cap_per_img = 5 # number of captions per image in MSCOCO set
    target = torch.ones(bs * cap_per_img).cuda()
    for epoch in range(args.epochs):
        tracker_epoch = defaultdict(lambda: defaultdict(dict))
        for split, dataset in datasets.items():
            data_loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True)
            for iteration, dat in enumerate(data_loader):
                with torch.no_grad():
                    img = resnet(torch.tensor(dat[0]).cuda()).view(bs, -1) # bs x args.inp_size
                    img = img.repeat(1, cap_per_img).view(-1, 512) # replicate image features for each caption 
                    neg_img = torch.cat((img[cap_per_img:], img[:cap_per_img]), dim=0) # permute the images to create a mismatch
                    raw_captions = list(zip(*dat[1])) # list of captions, each grouped by image
                    cap = []
                    for x in raw_captions:
                        cap += list(x)
                    captions = torch.tensor(sentenc.encode(cap, bsize=16)).cuda() # num_sentences x 4096
                prob, neg_prob = sim(img, captions, neg_img) ### sim model 
                num_data = prob.shape[0]

                loss = sim_loss(prob, target[:num_data]) + sim_loss(neg_prob, 1 - target[:num_data])
                optimizer.zero_grad() ## initialize gradient to 0
                loss.backward() ## calculate the gradient (back propagation)
                optimizer.step() ## update parameter with calculated gradient 

                if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                    _loss = loss.item()
                    print("Epoch %d, Batch %04d/%i, Loss %9.4f"%(epoch, iteration, len(data_loader)-1, _loss))
                    if _loss < min_loss:
                        min_loss = _loss
                        print("Saving model to " + save_path)
                        torch.save(sim.state_dict(), save_path)
                    print("-----------------------------------------------")
                    print("Pos min:%.2f max:%.2f, mean:%.2f"%(prob.min(), prob.max(), prob.mean()))
                    print("Neg min:%.2f max:%.2f, mean:%.2f"%(neg_prob.min(), neg_prob.max(), neg_prob.mean()))
                    print("-----------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--print_every", type=int, default=10)
    args = parser.parse_args()
    main(args)

