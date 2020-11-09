from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import time
import copy
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

def train_model(args, datasets, stcencoder, resnet, model, optimizer, model_loss, batch, save_path, min_loss, cap_per_img, target, device="gpu", num_epochs=100):
    for epoch in range(num_epochs):
        #tracker_epoch = defaultdict(lambda: defaultdict(dict))
        for split, dataset in datasets.items():
            data_loader = DataLoader(dataset=dataset, num_workers=8, batch_size=batch, shuffle=True, pin_memory = True)
            for iteration, dat in enumerate(data_loader):
                with torch.no_grad():
                    img = resnet(torch.tensor(dat[0]).cuda()).view(batch, -1) # bs x args.inp_size
                    img = img.repeat(1, cap_per_img).view(-1, 512) # replicate image features for each caption 
                    neg_img = torch.cat((img[cap_per_img:], img[:cap_per_img]), dim=0).cuda() # permute the images to create a mismatch
                    raw_captions = list(zip(*dat[1])) # list of captions, each grouped by image
                    cap = []
                    for x in raw_captions:
                        cap += list(x)
                    captions = torch.tensor(stcencoder.encode(cap, bsize=16)).cuda() # num_sentences x 4096
                prob, neg_prob = model(img, captions, neg_img) ### model model 
                num_data = prob.shape[0]

                loss = model_loss(prob, target[:num_data]) + model_loss(neg_prob, 1 - target[:num_data])
                optimizer.zero_grad() ## initialize gradient to 0
                loss.backward() ## calculate the gradient (back propagation)
                optimizer.step() ## update parameter with calculated gradient 

                if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                    _loss = loss.item()
                    print("Epoch %d, Batch %04d/%i, Loss %9.4f"%(epoch, iteration, len(data_loader)-1, _loss))
                    if _loss < min_loss:
                        min_loss = _loss
                        print("Saving model to " + save_path)
                        torch.save(model.state_dict(), save_path)
                    print("-----------------------------------------------")
                    print("Pos min:%.2f max:%.2f, mean:%.2f"%(prob.min(), prob.max(), prob.mean()))
                    print("Neg min:%.2f max:%.2f, mean:%.2f"%(neg_prob.min(), neg_prob.max(), neg_prob.mean()))
                    print("-----------------------------------------------")
    return model