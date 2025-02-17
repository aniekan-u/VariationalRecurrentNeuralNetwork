import os
import random
import math
from copy import copy, deepcopy
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from model import VRNN
from data_loader import NWB


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for
inference, prior, and generating models."""

def train(epoch):
    train_loss = 0
    for batch_idx, (data, _, _) in enumerate(train_loader):

        #transforming data
        data = data.to(torch.float)
        data = data.to(device)
        data = data.squeeze().transpose(0, 1) # (seq, batch, elem)
        data = (data - data.min()) / (data.max() - data.min())

        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _, _ = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        #printing
        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
            epoch, batch_idx * batch_size, batch_size * (len(train_loader.dataset)//batch_size),
            100. * batch_idx / len(train_loader),
            kld_loss / batch_size,
            nll_loss / batch_size))	

            if PLOT_SAMPLE:
                sample = model.sample(torch.tensor(100, device=device))
                sample = sample.squeeze().to(torch.device('cpu')).numpy()
                ex_neur = np.random.permutation(sample.shape[1])
                plt.clf()
                plt.plot(sample[:,ex_neur[:25]])
                plt.xlabel('Time Bins')
                plt.ylabel('Rates')
                plt.pause(1e-6)

        train_loss += loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
    epoch, train_loss / len(train_loader.dataset)))


def validate(epoch):
    """uses test data to evaluate
    likelihood of the model"""

    mean_kld_loss, mean_nll_loss = 0, 0
    with torch.no_grad():
        for i, (data, _, _) in enumerate(val_loader):

            #transforming data
            data = data.to(torch.float)
            data = data.to(device)
            data = data.squeeze().transpose(0, 1)
            data = (data - data.min()) / (data.max() - data.min())

            kld_loss, nll_loss, _, _, _ = model(data)
            mean_kld_loss += kld_loss.item()
            mean_nll_loss += nll_loss.item()

    mean_kld_loss /= len(val_loader.dataset)
    mean_nll_loss /= len(val_loader.dataset)

    print('====> Validation set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
        mean_kld_loss, mean_nll_loss))
    return mean_kld_loss + mean_nll_loss

# changing device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

if __name__ == '__main__':

    # Hyperparameters

    # Model Parameters
    x_dim = 100
    h_dim = 20
    z_dim = 16
    n_layers = 1
    print(f'x: {x_dim}, h: {h_dim}, z: {z_dim}')

    # Training Parameters
    n_epochs = 350
    clip = 10
    batch_size = 128
    parts = {'train': .8, 'val': .2}
    n_seq = 10000
    seq_len = 100
    seed = 1
    print(f'e: {n_epochs}, ns~: {n_seq}, sl: {seq_len}, s: {seed}')

    # Learning Rate Schedule
    learning_rate = 1e-3
    decay_every = 10 # epochs
    decay_factor = 0.5 
    start_decay = 500
    MAX_PATIENCE = 10
    print(f'sd: {start_decay}, p: {MAX_PATIENCE}')

    # Manual seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    plt.ion()

    # Init Datasets and Dataloaders
    print("Creating Training Dataset and Dataloader...")
    nwb_train = NWB(experiment=1, train=True, resample_val=5, seq_len=seq_len, neur_count = x_dim,
                                    N_seq=n_seq, parts_fract_seq=parts, shuffle=True, seq_start_mode='unique')
    nwb_train.set_curr_part('train')
    train_loader = torch.utils.data.DataLoader(nwb_train, batch_size=batch_size)

    print("Creating Validation Dataset and Dataloader...")
    nwb_val = copy(nwb_train)
    nwb_val.set_curr_part('val')
    val_loader = torch.utils.data.DataLoader(nwb_val, batch_size=batch_size)
    
    #update n_seq
    n_seq = nwb_train.get_total_num_seq()
    print(f'ns: {n_seq}')

    # IO
    PLOT_SAMPLE = True
    print_every = math.ceil(parts['train']*n_seq/(10*batch_size)) # batches
    print(f'Print every: {print_every}')
    save_every = 10 # epochs

    # Directories
    SAVES_DIR = f'./saves_x{x_dim}h{h_dim}z{z_dim}la{n_layers}_e{n_epochs}b{batch_size}ns{n_seq}sl{seq_len}s{seed}_sd{start_decay}p{MAX_PATIENCE}/'
    if not os.path.isdir(SAVES_DIR): os.mkdir(SAVES_DIR)
    print(f'SAVES DIR: {SAVES_DIR}')

    # Creating Model
    print("Creating Model...")
    model = VRNN(x_dim, h_dim, z_dim, n_layers)
    model = model.to(device)

    #Early stopping
    patience = MAX_PATIENCE
    old_val_loss = float('inf')

    print("Beginning Training...")
    for epoch in range(1, n_epochs + 1):

        #training + testing
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train(epoch)
        val_loss = validate(epoch)
        
        improved = val_loss <  old_val_loss
        old_val_loss = val_loss
        patience = MAX_PATIENCE if improved else patience - 1
        
        #saving model
        if epoch % save_every == 1:
                fn = SAVES_DIR + 'vrnn_nwb_state_dict_'+str(epoch)+'.pth'
                torch.save(model.state_dict(), fn)
                print('Saved model to '+fn)
        if patience == 0 or epoch ==  n_epochs:
                fn = SAVES_DIR + 'vrnn_nwb_final_state_dict_'+str(epoch)+'.pth'
                torch.save(model.state_dict(), fn)
                print('Saved model to '+fn)
                break

        if epoch > start_decay and epoch % decay_every:
                learning_rate *= decay_factor
                if learning_rate < 0.0001:
                        learning_rate = 0.0001
