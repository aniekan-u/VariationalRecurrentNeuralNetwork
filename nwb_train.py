import math
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

def train(epoch, train_loader, clip):
    train_loss = 0
    for batch_idx, (data, _, _) in enumerate(train_loader):

        #transforming data
        data = data.to(torch.float)
        data = data.to(device)
        data = data.squeeze().transpose(0, 1) # (seq, batch, elem)
        data = (data - data.min()) / (data.max() - data.min())
        
        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _ = model(data)
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

            sample = model.sample(torch.tensor(28, device=device))
            sample = sample.to(torch.device('cpu')).numpy()
            ex_neur = np.random.permutation(sample.shape[1])
            plt.clf()
            plt.plot(sample[:,ex_neur[:10]])
            plt.xlabel('Time Bins')
            plt.ylabel('Rates')
            plt.pause(1e-6)

        train_loss += loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, test_loader):
    """uses test data to evaluate
    likelihood of the model"""

    mean_kld_loss, mean_nll_loss = 0, 0
    with torch.no_grad():
        for i, (data, _, _) in enumerate(test_loader):

            #transforming data
            data = data.to(torch.float)
            data = data.to(device)
            data = data.squeeze().transpose(0, 1)
            data = (data - data.min()) / (data.max() - data.min())

            kld_loss, nll_loss, _, _ = model(data)
            mean_kld_loss += kld_loss.item()
            mean_nll_loss += nll_loss.item()

    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)

    print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
        mean_kld_loss, mean_nll_loss))


# changing device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

if __name__ == '__main__':

    #device = torch.device('cpu')

    # Hyperparameters
    
    # Model Parameters
    x_dim = 100
    h_dim = 20
    z_dim = 16
    n_layers =  1
    
    # Training Parameters
    n_epochs = 150
    clip = 10
    learning_rate = 1e-3
    batch_size = 4
    n_train_seq = 500
    n_test_seq = 100
    seed = 1

    # IO
    print_every = 50 # batches
    save_every = 10 # epochs

    #manual seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    plt.ion()

    #init model + optimizer + datasets
    print("Creating Training Dataset and Dataloader...")
    nwb_train = NWB(experiment=1, train=True, resample_val=5,
                    seq_len=10, neur_count = x_dim, N_seq=n_train_seq)
    train_loader = torch.utils.data.DataLoader(nwb_train, batch_size=batch_size)

    print("Creating Test Dataset and Dataloader...")
    nwb_test = NWB(experiment=1, train=False, resample_val=5,
                    seq_len=10, neur_count = x_dim, N_seq=n_test_seq)
    test_loader = torch.utils.data.DataLoader(nwb_test, batch_size=batch_size)

    print("Creating Model...")
    model = VRNN(x_dim, h_dim, z_dim, n_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Beginning Training...")
    for epoch in range(1, n_epochs + 1):

        #training + testing
        train(epoch, train_loader, clip)
        test(epoch, test_loader)

        #saving model
        if epoch % save_every == 1:
            fn = 'saves/vrnn_nwb_state_dict_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)

