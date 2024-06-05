from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

import os
import os.path as osp
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from PAMNet.models import PAMNet, PAMNet_s, Config
from PAMNet.utils import EMA
from PAMNet.datasets import QM9


def baseline_trainer(model, 
                     optimizer, 
                     train_loader, 
                     val_loader, 
                     num_epochs, 
                     patience, 
                     target, 
                     verbose=False
                     ):

    train_losses = []
    val_losses = []
    early_stop = 0

    for epoch in range(num_epochs):
        model.train()

        epoch_train_losses = []
        for data in tqdm(train_loader):
            if verbose:
                print("data.x", data.x)
                print("data.edge_index", data.edge_index)
                print("data.edge_attr", data.edge_attr)
                print("data.y[0]", data.y)
                print("data.pos", data.pos)
                print("data.idx", data.idx)
                print("data.name", data.name)
                print("data.z", data.z)
                print("data.batch", data.batch)
                print("data.ptr", data.ptr)
        
            y = data.y[:, target]

            optimizer.zero_grad()
            out = model(data)
            loss = F.l1_loss(out.squeeze(), y)
            epoch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_train_loss = np.mean(epoch_train_losses)
        train_losses.append(epoch_train_loss)

        epoch_val_losses = []
        for data in tqdm(val_loader):
            y = data.y[:, target]
            model.eval()
            out = model(data)
            loss = F.l1_loss(out.squeeze(), y)
            epoch_val_losses.append(loss.item())
        epoch_val_loss = np.mean(epoch_val_losses)
        print(f"Epoch {epoch+1}     Train Loss {epoch_train_loss:.4f}    Validation Loss: {epoch_val_loss:.4f}")
        if len(val_losses) > 0 and epoch_val_loss > val_losses[-1]:
            early_stop += 1
        val_losses.append(epoch_val_loss)
        if early_stop == patience:
            print("!!! Early Stopping !!!")
            break
    print('\n')
    return train_losses, val_losses

def baseline_eval(model, loader, target, train_std):
    model.eval()
    mae = 0
    n_loader = 0
    for data in loader:
        y = data.y[:, target].detach()
        out = model(data).squeeze().detach()
        mae += (train_std * (out - y)).abs().sum().item()
        n_loader += y.shape[0]
    return mae / n_loader


####################################################################################
####################### Code for PAMNet adapted from ###############################
##### https://github.com/XieResearchGroup/Physics-aware-Multiplex-GNN/tree/main ####

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pamnet_test(model, loader, ema, device):
    mae = 0
    ema.assign(model)
    for data in loader:
        data = data.to(device)
        output = model(data)
        mae += (output - data.y).abs().sum().item()
    ema.resume(model)
    return mae / len(loader.dataset)

def pamnet_main(
        train_loader,
        val_loader,
        test_loader=None,
        gpu=0, 
        seed=480, 
        dataset='QM9', 
        model='PAMNet', 
        num_epochs=300, 
        lr=1e-4, 
        wd=0, 
        n_layer=6, 
        dim=128, 
        batch_size=32, 
        cutoff_l=5.0, 
        cutoff_g=5.0,
        optimizer=None,
        patience=None,
        target=0,
        verbose=True,
    ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    set_seed(seed)


    config = Config(dataset=dataset, dim=dim, n_layer=n_layer, cutoff_l=cutoff_l, cutoff_g=cutoff_g)

    if model == 'PAMNet':
        model = PAMNet(config).to(device)
    else:
        model = PAMNet_s(config).to(device)
    print("Number of model parameters: ", count_parameters(model))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

    ema = EMA(model, decay=0.999)

    print("Start training!")
    best_val_loss = None
    for epoch in range(num_epochs):
        loss_all = 0
        step = 0
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = F.l1_loss(output, data.y)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
            optimizer.step()

            curr_epoch = epoch + float(step) / (len(train_loader.dataset) / batch_size)
            scheduler_warmup.step(curr_epoch)

            ema(model)
            step += 1
        loss = loss_all / len(train_loader.dataset)
        val_loss = pamnet_test(model, val_loader, ema, device)

        save_folder = osp.join(".", "save", dataset)
        if not osp.exists(save_folder):
            os.makedirs(save_folder)

        if best_val_loss is None or val_loss <= best_val_loss:
            test_loss = pamnet_test(model, test_loader, ema, device)
            best_val_loss = val_loss
            torch.save(model.state_dict(), osp.join(save_folder, "best_model.h5"))

        print('Epoch: {:03d}, Train MAE: {:.7f}, Val MAE: {:.7f}, '
            'Test MAE: {:.7f}'.format(epoch+1, loss, val_loss, test_loss))
    print('Best Validation MAE:', best_val_loss)
    print('Testing MAE:', test_loss)

####################################################################################