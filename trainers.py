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

from models import *
from models_pamnet import PAMNet, PAMNet_s, Config
from utils_pamnet import EMA

def baseline_trainer(optimizer, 
                     train_loader, 
                     val_loader, 
                     test_loader,
                     num_epochs, 
                     patience, 
                     target, 
                     model_kwargs=dict(),
                     verbose=False,
                     ):
    
    """
    Trainer for the Baseline model
    Args:
        optimizer: torch.optim.Optimizer
        train_loader: torch_geometric.data.DataLoader
        val_loader: torch_geometric.data.DataLoader
        test_loader: torch_geometric.data.DataLoader
        num_epochs: int
        patience: int
        target: int
        model_kwargs: dict
        verbose: bool

    Returns:
        model: torch.nn.Module
        train_losses: list
        val_losses: list
        test_loss: float
    """
    
    num_features = train_loader.dataset.num_features
    model = Baseline(num_features, **model_kwargs)
    optimizer = optimizer(model.parameters(), lr=1e-2, weight_decay=5e-4)
    print("Number of model parameters: ", count_parameters(model))

    train_losses = []
    val_losses = []
    early_stop = 0

    for epoch in range(num_epochs):
        model.train()

        epoch_train_losses = []
        for data in tqdm(train_loader):
        
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
        else:
            test_loss = baseline_eval(model, test_loader, target)
        val_losses.append(epoch_val_loss)
        if early_stop == patience:
            print("!!! Early Stopping !!!")
            break
    print('\n')
    return model, train_losses, val_losses, test_loss

def baseline_eval(model, loader, target):
    """
    Evaluate the model on the test set
    Args:
        model: torch.nn.Module
        loader: torch_geometric.data.DataLoader
        target: int

    Returns:
        mae: float
    """
    model = model.to('cpu')
    model.eval()
    mae = 0
    n_loader = 0
    for data in loader:
        data = data.to('cpu')
        y = data.y[:, target].detach()
        out = model(data).squeeze().detach()
        mae += (out - y).abs().sum().item()
        n_loader += y.shape[0]
    return mae / n_loader



def dimenet_trainer(optimizer, 
                     train_loader, 
                     val_loader, 
                     test_loader,
                     num_epochs, 
                     patience, 
                     target, 
                     model_kwargs=dict(),
                     verbose=False,
                     ):
    """
    Trainer for the DimeNet model
    Args:
        optimizer: torch.optim.Optimizer
        train_loader: torch_geometric.data.DataLoader
        val_loader: torch_geometric.data.DataLoader
        test_loader: torch_geometric.data.DataLoader
        num_epochs: int
        patience: int
        target: int
        model_kwargs: dict
        verbose: bool

    Returns:
        model: torch.nn.Module
        train_losses: list
        val_losses: list
        test_loss: float
    """
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = "cpu"
    
    model = DimeNetModel(**model_kwargs).to(device)
    optimizer = optimizer(model.parameters(), lr=1e-2, weight_decay=5e-4)
    print("Number of model parameters: ", count_parameters(model))

    train_losses = []
    val_losses = []
    early_stop = 0

    for epoch in range(num_epochs):
        model.train()

        epoch_train_losses = []
        for data in tqdm(train_loader):
            data = data.to(device)
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
        else:
            test_loss = baseline_eval(model, test_loader, target)
        val_losses.append(epoch_val_loss)
        if early_stop == patience:
            print("!!! Early Stopping !!!")
            break
    print('\n')
    return model, train_losses, val_losses, test_loss

def graphsage_trainer(optimizer, 
                     train_loader, 
                     val_loader, 
                     test_loader,
                     num_epochs, 
                     patience, 
                     target, 
                     model_kwargs=dict(),
                     verbose=False,
                     ):
    
    """
    Trainer for the GraphSAGE model
    Args:
        optimizer: torch.optim.Optimizer
        train_loader: torch_geometric.data.DataLoader
        val_loader: torch_geometric.data.DataLoader
        test_loader: torch_geometric.data.DataLoader
        num_epochs: int
        patience: int
        target: int
        model_kwargs: dict
        verbose: bool

    Returns:
        model: torch.nn.Module
        train_losses: list
        val_losses: list
        test_loss: float
    """
    
    num_features = train_loader.dataset.num_features
    model = GraphSAGEModel(num_features, **model_kwargs)
    optimizer = optimizer(model.parameters(), lr=1e-2, weight_decay=5e-4)
    print("Number of model parameters: ", count_parameters(model))

    train_losses = []
    val_losses = []
    early_stop = 0

    for epoch in range(num_epochs):
        model.train()

        epoch_train_losses = []
        for data in tqdm(train_loader):
        
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
        else:
            test_loss = baseline_eval(model, test_loader, target)
        val_losses.append(epoch_val_loss)
        if early_stop == patience:
            print("!!! Early Stopping !!!")
            break
    print('\n')
    return model, train_losses, val_losses, test_loss

def gat_trainer(optimizer, 
                     train_loader, 
                     val_loader, 
                     test_loader,
                     num_epochs, 
                     patience, 
                     target, 
                     model_kwargs=dict(),
                     verbose=False,
                     ):
    
    """
    Trainer for the GAT model
    Args:
        optimizer: torch.optim.Optimizer
        train_loader: torch_geometric.data.DataLoader
        val_loader: torch_geometric.data.DataLoader
        test_loader: torch_geometric.data.DataLoader
        num_epochs: int
        patience: int
        target: int
        model_kwargs: dict
        verbose: bool
    
    Returns:
        model: torch.nn.Module
        train_losses: list
        val_losses: list
        test_loss: float
    """
    
    num_features = train_loader.dataset.num_features
    model = GATModel(num_features, **model_kwargs)
    optimizer = optimizer(model.parameters(), lr=1e-2, weight_decay=5e-4)
    print("Number of model parameters: ", count_parameters(model))

    train_losses = []
    val_losses = []
    early_stop = 0

    for epoch in range(num_epochs):
        model.train()

        epoch_train_losses = []
        for data in tqdm(train_loader):
        
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
        else:
            test_loss = baseline_eval(model, test_loader, target)
        val_losses.append(epoch_val_loss)
        if early_stop == patience:
            print("!!! Early Stopping !!!")
            break
    print('\n')
    return model, train_losses, val_losses, test_loss


####################################################################################
####################### Code for PAMNet adapted from ###############################
##### https://github.com/XieResearchGroup/Physics-aware-Multiplex-GNN/tree/main ####

def set_seed(seed):
    """
    Set seed for reproducibility (PyTorch and Numpy and Random)
    Caution: does not set seed for cudnn, which may introduce randomness if cuda is used
    Only used for PAMNet
    Args:
        seed: int
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    """
    Count the number of parameters in a model
    Args:
        model: torch.nn.Module

    Returns:
        int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pamnet_test(model, loader, ema, device):
    """
    Evaluate the model on the test set
    Args:
        model: torch.nn.Module
        loader: torch_geometric.data.DataLoader
        ema: utils_pamnet.EMA
        device: str

    Returns:
        mae: float
    """

    mae = 0
    ema.assign(model)
    for data in tqdm(loader):
        data = data.to(device)
        output = model(data)
        mae += (output - data.y).abs().sum().item()
    ema.resume(model)
    return mae / len(loader.dataset)

def pamnet_main(
        train_loader,
        val_loader,
        test_loader,
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
        model_kwargs=None,
        patience=None,
        target=0,
        verbose=True,
    ):
    """
    Main function for training PAMNet
    Args:
        train_loader: torch_geometric.data.DataLoader
        val_loader: torch_geometric.data.DataLoader
        test_loader: torch_geometric.data.DataLoader
        gpu: int
        seed: int
        dataset: str
        model: str
        num_epochs: int
        lr: float
        wd: float
        n_layer: int
        dim: int
        batch_size: int
        cutoff_l: float
        cutoff_g: float
        optimizer: torch.optim.Optimizer
        model_kwargs: dict
        patience: int
        target: int
        verbose: bool
        
    Returns:
        model: torch.nn.Module
        train_losses: list
        val_losses: list
        test_loss: float
    """

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = "cpu"
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
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        loss_all = 0
        step = 0
        model.train()
        for data in tqdm(train_loader):
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
        train_losses.append(loss)
        val_losses.append(val_loss)

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
    return model, train_losses, val_losses, test_loss

####################################################################################