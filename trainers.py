from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

def baseline_trainer(model, optimizer, train_loader, val_loader, num_epochs, patience, target, verbose=False):

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

def baseline_eval(model, loader, target):
    model.eval()
    mae = 0
    n_loader = 0
    for data in loader:
        y = data.y[:, target].detach()
        out = model(data).squeeze().detach()
        mae += (out - y).abs().sum().item()
        n_loader += y.shape[0]
    return mae / n_loader