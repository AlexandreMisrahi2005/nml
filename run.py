import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
print("in 0")
from torch_geometric.datasets import QM9
print("in 1")
from torch_geometric.loader import DataLoader
print("in 2")
from models import Baseline, GCN, DimeNetModel
from utils import ExtendAtomFeatures, RemoveHydrogens
from trainers import baseline_trainer, baseline_eval, pamnet_main

# """
# +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | Target | Property                         | Description                                                                       | Unit                                        |
#     +========+==================================+===================================================================================+=============================================+
#     | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
#     | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
#     +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
# """

verbose = False

# Target property to train on
target = 0
# We choose in [0,1,2,3,5,6,7,8,9,10,11]
# Target 4 delta epsilon is kind of redundant ( = ε_LUMO − ε_HOMO in Density Functional Theory)
# Other targets are also less 

model_name = "Baseline"

batch_size = 32
num_epochs = 50
patience = 3
hydrogens = False

pred_dir = './predictions/'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

config = {
    "Baseline": {"model":Baseline, "trainer":baseline_trainer, "model_kwargs":{"hidden_dim":16}, "evaluator":baseline_eval},
    "GCN": GCN,
    "DimeNet": DimeNetModel,
    "PAMNet": {"model":'PAMNet', "trainer":pamnet_main},
}

def main(model_name):

    log_file = open(os.path.join(pred_dir, model_name + f'_hydrogens_{hydrogens}' + ".txt"), 'a')
    sys.stdout = log_file

    # print parameters
    print("model =", model_name)
    print("target =", target)
    print("batch_size =", batch_size)
    print("num_epochs =", num_epochs)
    print("patience =", patience)
    print("hydrogens =", hydrogens)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gdb9')
    if hydrogens:
        dataset = QM9(path).shuffle()
    else:
        dataset = QM9(path, transform=RemoveHydrogens())

    train_dataset = dataset[:110000]
    val_dataset = dataset[110000:120000]
    test_dataset = dataset[120000:]

    # Normalize targets
    mean = train_dataset.data.y.mean(dim=0, keepdim=True)
    std = train_dataset.data.y.std(dim=0, keepdim=True)
    # dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, target].item(), std[:, target].item()
    # don't standardize the target
    mean, std = 0, 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = config[model_name]["model"](dataset.num_features, **config[model_name]["model_kwargs"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = config[model_name]["trainer"]
    train_losses, val_losses = trainer(model=model, 
                                       optimizer=optimizer, 
                                       train_loader=train_loader, 
                                       val_loader=val_loader, 
                                       num_epochs=num_epochs, 
                                       patience=patience, 
                                       target=target, 
                                       verbose=False,
                                       )
    evaluator = config[model_name]["evaluator"]
    test_loss = evaluator(model, test_loader, target, std)
    print(f"Final Test Loss: {test_loss:.4f}")
    log_file.close()
    return train_losses, val_losses, test_loss
    

if __name__ == '__main__':
    train_losses, val_losses, test_loss = main(model_name)


    # plt.plot(train_losses, label="Train Loss")
    # plt.plot(val_losses, label="Validation Loss")
    # plt.legend()
    # plt.show()