import os
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_geometric.datasets import QM9 as QM9_pyg
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from utils import ExtendAtomFeatures, RemoveHydrogens
from trainers import *
from datasets_pamnet import QM9 as QM9_pamnet

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

# Target property to train on
# Our experiments are performed on targets 0, 5, 11 
# 0: Dipole moment (dependent on the structure)                    " easy "
# 5: Electronic spatial extent (dependent on the size)             " medium "
# 11: Heat capacity at 298.15K (dependent on the vibration modes)  " hard "
parser = argparse.ArgumentParser(description="A simple command line program")
parser.add_argument("target", type=int, help="Target property to train on")
args = parser.parse_args()
target = args.target
if target not in range(19):
    print("Invalid target property")
    sys.exit(1)

model_name = "Baseline"    # Baseline DimeNet GraphSAGE GAT PAMNet

batch_size = 32
num_epochs = 200
patience = 20
hydrogens = True
extra_atom_features = True

pred_dir = './predictions/'
fig_dir = './figures/'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

config = {
    "Baseline": {"trainer":baseline_trainer, "model_kwargs":{"hidden_dim":16}, "evaluator":baseline_eval},
    "DimeNet": {"trainer":dimenet_trainer, "model_kwargs":{"hidden_dim":16}, "evaluator":baseline_eval},
    "PAMNet": {"trainer":pamnet_main, "model_kwargs":{}, "evaluator":pamnet_test},
}

def main(model_name, extra_atom_kwargs=None):

    # get the extra atom kwarg for which the value is True
    if extra_atom_kwargs:
        extra_feature = [key for key, value in extra_atom_kwargs.items() if value]
        assert len(extra_feature) == 1
        extra_feature = extra_feature[0]

    log_file = open(os.path.join(pred_dir, model_name + f'_target_{target}_extrafeature_{extra_feature}' + ".txt"), 'a')
    print(f"Logging to {log_file.name}")
    normal_std_out = sys.stdout
    sys.stdout = log_file

    # print parameters
    print("model =", model_name)
    print("target =", target)
    print("batch_size =", batch_size)
    print("num_epochs =", num_epochs)
    print("patience =", patience)
    print("hydrogens =", hydrogens)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gdb9')
    transforms = []
    if not hydrogens:
        transforms.append(RemoveHydrogens())
    if extra_atom_features:
        transforms.append(ExtendAtomFeatures(with_mol=True, **extra_atom_kwargs))

    class MyTransform(object):
        target = target if target not in [7, 8, 9, 10] else target + 5
        def __call__(self, data):
            data.y = data.y[:, self.target]
            return data
        
    dataset = QM9_pyg(path, transform=Compose(transforms)).shuffle() if model_name not in ["PAMNet", "PAMNet_s"] else QM9_pamnet(path, transform=MyTransform()).shuffle()

    train_dataset = dataset[:110000]
    val_dataset = dataset[110000:120000]
    test_dataset = dataset[120000:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam
    trainer = config[model_name]["trainer"]
    _, train_losses, val_losses, test_loss = trainer(optimizer=optimizer, 
                                       train_loader=train_loader, 
                                       val_loader=val_loader, 
                                       test_loader=test_loader,
                                       num_epochs=num_epochs, 
                                       patience=patience, 
                                       target=target, 
                                       verbose=False,
                                       model_kwargs=config[model_name]["model_kwargs"],
                                       )
    np.save(os.path.join(pred_dir, model_name + f'_target_{target}_extrafeature_{extra_feature}' + "_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(pred_dir, model_name + f'_target_{target}_extrafeature_{extra_feature}' + "_val_losses.npy"), np.array(val_losses))
    np.save(os.path.join(pred_dir, model_name + f'_target_{target}_extrafeature_{extra_feature}' + "_test_loss.npy"), np.array([test_loss]))

    print(f"Final Test Loss: {test_loss:.4f}")
    log_file.close()
    sys.stdout = normal_std_out
    print("run.py done.")
    return train_losses, val_losses, test_loss
    

if __name__ == '__main__':
    extra_atom_kwargs = {
        'formal_charge':False, 
        'total_valence':False, 
        'degree':False, 
        'implicit_valence':False, 
        'r_covalent':False, 
        'r_vdw':False, 
        'chiral_tag':False, 
        'mass':False
    }
    # for each extra feature create a new run with the corresponding feature set to True
    for key in extra_atom_kwargs.keys():
        print(f"Running {model_name} with {key} extra feature")
        extra_atom_kwargs[key] = True
        train_losses, val_losses, test_loss = main(model_name, extra_atom_kwargs)
        extra_atom_kwargs[key] = False

        plt.plot(train_losses, label="Train Error")
        plt.plot(val_losses, label="Validation Error")
        plt.axhline(y=test_loss, color='r', linestyle='-', label="Final Test Error")
        plt.legend()
        plt.title(f"{model_name} target {target} hydrogens {hydrogens}")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.savefig(os.path.join(fig_dir, model_name + f'_target_{target}_extrafeature_{key}' + "_losses.png"))
        # plt.show()