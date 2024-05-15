from rdkit import Chem
from rdkit.Chem import rdchem

import torch
from torch_geometric.datasets import QM9
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

"""
x features:
x = [H, C, N, O, F, atomic_number, aromatic, sp, sp2, sp3, num_hs]     # 11 features

x with extended features
x = [H, C, N, O, F, atomic_number, aromatic, sp, sp2, sp3, num_hs, formal_charge, r_covalent, r_vdw, mass]     # 15 features

- first five features are the one-hot encoded atom type
- atomic_number is the atomic number of the atom
- aromatic is a boolean indicating whether the atom is part of an aromatic ring
- sp, sp2, sp3 are atom hybridization states
- num_hs is the number of hydrogen atoms bonded to the atom
- formal_charge is used to understand the way electrons are allocated throughout the molecule or the ion
- r_covalent is the covalent radius of the atom
- r_vdw is the van der Waals radius of the atom
- mass is the atom mass

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
"""

# def data_to_mol(data):
#     mol = Chem.RWMol()
#     atomic_nums = data.z.tolist()

#     for atomic_num in atomic_nums:
#         mol.AddAtom(Chem.Atom(atomic_num))
    
#     num_bonds = data.edge_index.shape[1]
#     cache = set()
#     for i in range(num_bonds):
#         start = int(data.edge_index[0, i])
#         end = int(data.edge_index[1, i])
#         bond_type = int(data.edge_attr[i].tolist()[0])
#         if (start, end, bond_type) in cache or (end, start, bond_type) in cache:
#             continue
#         cache.add((start, end, bond_type))
#         mol.AddBond(start, end, Chem.rdchem.BondType.values[bond_type])
    
#     mol = mol.GetMol()  # Convert to Mol object
#     return mol

# def mol_to_data(mol):
#     atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
#     edge_indices = [[], []]
#     bond_types = []

#     for bond in mol.GetBonds():
#         edge_indices[0].append(bond.GetBeginAtomIdx())
#         edge_indices[1].append(bond.GetEndAtomIdx())
#         bond_types.append(bond.GetBondTypeAsDouble())

#     edge_index = torch.tensor(edge_indices, dtype=torch.long)
#     x = torch.tensor(atom_features, dtype=torch.long).view(-1, 1)
#     edge_attr = torch.tensor(bond_types, dtype=torch.float).view(-1, 1)
    
#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# def remove_hydrogens(data):
#     mol = data_to_mol(data)
#     mol = Chem.RemoveHs(mol)
#     new_data = mol_to_data(mol)
    
#     new_data.y = data.y
#     new_data.idx = data.idx
#     new_data.name = data.name
#     return new_data


class RemoveHydrogens(BaseTransform):
    def __call__(self, data: Data) -> Data:
        # Identify hydrogen atoms (atomic number 1)
        hydrogen_mask = data.z == 1
        non_hydrogen_mask = ~hydrogen_mask

        # Keep only non-hydrogen atoms
        data.x = data.x[non_hydrogen_mask]
        data.z = data.z[non_hydrogen_mask]

        # Map old indices to new indices
        mapping = torch.cumsum(non_hydrogen_mask, dim=0) - 1
        mapping[hydrogen_mask] = -1

        # Filter out edges connected to hydrogen atoms
        edge_index = data.edge_index.numpy()
        edge_mask = non_hydrogen_mask[edge_index[0]] & non_hydrogen_mask[edge_index[1]]

        data.edge_index = data.edge_index[:, edge_mask]
        data.edge_attr = data.edge_attr[edge_mask]

        # Remap edge indices
        data.edge_index = mapping[data.edge_index]

        return data
    
class ExtendAtomFeatures(BaseTransform):

    def get_atom_features(self, atom: Chem.Atom) -> list:
        features = []
        features.append(atom.GetFormalCharge())
        # features.append(atom.GetTotalValence())
        # features.append(atom.GetDegree())
        # features.append(atom.GetImplicitValence())
        features.append(rdchem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()))
        features.append(rdchem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))
        # features.append(atom.GetChiralTag())
        features.append(atom.GetMass())
        return features


    def __call__(self, data: Data) -> Data:
        extended_features = []
        for i in range(data.x.shape[0]):
            atom = Chem.Atom(data.z[i].item())
            features = self.get_atom_features(atom)
            extended_features.append(features)
        data.x = torch.cat((data.x, torch.tensor(extended_features, dtype=torch.float)), dim=1)
        return data
    
if __name__ == '__main__':

    ########## Remove hyrogens example #############
    example_index = 100

    dataset = QM9(root='gdb9')
    example = dataset[example_index]
    # print("x", example.x)
    # print("edge_index", example.edge_index)
    # print("edge_attr", example.edge_attr)
    # print("z", example.z)
    print(f"dataset[{example_index}]", example)

    dataset_no_h = QM9(root='gdb9', transform=RemoveHydrogens())
    new_example = dataset_no_h[example_index]
    print(f"dataset_no_h[{example_index}]", new_example)
    # print("x", new_example.x)
    # print("edge_index", new_example.edge_index)
    # print("edge_attr", new_example.edge_attr)
    # print("z", new_example.z)

    ###############################################

    ########## Extend atom features example #############
    example_index = 100
    dataset = QM9(root='gdb9')
    example = dataset[example_index]
    # print("x", example.x)
    print(f"dataset[{example_index}]", example)

    dataset_extended = QM9(root='gdb9', transform=ExtendAtomFeatures())
    new_example = dataset_extended[example_index]
    print(f"dataset_extended[{example_index}]", new_example)
    # print("x", new_example.x)