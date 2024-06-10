from rdkit import Chem
from rdkit.Chem import rdchem

import torch
from torch_geometric.datasets import QM9 as QM9_pyg
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from datasets_pamnet import QM9 as QM9_pamnet

"""
x features:
x = [H, C, N, O, F, atomic_number, aromatic, sp, sp2, sp3, num_hs]     # 11 features

x with extended features (with_mol=True):
x = [H, C, N, O, F, atomic_number, aromatic, sp, sp2, sp3, num_hs, formal_charge, r_covalent, r_vdw, mass, total_valence, degree, implicit_valence, chiral_tag]     # 19 features

- first five features are the one-hot encoded atom type
- atomic_number is the atomic number of the atom
- aromatic is a boolean indicating whether the atom is part of an aromatic ring
- sp, sp2, sp3 are atom hybridization states
- num_hs is the number of hydrogen atoms bonded to the atom
- formal_charge is used to understand the way electrons are allocated throughout the molecule or the ion
- r_covalent is the covalent radius of the atom
- r_vdw is the van der Waals radius of the atom
- mass is the atom mass
- total_valence is the total number of electrons in the valence shell of the atom
- degree is the number of bonds connected to the atom
- implicit_valence is the number of implicit valence electrons
- chiral_tag is the chiral tag of the atom

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
"""

def data_to_mol(data):
    """
    Convert a PyG Data object to an RDKit Mol object.
    """
    mol = Chem.RWMol()
    atomic_nums = data.z.tolist()

    for atomic_num in atomic_nums:
        mol.AddAtom(Chem.Atom(atomic_num))
    
    num_bonds = data.edge_index.shape[1]
    cache = set()
    for i in range(num_bonds):
        start = int(data.edge_index[0, i])
        end = int(data.edge_index[1, i])
        bond_type = int(data.edge_attr[i].tolist()[0])
        if (start, end, bond_type) in cache or (end, start, bond_type) in cache:
            continue
        cache.add((start, end, bond_type))
        mol.AddBond(start, end, Chem.rdchem.BondType.values[bond_type])
    
    mol = mol.GetMol()  # Convert to Mol object
    return mol

class RemoveHydrogens(BaseTransform):
    """
    Transform class to remove hydrogen atoms from a molecule.
    Usage:
    dataset = QM9(root='gdb9', transform=RemoveHydrogens())
    """
    def __init__(self, model=None):
        super(RemoveHydrogens, self).__init__()
        self.model = model

    def __call__(self, data: Data) -> Data:
        if self.model == 'PAMNet':
            # in this case x is a 1D tensor of atomic numbers
            new_data = data.clone()
            mask = new_data.x != 0

            # mapping from old indices to new indices
            index_mapping = torch.arange(len(mask))[mask]

            new_data.x = new_data.x[mask]
            new_data.pos = new_data.pos[mask]

            # map edge_indexes to new indexes
            mask_edge = [torch.isin(new_data.edge_index[i], index_mapping) for i in range(2)]
            mask_edge = mask_edge[0] & mask_edge[1]
            new_data.edge_index = new_data.edge_index[:, mask_edge]
            for i in range(2):
                for j in range(new_data.edge_index[i].size(0)):
                    new_data.edge_index[i][j] = torch.where(index_mapping == new_data.edge_index[i][j])[0][0]
            return new_data
        else:
            # get hydrogen atoms (z is the atomic numbers)
            hydrogen_mask = data.z == 1
            non_hydrogen_mask = ~hydrogen_mask

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
    """
    Transform class to add atomic features.
    By default it converts the data object to a molecule object and optionally can add the following features:
    - formal_charge
    - r_covalent (float)
    - r_vdw (float)
    - mass (float)
    - total_valence (int)
    - degree (int)
    - implicit_valence (int)
    - chiral_tag (one hot encoded as [0,0], [1,0], [0,1], [1,1] for CHI_UNSPECIFIED, CHI_TETRAHEDRAL_CW, CHI_TETRAHEDRAL_CCW, CHI_OTHER respectively)

    Usage:
    dataset = QM9(root='gdb9', transform=ExtendAtomFeatures(with_mol=True, formal_charge=True, total_valence=True, degree=True, implicit_valence=True, r_covalent=True, r_vdw=True, chiral_tag=True, mass=True))
    """

    def __init__(self, 
                 with_mol=True, 
                 formal_charge=False,
                 total_valence=False,
                 degree=False,
                 implicit_valence=False,
                 r_covalent=False,
                 r_vdw=False,
                 chiral_tag=False,
                 mass=False,
                 ):
        super(ExtendAtomFeatures, self).__init__()
        self.with_mol = with_mol
        self.formal_charge = formal_charge
        self.total_valence = total_valence
        self.degree = degree
        self.implicit_valence = implicit_valence
        self.r_covalent = r_covalent
        self.r_vdw = r_vdw
        self.chiral_tag = chiral_tag
        self.mass = mass

    def get_atom_features(self, atom: Chem.Atom) -> list:
        features = []
        features.append(atom.GetFormalCharge())
        features.append(rdchem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()))
        features.append(rdchem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))
        features.append(atom.GetMass())
        return features
    
    def get_atom_features_with_mol(self, mol: Chem.Mol) -> list:
        chiral_tag_to_int = {
            rdchem.ChiralType.CHI_UNSPECIFIED: [0,0],
            rdchem.ChiralType.CHI_TETRAHEDRAL_CW: [1,0],
            rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: [0,1],
            rdchem.ChiralType.CHI_OTHER: [1,1],
        }
        all_features = []
        # Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache(strict=False)
        for atom in mol.GetAtoms():
            features = []
            if self.formal_charge:
                features.append(atom.GetFormalCharge())
            if self.total_valence:
                features.append(atom.GetTotalValence())
            if self.degree:
                features.append(atom.GetDegree())
            if self.implicit_valence:
                features.append(atom.GetImplicitValence())
            if self.r_covalent:
                features.append(rdchem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()))
            if self.r_vdw:
                features.append(rdchem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))
            if self.chiral_tag:
                features += chiral_tag_to_int.get(atom.GetChiralTag(), [0,0])
            if self.mass:
                features.append(atom.GetMass())
            all_features.append(features)
        return all_features

    def __call__(self, data: Data) -> Data:
        if self.with_mol:
            mol = data_to_mol(data)
            extended_features = self.get_atom_features_with_mol(mol)
        else:
            extended_features = []
            for i in range(data.x.shape[0]):
                atom = Chem.Atom(data.z[i].item())
                features = self.get_atom_features(atom)
                extended_features.append(features)
        data.x = torch.cat((data.x, torch.tensor(extended_features, dtype=torch.float)), dim=1)
        return data


"""
Testing functions
"""
def test_remove_hydrogens(model=None):
    ########## Remove hyrogens example #############
    example_index = 100

    if model == 'PAMNet':
        dataset = QM9_pamnet(root='gdb9')
    else:
        dataset = QM9_pyg(root='gdb9')
    example = dataset[example_index]
    print(f"dataset[{example_index}]", example)

    if model == 'PAMNet':
        dataset_no_h = QM9_pamnet(root='gdb9', transform=RemoveHydrogens(model))
    else:
        dataset_no_h = QM9_pyg(root='gdb9', transform=RemoveHydrogens(model))
    new_example = dataset_no_h[example_index]
    print(f"dataset_no_h[{example_index}]", new_example)

def test_extend_features(with_mol, formal_charge=False, total_valence=False, degree=False, implicit_valence=False, r_covalent=False, r_vdw=False, chiral_tag=False, mass=False):
    ########## Extend atom features example #############
    example_index = 100
    dataset = QM9_pyg(root='gdb9')
    example = dataset[example_index]
    print(f"dataset[{example_index}]", example)

    dataset_extended = QM9_pyg(root='gdb9', transform=ExtendAtomFeatures(with_mol, formal_charge, total_valence, degree, implicit_valence, r_covalent, r_vdw, chiral_tag, mass))
    new_example = dataset_extended[example_index]
    print(f"dataset_extended[{example_index}]", new_example)
    
if __name__ == '__main__':
    print()
    print("Running tests.")
    print()
    print("test_remove_hydrogens()")
    test_remove_hydrogens()
    print()
    print("test_remove_hydrogens(model='PAMNet')")
    test_remove_hydrogens(model='PAMNet')
    print()
    print("test_extend_features(with_mol=False)")
    test_extend_features(with_mol=False)
    print()
    print("test_extend_features(with_mol=True)")
    test_extend_features(with_mol=True)
    print()
    print("test_extend_features(with_mol=True, formal_charge=True)")
    test_extend_features(with_mol=True, formal_charge=True)
    print()
    print("test_extend_features(with_mol=True, total_valence=True)")
    test_extend_features(with_mol=True, total_valence=True)
    print()
    print("test_extend_features(with_mol=True, degree=True)")
    test_extend_features(with_mol=True, degree=True)
    print()
    print("test_extend_features(with_mol=True, implicit_valence=True)")
    test_extend_features(with_mol=True, implicit_valence=True)
    print()
    print("test_extend_features(with_mol=True, r_covalent=True)")
    test_extend_features(with_mol=True, r_covalent=True)
    print()
    print("test_extend_features(with_mol=True, r_vdw=True)")
    test_extend_features(with_mol=True, r_vdw=True)
    print()
    print("test_extend_features(with_mol=True, chiral_tag=True)")
    test_extend_features(with_mol=True, chiral_tag=True)
    print()
    print("test_extend_features(with_mol=True, mass=True)")
    test_extend_features(with_mol=True, mass=True)
    print()
    print("test_extend_features(with_mol=True, formal_charge=True, total_valence=True, degree=True, implicit_valence=True, r_covalent=True, r_vdw=True, chiral_tag=True, mass=True)")
    test_extend_features(with_mol=True, formal_charge=True, total_valence=True, degree=True, implicit_valence=True, r_covalent=True, r_vdw=True, chiral_tag=True, mass=True)
    print()
    print("All tests passed.")
    print()