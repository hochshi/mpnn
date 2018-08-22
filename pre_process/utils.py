import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

if torch.cuda.is_available():
    def from_numpy(arr):
        # type: (np.ndarray) -> torch.Tensor
        return torch.from_numpy(arr).half().cuda()
else:
    def from_numpy(arr):
        # type: (np.ndarray) -> torch.Tensor
        return torch.from_numpy(arr).half()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def choose_largest_fragment(mol):
    """Return the largest covalent unit.

    The largest fragment is determined by number of atoms (including hydrogens). Ties are broken by taking the
    fragment with the higher molecular weight, and then by taking the first alphabetically by SMILES if needed.

    :param mol: The molecule to choose the largest fragment from.
    :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
    :return: The largest fragment.
    :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
    """
    # TODO: Alternatively allow a list of fragments to be passed as the mol parameter
    fragments = Chem.GetMolFrags(mol, asMols=True)
    largest = None
    for f in fragments:
        smiles = Chem.MolToSmiles(f, isomericSmiles=True)
        # Count atoms
        atoms = 0
        for a in f.GetAtoms():
            atoms += 1 + a.GetTotalNumHs()
        # Skip this fragment if fewer atoms than the largest
        if largest and atoms < largest['atoms']:
            continue
        # Skip this fragment if equal number of atoms but weight is lower
        weight = rdMolDescriptors.CalcExactMolWt(f)
        if largest and atoms == largest['atoms'] and weight < largest['weight']:
            continue
        # Skip this fragment if equal atoms and equal weight but smiles comes last alphabetically
        if largest and atoms == largest['atoms'] and weight == largest['weight'] and smiles > largest['smiles']:
            continue
        # Otherwise this is the largest so far
        largest = {'smiles': smiles, 'fragment': f, 'atoms': atoms, 'weight': weight}
    return largest['fragment']