from abc import ABCMeta, abstractmethod

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import LabelBinarizer
from typing import List

from pre_process.utils import Singleton

AtomEncoder = List[LabelBinarizer]
BondEncoder = List[LabelBinarizer]


class GraphEncoder(object):
    __metaclass__ = Singleton

    def __init__(self):
        super(GraphEncoder, self).__init__()
        self.atom_enc = None
        self.bond_enc = None
        self.label_enc = None


class AtomFeatures:
    """
    This class defines which features of the atom are considered
    hot encoded:
    GetIsAromatic - values: 0,1
    GetAtomicNum
    GetHybridization: values: 0-7
    GetNumRadicalElectrons
    GetTotalDegree
    GetTotalValence
    """

    DEAFULT_FEATURES = "GetIsAromatic,GetAtomicNum,GetTotalNumHs,GetFormalCharge,IsInRing,GetIsAromatic," \
                       "GetHybridization,GetNumRadicalElectrons,GetTotalDegree,GetTotalValence".split(',')

    def __init__(self, features=DEAFULT_FEATURES, ret_pos=True):
        self.features = features
        self.ret_pos = ret_pos

    def __call__(self, atom):
        # type: (Chem.Atom) -> object

        features = [getattr(atom, feature)() for feature in self.features]
        features += [len(atom.GetNeighbors())]
        if self.ret_pos:
            return tuple([[atom.GetIdx()]]), features
        return features


class BondFeatures:
    """
    This class defines which features of the bond are considered
    This features ,ust be multiplied by 2 to be hot encoded and have 1 added to the, to single out no bonds
    hot encoded
    GetBondTypeAsDouble - 0.0 for no bond, 1.0 for SINGLE, 1.5 for AROMATIC, 2.0 for DOUBLE, 3.0 for TRIPLE
    GetIsConjugated
    # numeric:
    # GetValenceContrib - numeric for each atom
    For positioning - tuple:
    GetBeginAtomIdx
    GetEndAtomIdx
    """

    DEFAULT_FEATURES = ['GetBondTypeAsDouble', 'GetIsConjugated']

    def __init__(self, features=DEFAULT_FEATURES, ret_pos=True):
        self.features = features
        self.ret_pos = ret_pos

    def __call__(self, bond):
        # type: (Chem.Bond) -> object

        features = ()
        if self.ret_pos:
            features += tuple([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]])
        features += tuple([[2*getattr(bond, feature)()+1 for feature in self.features]])

        return features


class Graph:
    __metaclass__ = ABCMeta
    TYPE = None

    GraphAttributes = ['afm', 'bfm', 'adj', 'mask', 't_dist']

    @abstractmethod
    def __init__(self):
        for att in self.GraphAttributes:
            setattr(self, att, np.empty(0))
        # self.afm = np.empty(0)
        # self.adj = np.empty(0)
        # self.t_dist = np.empty(0)
        # self.bfm = np.empty(0)
        self.is_encoded = False
        self.label = None

    def encode_afm(self, atom_enc):
        # type: (AtomEncoder) -> None
        self.afm = np.hstack([atom_enc[i].transform(self.afm[:,i]) for i in range(self.afm.shape[-1])])

    def encode_bfm(self, bond_enc):
        # type: (BondEncoder) -> None
        self.bfm = np.concatenate(
            [bond_enc[i].transform(self.bfm.reshape(-1, self.bfm.shape[-1])[:,i]) for i in range(self.bfm.shape[-1])],
            axis=-1).reshape(self.bfm.shape[0:-1] + (-1,))

    def encode(self, atom_enc, bond_enc):
        if not self.is_encoded:
            self.encode_afm(atom_enc)
            self.encode_bfm(bond_enc)
            self.is_encoded = True


class Graph2D(Graph):

    TYPE = '2D'

    def __init__(self):
        super(Graph2D, self).__init__()


class Graph3D(Graph2D):

    TYPE = '3D'
    Graph3DAttributes = ['e_dist']

    def __init__(self, orig=None):
        super(Graph3D, self).__init__()
        for att in self.Graph3DAttributes:
            setattr(self, att, np.empty(0))
        if orig is not None:
            self.copy(orig)
        # self.e_dist = None

    def copy(self, graph3d):
        for att in self.GraphAttributes:
            setattr(self, att, getattr(graph3d, att))
        # self.afm = graph3d.afm
        # self.adj = graph3d.adj
        # self.t_dist = graph3d.t_dist
        # self.bfm = graph3d.bfm


class MolGraph:
    __metaclass__ = ABCMeta

    TYPE = None

    @abstractmethod
    def __init__(self, mol, atom_extractor, bond_extractor, include_hs=False, graph_class=Graph2D):
        # type: (Chem.Atom, AtomFeatures, BondFeatures, bool) -> object
        self.mol = mol
        self.include_hs = include_hs
        if include_hs:
            self.mol = Chem.AddHs(mol)

        self.ae = atom_extractor
        self.be = bond_extractor
        self.graph_class = graph_class
        self.graph = None
        self.encoded = False

    def get_graph(self):
        return self.graph

    def populate_afm(self):
        self.ae.ret_pos = True
        afm = np.empty([self.mol.GetNumAtoms(), len(self.ae.features)+1], dtype=np.int)
        for atom in self.mol.GetAtoms():
            pos, features = self.ae(atom)
            afm[pos] = map(int, features)
        self.graph.afm = afm

    def populate_bfm(self):
        bfm = np.zeros([self.mol.GetNumAtoms(), self.mol.GetNumAtoms(), len(self.be.features) + 2*(len(self.ae.features)+1)], dtype=np.int)
        self.ae.ret_pos = False
        for bond in self.mol.GetBonds():
            pos, features = self.be(bond)
            pos = sorted(pos)
            for p in pos:
                features += self.ae(self.mol.GetAtomWithIdx(p))
            features = map(int, features)
            bfm[tuple(pos)] = features
            bfm[tuple(reversed(pos))] = features
        self.graph.bfm = bfm

    def populate_adj(self):
        self.graph.adj = Chem.rdmolops.GetAdjacencyMatrix(self.mol)

    def populate_t_dist(self):
        self.graph.t_dist = Chem.rdmolops.GetDistanceMatrix(self.mol)

    def create_graph(self):
        self.graph = self.graph_class()
        self.populate_afm()
        self.populate_bfm()
        self.populate_adj()
        self.populate_t_dist()


class Mol2DGraph(MolGraph):

    # A 2D graph is constructed of:
    # 1. atom features matrix of shape no. atoms x atom features
    # 2. adjacency matrix
    # 3. Topological distance matrix
    # 4. bond feature adjacency matrix - of shape no. atoms x no. atoms x bond features

    TYPE = Graph2D.TYPE
    
    def __init__(self, mol, atom_extractor, bond_extractor, include_hs=False, graph_class=Graph2D):
        super(Mol2DGraph, self).__init__(mol, atom_extractor, bond_extractor, include_hs=include_hs, graph_class=graph_class)

    def get_graph(self):
        if self.graph is None:
            self.create_graph()
        return self.graph

    def to_3d(self):

        if not self.include_hs:
            self.mol = AllChem.AddHs(self.mol)
            self.include_hs = True
            self.create_graph()

        if self.graph is None:
            self.create_graph()

        no_rot = Chem.rdMolDescriptors.CalcNumRotatableBonds(self.mol)
        no_conf = Mol3DGraph.calc_no_conf(no_rot)

        filtered_cids = Mol3DGraph.filter_conformers(self.mol, no_conf)

        mol3dgraphs = []
        for cid in filtered_cids:
            mol3dgraphs.append(Mol3DGraph(self, conformer_id=cid))
        return mol3dgraphs


class Mol3DGraph(MolGraph):

    # A 3D graph consists of:
    # 1. atom features matrix of shape no. atoms x atom features
    # 2. adjacency matrix
    # 3. Topological distance matrix
    # 4. bond feature adjacency matrix - of shape no. atoms x no. atoms x bond features
    # 5. 3D distance matrix of shape no. atoms x no. atoms
    # The above is generated for each conformer - so the resulting is of shape: no. conformers x graph

    TYPE = Graph3D.TYPE

    # def __init__(self, mol, atom_extractor, bond_extractor, include_hs=False, graph_class=Graph2D, conformer_id=-1):
    #     super(Mol3DGraph, self).__init__(mol, atom_extractor, bond_extractor, include_hs=include_hs, graph_class=graph_class)
    #     self.conf_id = conformer_id
    def __init__(self, mol2Dgraph, conformer_id=-1):
        # type: (Mol2DGraph, int) -> None
        super(Mol3DGraph, self).__init__(mol2Dgraph.mol, mol2Dgraph.ae, mol2Dgraph.be, mol2Dgraph.include_hs,
                                         graph_class=Graph3D)
        self.mol2Dgraph = mol2Dgraph
        self.conf_id = conformer_id

    def get_graph(self, graph3d=None):
        if self.graph is None:
            self.create_graph(graph3d=graph3d)
        return self.graph

    def create_graph(self, graph3d=None):
        if graph3d is None:
            self.graph = self.graph_class(self.mol2Dgraph.graph)
            # super(Mol3DGraph, self).create_graph()
        else:
            self.graph = self.graph_class(graph3d)
        self.populate_e_dist()

    def populate_e_dist(self):
        self.graph.e_dist = Chem.rdmolops.Get3DDistanceMatrix(self.mol, confId=self.conf_id)

    @staticmethod
    def calc_no_conf(no_rot):
        if no_rot < 8:
            return 50
        if no_rot < 13:
            return 200
        return 300

    @staticmethod
    def filter_conformers(mol, no_conf):

        print no_conf

        cids = AllChem.EmbedMultipleConfs(mol, no_conf, AllChem.ETKDG())
        mol = AllChem.RemoveHs(mol)
        print len(cids)

        def get_rms(mol, c1, c2):
            return AllChem.GetBestRMS(mol, mol, c1, c2)

        cenergy = []
        for conf in cids:
            cenergy.append(AllChem.UFFGetMoleculeForceField(mol, confId=conf).CalcEnergy())
        sortedcids = sorted(cids, key=lambda cid: cenergy[cid])
        processed = []
        for conf in sortedcids:
            passed = True
            for seenconf in processed:
                rms = get_rms(mol, seenconf, conf)
                print (seenconf, conf, rms)
                if rms < 0.35:
                    passed = False
                    break
            if passed:
                processed.append(conf)
        return processed


class MolGraphFactory:

    MolGraphTypes = [Mol2DGraph, Mol3DGraph]

    def __init__(self, type, atom_extractor, bond_extractor, include_hs=False):
        self.type = type
        self.mol_graph_class = Mol2DGraph if type == Mol2DGraph.TYPE else Mol3DGraph
        self.graph_class = Graph2D if type == Graph2D.TYPE else Graph3D
        self.ae = atom_extractor
        self.be = bond_extractor
        self.in_hs = include_hs

    def prep_graph(self, mol):
        return self.mol_graph_class(mol, self.ae, self.be, include_hs=self.in_hs, graph_class=self.graph_class)


__all__ = ['MolGraphFactory', 'Mol2DGraph', 'Mol3DGraph', 'MolGraph', 'AtomFeatures', 'BondFeatures', 'Graph',
           'Graph2D', 'Graph3D', 'GraphEncoder']


