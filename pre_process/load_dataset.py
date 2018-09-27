import numpy as np
import pandas as pd
from mol_graph import MolGraphFactory, MolGraph, Graph, GraphEncoder, AtomFeatures, BondFeatures
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from typing import List, Tuple
from rdkit.Chem import AllChem
from utils import choose_largest_fragment
import rdkit
from rdkit import Chem


def generate_molgraphs(mol_strs, labels, text2molfunc, mol_graph_factory):
    # type: (np.array, np.array, function, MolGraphFactory) -> List[MolGraph]
    m2gs = []
    for mol_str, label in zip(mol_strs, labels):
        mol = text2molfunc(mol_str)
        if mol is None:
            continue
        AllChem.SanitizeMol(mol)
        # mol = choose_largest_fragment(mol)
        m2g = mol_graph_factory.prep_graph(mol)
        m2g.create_graph()
        m2g.graph.label = label
        m2gs.append(m2g)
    return m2gs

def generate_affinity_molgraphs(mol_strs, labels, text2molfunc, mol_graph_factory, affinities):
    # type: (np.array, function, MolGraphFactory) -> List[MolGraph]
    m2gs = []
    for mol_str, label, affinity in zip(mol_strs, labels, affinities):
        mol = text2molfunc(mol_str)
        if mol is None:
            continue
        AllChem.SanitizeMol(mol)
        # mol = choose_largest_fragment(mol)
        m2g = mol_graph_factory.prep_graph(mol)
        m2g.create_graph()
        m2g.graph.label = label
        m2g.graph.affinity = affinity
        m2gs.append(m2g)
    return m2gs


def encode_molgraphs(m2gs):
    # type: (List[MolGraph]) -> List[MolGraph]
    graph_encoder = GraphEncoder()
    if graph_encoder.atom_enc is None:
        atom_enc = build_atom_enc(m2gs)
        graph_encoder.atom_enc = atom_enc
    if graph_encoder.bond_enc is None:
        graph_encoder.bond_enc = build_bond_enc(m2gs)
    # if graph_encoder.a_bond_enc is None:
    #     graph_encoder.a_bond_enc = build_a_bond_enc(m2gs)

    for m2g in m2gs:
        m2g.graph.encode(graph_encoder)
    return m2gs


def build_atom_enc(m2gs):
    all_afms = np.vstack([m2g.graph.afm for m2g in m2gs])
    atom_encs = []
    for i in AtomFeatures.HOT_FEATURES:
        atom_enc = LabelBinarizer()
        atom_enc.fit(all_afms[:, i])
        atom_encs.append((i, atom_enc))
    atom_encs.append((AtomFeatures.BOOL_FEATURES, None))
    return atom_encs


def build_bond_enc(m2gs):
    # bond_features = m2gs[0].graph.bfm.shape[-1]
    # all_bfms = np.concatenate([m2g.graph.bfm.reshape(-1) for m2g in m2gs])
    # mask = 1 == np.vstack([m2g.graph.adj.reshape(-1, 1) for m2g in m2gs]).reshape(-1)
    # le = LabelEncoder()
    # le.fit(all_bfms)
    # return [le]
    bond_encs = []
    all_bfms = np.vstack([m2g.graph.bfm.reshape(-1, len(BondFeatures.DEFAULT_FEATURES)) for m2g in m2gs])
    mask = 1 == np.vstack([m2g.graph.adj.reshape(-1, 1) for m2g in m2gs]).reshape(-1)
    for i in BondFeatures.HOT_FEATURES:
        bond_enc = LabelBinarizer()
        bond_enc.fit(all_bfms[mask, i])
        bond_encs.append((i, bond_enc))
    bond_encs.append((BondFeatures.BOOL_FEATURES, None))
    return bond_encs

def build_a_bond_enc(m2gs):
    # bond_features = m2gs[0].graph.bfm.shape[-1]
    all_bfms = np.concatenate([m2g.graph.a_bfm.reshape(-1) for m2g in m2gs])
    # mask = 1 == np.vstack([m2g.graph.adj.reshape(-1, 1) for m2g in m2gs]).reshape(-1)
    le = LabelEncoder()
    le.fit(all_bfms)
    return [le]


def load_classification_dataset(file_name, moltext_colname, text2molfunc, mol_graph_factory, label_colname):
    # type: (str, str, function, MolGraphFactory, str) -> Tuple(List[Graph], int)
    df = pd.read_csv(file_name)
    graphs = [m2g.graph for m2g in
              encode_molgraphs(
                  generate_molgraphs(
                      df[moltext_colname].values, df[label_colname].values, text2molfunc, mol_graph_factory
                  )
              )]
    labels = [graph.label for graph in graphs]
    max_label = np.NINF

    graph_encoder = GraphEncoder()
    if graph_encoder.label_enc is None:
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        graph_encoder.label_enc = le
    else:
        encoded_labels = graph_encoder.label_enc.transform(labels)
    for graph, label in zip(graphs, encoded_labels):
        graph.label = label
        max_label = label if label > max_label else max_label

    return graphs, (max_label+1), encoded_labels


def ecfp_bits(mol, nbits=16384):
    # type: (Chem.Mol, int) -> np.core.multiarray
    arr = np.zeros((mol.GetNumAtoms(), nbits), dtype=np.float32)
    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=nbits, bitInfo=info)
    for col, positions in info.iteritems():
        for pos, _ in positions:
            arr[pos, col] = 1
    return arr


def load_ecfp_dataset(file_name, moltext_colname, text2molfunc, mol_graph_factory, label_colname):
    df = pd.read_csv(file_name)
    m2gs = encode_molgraphs(
                  generate_molgraphs(
                      df[moltext_colname].values, df[label_colname].values, text2molfunc, mol_graph_factory
                  )
              )
    for m2g in m2gs:
        m2g.graph.label = ecfp_bits(m2g.mol)
    return [m2g.graph for m2g in m2gs]


def load_affinity_dataset(file_name, moltext_colname, text2molfunc, mol_graph_factory, label_colname,
                          affinity_col):
    df = pd.read_csv(file_name)
    graphs = [m2g.graph for m2g in encode_molgraphs(
        generate_affinity_molgraphs(
            df[moltext_colname].values, df[label_colname].values, text2molfunc, mol_graph_factory,
            df[affinity_col].values))]

    labels = [graph.label for graph in graphs]
    max_label = np.NINF

    graph_encoder = GraphEncoder()
    if graph_encoder.label_enc is None:
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        graph_encoder.label_enc = le
    else:
        encoded_labels = graph_encoder.label_enc.transform(labels)
    for graph, label in zip(graphs, encoded_labels):
        graph.label = label
        max_label = label if label > max_label else max_label

    return graphs, (max_label + 1), encoded_labels


def load_number_dataset(file_name, moltext_colname, text2molfunc, mol_graph_factory, label_colname):
    df = pd.read_csv(file_name)
    graphs = [m2g.graph for m2g in encode_molgraphs(
        generate_molgraphs(
            df[moltext_colname].values, df[label_colname].values, text2molfunc, mol_graph_factory,
            ))]

    return graphs, None, None



__all__ = ['load_classification_dataset']
