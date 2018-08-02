import numpy as np
import pandas as pd
from pre_process import MolGraphFactory, MolGraph, Graph, GraphEncoder
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from typing import List, Tuple


def generate_molgraphs(mol_strs, text2molfunc, mol_graph_factory):
    # type: (np.array, function, MolGraphFactory) -> List[MolGraph]
    m2gs = []
    for mol_str in mol_strs:
        mol = text2molfunc(mol_str)
        if mol is None:
            continue
        m2g = mol_graph_factory.prep_graph(mol)
        m2g.create_graph()
        m2gs.append(m2g)
    return m2gs


def encode_molgraphs(m2gs):
    # type: (List[MolGraph]) -> None
    graph_encoder = GraphEncoder()
    graph_encoder.atom_enc = build_atom_enc(m2gs)
    graph_encoder.bond_enc = build_bond_enc(m2gs)

    for m2g in m2gs:
        m2g.graph.encode(graph_encoder.atom_enc, graph_encoder.bond_enc)
    return m2gs


def build_atom_enc(m2gs):
    all_afms = np.vstack([m2g.graph.afm for m2g in m2gs])
    atom_encs = []
    for i in range(all_afms.shape[1]):
        atom_enc = LabelBinarizer()
        atom_enc.fit(all_afms[:, i])
        atom_encs.append(atom_enc)
    return atom_encs


def build_bond_enc(m2gs):
    bond_features = m2gs[0].graph.bfm.shape[-1]
    all_bfms = np.vstack([m2g.graph.bfm.reshape(-1, bond_features) for m2g in m2gs])
    bond_encs = []
    for i in range(all_bfms.shape[1]):
        bond_enc = LabelBinarizer()
        bond_enc.fit(all_bfms[:, i])
        bond_encs.append(bond_enc)
    return bond_encs


def load_classification_dataset(file_name, moltext_colname, text2molfunc, mol_graph_factory, label_colname):
    # type: (str, str, function, MolGraphFactory, str) -> Tuple(List[Graph], int)
    df = pd.read_csv(file_name)
    graphs = [m2g.graph for m2g in encode_molgraphs(generate_molgraphs(df[moltext_colname].values, text2molfunc, mol_graph_factory))]
    max_label = np.NINF
    le = LabelEncoder()
    encoded_labels = le.fit_transform(df[label_colname].values)
    for graph, label in zip(graphs, encoded_labels):
        graph.label = label
        max_label = label if label > max_label else max_label
    graph_encoder = GraphEncoder()
    graph_encoder.label_enc = le
    return graphs, (max_label+1), encoded_labels

__all__ = ['load_classification_dataset']
