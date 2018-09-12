import torch
from torch.autograd import Variable
from torch.utils import data
from torch.nn import functional as F
from typing import List
import numpy as np
from utils import from_numpy

from mol_graph import Graph, Graph2D


def embed_arr(arr, dims):
    new_arr = np.zeros(dims, dtype=np.float32)
    new_arr[:arr.shape[0], :arr.shape[1]] = arr
    return new_arr


def create_mask(arr_dims, dims):
    mask = np.zeros(dims, dtype=np.float32)
    mask[:arr_dims[0],:] = 1
    return mask


def collate_2d_tensors(graphs):
    max_graph = np.argmax([graph.afm.shape[0] for graph in graphs])
    max_size = np.max([graph.afm.shape[0] for graph in graphs])
    # afm shape is nodes x features
    afms = [F.pad(graph.afm, (0, 0, 0, max_size - graph.afm.shape[0]), "constant", 0).unsqueeze(0) for graph in graphs]
    afms = torch.cat(afms, dim=0)
    # bfm shape is nodes x nodes x features
    bfms = [F.pad(graph.bfm, (0, 0, 0, max_size - graph.afm.shape[0], 0, max_size - graph.afm.shape[0]), "constant", 0).unsqueeze(0) for graph in graphs]
    bfms = torch.cat(bfms, dim=0)
    # adj shape is nodes x nodes
    adjs = [F.pad(graph.adj, (0, max_size - graph.afm.shape[0], 0, max_size - graph.afm.shape[0]), "constant", 0).unsqueeze(0) for graph in graphs]
    adjs = torch.cat(adjs, dim=0)
    # afm_mask shape is nodes x 1
    afm_masks = [F.pad(graph.mask, (0, 0, 0, max_size - graph.afm.shape[0]), "constant", 0).unsqueeze(0) for graph in graphs]
    afm_masks = torch.cat(afm_masks, dim=0)
    labels = from_numpy(np.array([graph.label for graph in graphs]))

    return {
        'afm': Variable(afms),
        'bfm': Variable(bfms),
        'adj': Variable(adjs),
        'mask': Variable(afm_masks),
        'labels': Variable(labels)
    }


# def collate_2d_graphs(graphs):
#     # type: (List[Graph2D]) -> object
#     max_size = np.argmax([graph.afm.shape[0] for graph in graphs])
#     max_dims = graphs[max_size].afm.shape
#     afms = np.array([embed_arr(graph.afm, max_dims) for graph in graphs])
#     nafms = np.array([embed_arr(graph.nafm, graphs[max_size].nafm.shape) for graph in graphs])
#     bfms = np.array([embed_arr(graph.bfm, graphs[max_size].bfm.shape) for graph in graphs])
#     adjs = np.array([embed_arr(graph.adj, graphs[max_size].adj.shape) for graph in graphs])
#     # t_dists = np.array([embed_arr(graph.t_dist, graphs[max_size].t_dist.shape) for graph in graphs])
#     afm_masks = np.array([create_mask(graph.afm.shape[:-1]+(1,), max_dims[:-1] + (1,)) for graph in graphs])
#     labels = np.array([graph.label for graph in graphs])
#
#     return {
#         'afm': Variable(from_numpy(afms)),
#         'nafm': Variable(from_numpy(nafms)),
#         'bfm': Variable(from_numpy(bfms)),
#         'adj': Variable(from_numpy(adjs)),
#         # 't_dist': Variable(from_numpy(t_dists)),
#         'mask': Variable(from_numpy(afm_masks)),
#         'labels': Variable(from_numpy(labels))
#         }

def collate_2d_graphs(graphs):
    # type: (List[Graph2D]) -> object
    max_size = np.argmax([graph.afm.shape[0] for graph in graphs])
    max_dims = graphs[max_size].afm.shape
    afms = np.array([embed_arr(graph.afm, max_dims) for graph in graphs])
    nafms = np.array([embed_arr(graph.nafm, graphs[max_size].nafm.shape) for graph in graphs])
    bfms = np.array([embed_arr(graph.bfm, graphs[max_size].bfm.shape) for graph in graphs])
    adjs = np.array([embed_arr(graph.adj, graphs[max_size].adj.shape) for graph in graphs])
    # t_dists = np.array([embed_arr(graph.t_dist, graphs[max_size].t_dist.shape) for graph in graphs])
    afm_masks = np.array([create_mask(graph.afm.shape[:-1]+(1,), max_dims[:-1] + (1,)) for graph in graphs])
    labels = np.array([graph.label for graph in graphs])

    return {
        'afm': Variable(from_numpy(afms)),
        'nafm': Variable(from_numpy(nafms)),
        'bfm': Variable(from_numpy(bfms).long()),
        'adj': Variable(from_numpy(adjs)),
        # 't_dist': Variable(from_numpy(t_dists)),
        'mask': Variable(from_numpy(afm_masks)),
        'labels': Variable(from_numpy(labels))
        }

def collate_2d_ecfp_graphs(graphs):
    # type: (List[Graph2D]) -> object
    max_size = np.argmax([graph.afm.shape[0] for graph in graphs])
    max_dims = graphs[max_size].afm.shape
    afms = np.array([embed_arr(graph.afm, max_dims) for graph in graphs])
    bfms = np.array([embed_arr(graph.bfm, graphs[max_size].bfm.shape) for graph in graphs])
    adjs = np.array([embed_arr(graph.adj, graphs[max_size].adj.shape) for graph in graphs])
    afm_masks = np.array([create_mask(graph.afm.shape[:-1]+(1,), max_dims[:-1] + (1,)) for graph in graphs])
    labels = np.array([embed_arr(graph.label, graphs[max_size].label.shape) for graph in graphs])

    return {
        'afm': Variable(from_numpy(afms)),
        'bfm': Variable(from_numpy(bfms)),
        'adj': Variable(from_numpy(adjs)),
        'mask': Variable(from_numpy(afm_masks)),
        'labels': Variable(from_numpy(labels)).float()
        }


class GraphDataSet(data.Dataset):

    def __init__(self, dataset):
        # type: (List[Graph]) -> object
        super(GraphDataSet, self).__init__()
        self.data = dataset
        self.len = len(dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index]


__all__ = ['collate_2d_graphs', 'GraphDataSet']