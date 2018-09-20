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
    mask = [slice(0, i) for i in arr.shape]
    new_arr[tuple(mask)] = arr
    return new_arr


def create_mask(arr_dims, dims):
    mask = np.zeros(dims, dtype=np.float32)
    mask[:arr_dims[0],:] = 1
    return mask


def collate_2d_graphs(graphs):
    # type: (List[Graph2D]) -> object
    max_size = np.argmax([graph.a_bfm.shape[0] for graph in graphs])
    bfms = np.array([embed_arr(graph.bfm, graphs[max_size].bfm.shape) for graph in graphs])
    a_bfms = np.array([embed_arr(graph.a_bfm, graphs[max_size].a_bfm.shape) for graph in graphs])
    adjs = np.array([embed_arr(graph.adj, graphs[max_size].adj.shape) for graph in graphs])
    afm_masks = np.array([create_mask(graph.a_bfm.shape+(1,), graphs[max_size].a_bfm.shape + (1,)) for graph in graphs])
    labels = np.array([graph.label for graph in graphs])
    affs = np.array([graph.aff for graph in graphs])

    return {
        'bfm': Variable(from_numpy(bfms).long()),
        'a_bfm': Variable(from_numpy(a_bfms).long()),
        'adj': Variable(from_numpy(adjs)),
        'mask': Variable(from_numpy(afm_masks)),
        'labels': Variable(from_numpy(labels)),
        'affs': Variable(from_numpy(affs))
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