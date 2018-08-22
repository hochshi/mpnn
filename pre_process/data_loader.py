from torch.autograd import Variable
from torch.utils import data
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


def collate_2d_graphs(graphs):
    # type: (List[Graph2D]) -> object
    max_size = np.argmax([graph.afm.shape[0] for graph in graphs])
    max_dims = graphs[max_size].afm.shape
    afms = np.array([embed_arr(graph.afm, max_dims) for graph in graphs])
    bfms = np.array([embed_arr(graph.bfm, graphs[max_size].bfm.shape) for graph in graphs])
    adjs = np.array([embed_arr(graph.adj, graphs[max_size].adj.shape) for graph in graphs])
    # t_dists = np.array([embed_arr(graph.t_dist, graphs[max_size].t_dist.shape) for graph in graphs])
    afm_masks = np.array([create_mask(graph.afm.shape[:-1]+(1,), max_dims[:-1] + (1,)) for graph in graphs])
    labels = np.array([graph.label for graph in graphs])

    return {
        'afm': Variable(from_numpy(afms)),
        'bfm': Variable(from_numpy(bfms)),
        'adj': Variable(from_numpy(adjs)),
        # 't_dist': Variable(from_numpy(t_dists)),
        'mask': Variable(from_numpy(afm_masks)),
        'labels': Variable(from_numpy(labels))
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