import cPickle as pickle
import sys

import numpy as np
import torch
import torch.cuda
from rdkit import Chem
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models.lipo_basic_model import BasicModel
from models.graph_model_wrapper import GraphWrapper
from mol_graph import *
from mol_graph import GraphEncoder
from pre_process.data_loader import GraphDataSet, collate_2d_graphs, collate_2d_tensors
from pre_process.load_dataset import load_number_dataset
from mpnn_functions.encoders.c_autoencoder import AutoEncoder
import tqdm


def filter_dataset(data, labels, lower_cutoff=None, upper_cutoff=None, count_cutoff=None):
    uniq, count = np.unique(labels, return_counts=True)
    if lower_cutoff is not None:
        uniq_mask = count > lower_cutoff
    if upper_cutoff is not None:
        uniq_mask = np.logical_and(uniq_mask, count < upper_cutoff)
    if count_cutoff is not None:
        positive = np.argwhere(uniq_mask).reshape(-1)[:4]
        uniq_mask = np.zeros_like(uniq, dtype=np.bool)
        uniq_mask[positive] = True

    label_mask = np.isin(labels, uniq[uniq_mask])
    new_label_dict = dict(zip(uniq[uniq_mask], range(uniq_mask.sum())))
    filtered_dataset = []
    new_labels = []
    for graph, cond, label in zip(data, label_mask, labels):
        if cond:
            new_label = new_label_dict[label]
            new_labels.append(new_label)
            graph.label = new_label
            filtered_dataset.append(graph)
    return filtered_dataset, new_labels, uniq_mask.sum()

def count_model_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def save_model(model, model_name, model_att, model_metrics):
    # type: (nn.Module, dict) -> None
    torch.save(model.state_dict(), 'basic_model' + str(model_name) + '.state_dict')
    with open('basic_model_attributes.pickle', 'wb') as out_file:
        pickle.dump(model_att, out_file)
    with open('basic_model_' + str(model_name) + '_stats.pickle', 'wb') as out_file:
        pickle.dump(model_metrics, out_file)


def test_model(model, dataset):
    model.eval()
    labels = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            labels = labels + model(batch).squeeze().cpu().data.numpy().tolist()
            true_labels = true_labels + batch['labels'].cpu().data.numpy().tolist()
    return metrics.mean_squared_error(true_labels, labels)

seed = 317
torch.manual_seed(seed)
data_file = sys.argv[1]

mgf = MolGraphFactory(Mol2DGraph.TYPE, AtomFeatures(), BondFeatures())
try:
    file_data = np.load(data_file+'.npz')
    data = file_data['data']
    no_labels = file_data['no_labels']
    all_labels = file_data['all_labels']
    file_data.close()
except IOError:
    data, no_labels, all_labels = load_number_dataset(data_file+'.csv', 'smiles', Chem.MolFromSmiles, mgf, 'exp')
    for graph in data:
        graph.mask = np.ones(graph.afm.shape[0], dtype=np.float32).reshape(graph.afm.shape[0], 1)
        graph.afm = graph.afm.astype(np.float32)
        graph.bfm = graph.bfm.astype(np.float32)
        graph.adj = graph.adj.astype(np.float32)
        graph.label = float(graph.label)
    graph_encoder = GraphEncoder()
    with open('basic_model_graph_encoder.pickle', 'wb') as out:
        pickle.dump(graph_encoder, out)
    np.savez_compressed(data_file, data=data, no_labels=no_labels, all_labels=all_labels)


ae = AutoEncoder(data[0].afm.shape[-1])
be = AutoEncoder(data[0].bfm.shape[-1])
model_attributes = {
    'afm': ae.out_f,
    'bfm': be.out_f,
    'mfm': data[0].afm.shape[-1],
    'adj': data[0].adj.shape[-1],
    'out': 2*data[0].afm.shape[-1],
    'classification_output': 1
}


model = nn.Sequential(
    GraphWrapper(BasicModel(model_attributes['afm'], model_attributes['bfm'], model_attributes['mfm'],
                            model_attributes['adj'], model_attributes['out'],
                            atom_encoder=ae.encoder, bond_encoder=be.encoder)),
    nn.BatchNorm1d(model_attributes['out']),
    nn.Linear(model_attributes['out'], model_attributes['classification_output'])
)

model.float()
model.apply(BasicModel.init_weights)

print "Model has: {} parameters".format(count_model_params(model))
if torch.cuda.is_available():
    model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
model.train()

train, test = train_test_split(data, test_size=0.1, random_state=seed)
del data
del all_labels
train, val = train_test_split(train, test_size=0.1, random_state=seed)
train = GraphDataSet(train)
val = GraphDataSet(val)
test = GraphDataSet(test)
train = DataLoader(train, 128, shuffle=True, collate_fn=collate_2d_graphs)
val = DataLoader(val, 128, shuffle=False, collate_fn=collate_2d_graphs)
test = DataLoader(test, 128, shuffle=False, collate_fn=collate_2d_graphs)


epoch_losses = []
break_con = False
for epoch in tqdm.trange(1000):
    model.train()
    epoch_loss = 0
    for batch in tqdm.tqdm(train):
        batch['labels'] = batch['labels'].float().unsqueeze(-1)
        model.zero_grad()
        loss = criterion(model(batch), batch['labels'])
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_losses.append(epoch_loss)
    mse = test_model(model, train)
    tqdm.tqdm.write(
        "epoch {} training loss: {}, MSE: {}".format(epoch, epoch_loss, mse))
    mse = test_model(model, train)
    tqdm.tqdm.write(
        "epoch {} loss: {} validation MSE: {}".format(epoch, epoch_loss, mse))
    # if not np.isnan(f1) and f1 > 0.8:
    #     save_model(model, 'epoch_'+str(epoch), model_attributes, {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1})

acc, pre, rec = test_model(model, test)
f1 = 2 * (pre * rec) / (pre + rec)
tqdm.tqdm.write(
    "Testing acc: {}, pre: {}, rec: {}, F1: {}".format(acc, pre, rec, f1))