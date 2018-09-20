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

from models.lipo_basic_model import BasicModel, _DEF_STEPS
from models.graph_norm_wrapper import GraphWrapper
from mol_graph import *
from mol_graph import GraphEncoder
from pre_process.data_loader import GraphDataSet, collate_2d_graphs
from pre_process.load_dataset import load_classification_dataset
from mpnn_functions.encoders.c_autoencoder import AutoEncoder
import tqdm
from torch.nn import functional as F
from pre_process.utils import from_numpy


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
    tot_loss = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            output = model(batch).squeeze()
            loss = aff_criterion(output, batch['affs'])
            tot_loss += loss.item() * batch['a_bfm'].shape[0]
            labels.extend(output.ge(_NEG_CUTOFF).cpu().data.numpy().tolist())
            true_labels.extend(batch['labels'].cpu().data.numpy().tolist())
    return tot_loss/len(dataset.dataset), metrics.accuracy_score(true_labels, labels)

def test_model_class(model, dataset):
    model.eval()
    labels = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            labels = labels + model(batch).max(dim=-1)[1].cpu().data.numpy().tolist()
            true_labels = true_labels + batch['labels'].cpu().data.numpy().tolist()
    return metrics.accuracy_score(true_labels, labels)


def loss_func(pred, label, aff):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    batch_size = pred.shape[0]
    pred = F.threshold(pred, _NEG_CUTOFF, 0)
    return F.l1_loss(pred, label)
    # neg_aff = from_numpy(np.array([_NEG_AFF] * batch_size))
    # pos_loss = F.mse_loss(pred.mul(label), aff.mul(label), reduction='sum')
    # non_binders = pred.mul(1 - label)
    # non_mask = (non_binders >= _NEG_CUTOFF).float()
    # neg_loss = F.mse_loss(non_binders.mul(non_mask), neg_aff.mul(non_mask), reduction='sum')
    # return (neg_loss + pos_loss)/float(batch_size)
    # return F.mse_loss(pred, aff)


_NEG_AFF = np.float32(0.0)
_NEG_CUTOFF = 5
seed = 317
torch.manual_seed(seed)
data_file = sys.argv[1]
target = 0

mgf = MolGraphFactory(Mol2DGraph.TYPE, AtomFeatures(), BondFeatures())
# try:
#     pass
    # file_data = np.load(data_file+'.npz')
    # data = file_data['data']
    # no_labels = file_data['no_labels']
    # all_labels = file_data['all_labels']
    # file_data.close()
# except IOError:
data, no_labels, all_labels = load_classification_dataset(data_file+'.csv', 'InChI', Chem.MolFromInchi, mgf,
                                                          'Gene_Symbol', 'pXC50')
for graph in data:
    graph.mask = np.ones(graph.a_bfm.shape[0], dtype=np.float32).reshape(graph.a_bfm.shape[0], 1)
    graph.bfm = graph.bfm.astype(np.long)
    graph.a_bfm = graph.a_bfm.astype(np.long)
    graph.adj = graph.adj.astype(np.float32)
    graph.aff = np.float32(graph.aff if target == graph.label else 0)
    graph.label = np.float32(target == graph.label)
graph_encoder = GraphEncoder()
    # with open('basic_model_graph_encoder.pickle', 'wb') as out:
    #     pickle.dump(graph_encoder, out)
    # np.savez_compressed(data_file, data=data, no_labels=no_labels, all_labels=all_labels)


model_attributes = {
    'afm': 8,
    'bfm': sum(None != graph_encoder.bond_enc[0].classes_),
    'a_bfm': sum(None != graph_encoder.a_bond_enc[0].classes_),
    'mfm': 8,
    'adj': 1,
    'out': 8*(_DEF_STEPS+1),
    'classification_output': 1
}
den = int(model_attributes['out'])
dense_layer = []
for i in range(50):
    new_den = int(np.floor(den/2))
    if new_den < 32:
        break
    dense_layer.append(nn.Linear(den, new_den))
    dense_layer.append(nn.ReLU())
    den = new_den
dense_layer.append(nn.Linear(den, model_attributes['classification_output']))




aff_model = nn.Sequential(
    GraphWrapper(BasicModel(model_attributes['afm'], model_attributes['bfm'],model_attributes['a_bfm'], model_attributes['mfm'],
                            model_attributes['adj'], model_attributes['out'])),
    nn.BatchNorm1d(model_attributes['out']),
    nn.Sequential(*dense_layer)
)
cls_model = nn.Sequential(
    GraphWrapper(BasicModel(model_attributes['afm'], model_attributes['bfm'],model_attributes['a_bfm'], model_attributes['mfm'],
                            model_attributes['adj'], model_attributes['out'])),
    nn.BatchNorm1d(model_attributes['out']),
    nn.Sequential(*dense_layer)
)

aff_model.float()
aff_model.apply(BasicModel.init_weights)
cls_model.float()
cls_model.apply(BasicModel.init_weights)

print "Model has: {} parameters".format(count_model_params(aff_model))
print model_attributes
if torch.cuda.is_available():
    aff_model.cuda()
    cls_model.cuda()

cls_criterion = nn.BCEWithLogitsLoss()
aff_criterion = nn.MSELoss()
aff_optimizer = optim.Adam(aff_model.parameters(), lr=1e-4, weight_decay=1e-4)
aff_model.train()
cls_optimizer = optim.Adam(cls_model.parameters(), lr=1e-4, weight_decay=1e-4)
cls_model.train()

train, test, train_labels, _ = train_test_split(data, all_labels, test_size=0.1, random_state=seed, stratify=all_labels)
del data
del all_labels
train, val = train_test_split(train, test_size=0.1, random_state=seed, stratify=train_labels)
train = GraphDataSet(train)
val = GraphDataSet(val)
test = GraphDataSet(test)
train = DataLoader(train, 256, shuffle=True, collate_fn=collate_2d_graphs)
val = DataLoader(val, 256, shuffle=True, collate_fn=collate_2d_graphs)
test = DataLoader(test, 256, shuffle=True, collate_fn=collate_2d_graphs)


epoch_losses = []
break_con = False
for epoch in tqdm.trange(1000):
    aff_model.train()
    cls_model.train()
    for batch in tqdm.tqdm(train):
        aff_optimizer.zero_grad()
        cls_optimizer.zero_grad()
        cls_out = cls_model(batch).squeeze()
        cls_loss = cls_criterion(cls_out, batch['labels'])
        cls_out.detach()
        aff_loss = aff_criterion(aff_model(batch).squeeze().mul(cls_out), batch['affs'])
        cls_loss.backward()
        cls_optimizer.step()
        aff_loss.backward()
        aff_optimizer.step()
    # t_mse, t_acc = test_model(model, val)
    mse, acc = test_model(aff_model, val)
    # tqdm.tqdm.write(
    #     "epoch {} loss: {}, Train MSE: {}, Train ACC: {}, Val MSE: {}, Val ACC: {}.".format(epoch, epoch_loss/len(train.dataset), t_mse, t_acc, mse, acc))
    tqdm.tqdm.write(
        "epoch {} Val MSE: {}, Val ACC: {}.".format(epoch, mse, acc))
    # if mse < 1.55:
    #     break
    # if not np.isnan(f1) and f1 > 0.8:
    #     save_model(model, 'epoch_'+str(epoch), model_attributes, {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1})

mse, acc = test_model(aff_model, test)
tqdm.tqdm.write("Testing MSE: {}, ACC: {}".format(mse, acc))

