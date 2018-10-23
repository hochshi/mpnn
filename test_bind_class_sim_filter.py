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
from pre_process.data_loader import GraphDataSet, collate_2d_graphs, collate_2d_tensors
from pre_process.load_dataset import load_classification_dataset
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
    torch.save(model.state_dict(), str(model_name) + '.state_dict')
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
            output = model(batch)
            loss = criterion(output, batch['labels'])
            tot_loss += loss.item() * batch['afm'].shape[0]
            labels.extend(output.max(dim=-1)[1].cpu().data.numpy().tolist())
            true_labels.extend(batch['labels'].cpu().data.numpy().tolist())
    pre = metrics.precision_score(true_labels, labels, average='weighted')
    rec = metrics.recall_score(true_labels, labels, average='weighted')
    f1 = 2*pre*rec/(pre + rec + 1e-8)
    return tot_loss/len(dataset.dataset), (pre, rec, f1)

def test_model_class(model, dataset):
    model.eval()
    labels = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            labels = labels + model(batch).max(dim=-1)[1].cpu().data.numpy().tolist()
            true_labels = true_labels + batch['labels'].cpu().data.numpy().tolist()
    return metrics.accuracy_score(true_labels, labels)


try:
    seed = int(sys.argv[2])
except KeyError:
    seed = 317

torch.manual_seed(seed)
data_file = sys.argv[1]

mgf = MolGraphFactory(Mol2DGraph.TYPE, AtomFeatures(), BondFeatures())
# try:
#     pass
    # file_data = np.load(data_file+'.npz')
    # data = file_data['data']
    # no_labels = file_data['no_labels']
    # all_labels = file_data['all_labels']
    # file_data.close()
# except IOError:
data, no_labels, all_labels = load_classification_dataset(data_file+'.csv', 'InChI', Chem.MolFromInchi, mgf, 'target')
for graph in data:
    graph.mask = np.ones(graph.afm.shape[0], dtype=np.float32).reshape(graph.afm.shape[0], 1)
    graph.afm = graph.afm.astype(np.float32)
    graph.bfm = graph.bfm.astype(np.long)
    graph.a_bfm = graph.a_bfm.astype(np.long)
    graph.adj = graph.adj.astype(np.float32)
    graph.label = long(graph.label)
graph_encoder = GraphEncoder()
    # with open('basic_model_graph_encoder.pickle', 'wb') as out:
    #     pickle.dump(graph_encoder, out)
    # np.savez_compressed(data_file, data=data, no_labels=no_labels, all_labels=all_labels)

data, all_labels, no_labels = filter_dataset(data, all_labels, 50)

# ae = AutoEncoder(data[0].afm.shape[-1])
# be = AutoEncoder(data[0].bfm.shape[-1])

den = int(2*data[0].afm.shape[-1])
dense_layer = []
# while den > 10:
#     new_den = int(np.ceil(den/2))
#     dense_layer.append(nn.Linear(den, new_den))
#     dense_layer.append(nn.ReLU())
#     den = new_den
for i in range(50):
    new_den = 2 * den
    if new_den > no_labels:
        break
    dense_layer.append(nn.Linear(den, new_den))
    dense_layer.append(nn.ReLU())
    den = new_den
dense_layer.append(nn.Linear(den, no_labels))

bfm = int(sum(None != graph_encoder.bond_enc[0].classes_))
a_bfm = int(sum(None != graph_encoder.a_bond_enc[0].classes_))
afm = int(np.ceil(a_bfm ** 0.25))
mfm = int(np.ceil((a_bfm * bfm) ** 0.25))
# mfm = int(np.ceil(no_labels/2))

model_attributes = {
    'afm': afm,
    'bfm': bfm,
    'a_bfm': a_bfm,
    'mfm': mfm,
    'adj': 1,
    'out': mfm*2,
    'classification_output': no_labels
}


model = nn.Sequential(
    GraphWrapper(BasicModel(model_attributes['afm'], model_attributes['bfm'],model_attributes['a_bfm'], model_attributes['mfm'],
                            model_attributes['adj'], model_attributes['out'])),
    nn.BatchNorm1d(model_attributes['out']),
    nn.Linear(model_attributes['out'], model_attributes['classification_output'])
    # nn.Sequential(*dense_layer)
)

model.float()
model.apply(BasicModel.init_weights)

print "Model has: {} parameters".format(count_model_params(model))
print model
print model_attributes
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
model.train()

train, test, train_labels, _ = train_test_split(data, all_labels, test_size=0.1, random_state=seed, stratify=all_labels)
del data
del all_labels
train, val = train_test_split(train, test_size=0.1, random_state=seed, stratify=train_labels)
train = GraphDataSet(train)
val = GraphDataSet(val)
test = GraphDataSet(test)
train = DataLoader(train, 32, shuffle=True, collate_fn=collate_2d_graphs)
val = DataLoader(val, 32, shuffle=True, collate_fn=collate_2d_graphs)
test = DataLoader(test, 32, shuffle=True, collate_fn=collate_2d_graphs)


epoch_losses = []
break_con = False
best_val_f1 = 0
for epoch in tqdm.trange(1000):
    model.train()
    epoch_loss = 0
    for batch in tqdm.tqdm(train):
        model.zero_grad()
        loss = criterion(model(batch), batch['labels'])
        epoch_loss += loss.item() * batch['afm'].shape[0]
        loss.backward()
        optimizer.step()
    epoch_losses.append(epoch_loss)
    t_mse = test_model(model, train)
    mse = test_model(model, val)
    # acc = test_model_class(model, val)
    tqdm.tqdm.write(
        "epoch {} loss: {} Train MSE: {} Val MSE: {}".format(epoch, epoch_loss/len(train.dataset), t_mse, mse))
    if mse[1][2] > best_val_f1:
        best_val_f1 = mse[1][2]
        save_model(model, 'filter_model_best_loss', model_attributes, {})

model.load_state_dict(torch.load('filter_model_best_loss.state_dict', map_location=lambda storage, loc: storage))
model.cuda()
mse = test_model(model, test)
tqdm.tqdm.write("Testing MSE: {}".format(mse))
acc = test_model_class(model, val)
tqdm.tqdm.write("Val acc: {}".format(acc))
acc = test_model_class(model, test)
tqdm.tqdm.write("Test acc: {}".format(acc))
