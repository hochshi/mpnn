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

from models.basic_model import BasicModel
from models.graph_model_wrapper import GraphWrapper
from mol_graph import *
from mol_graph import GraphEncoder
from pre_process.data_loader import GraphDataSet, collate_2d_graphs, collate_2d_tensors
from pre_process.load_dataset import load_classification_dataset
import tqdm


class BondEncoder(nn.Module):
    def __init__(self):
        super(BondEncoder, self).__init__()
        self.encoder = torch.nn.Embedding(8, 2, max_norm=1)
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, 8)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AtomEncoder(nn.Module):
    def __init__(self):
        super(AtomEncoder, self).__init__()
        self.encoder = torch.nn.Embedding(58, 4, max_norm=1)
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Linear(4, 58)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


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
            labels = labels + model(batch).max(dim=-1)[1].cpu().data.numpy().tolist()
            true_labels = true_labels + batch['labels'].cpu().data.numpy().tolist()
    return (
        metrics.accuracy_score(true_labels, labels),
        metrics.precision_score(true_labels, labels, average='weighted'),
        metrics.recall_score(true_labels, labels, average='weighted')
    )


def embed_afm(afm):
    all_atoms = [atom.reshape(-1) for atom in np.vsplit(afm, afm.shape[0])]
    all_atoms = [atom_dict[''.join(map(str, entry))] for entry in all_atoms]
    all_atoms = map(long, all_atoms)
    return atom_embed(torch.as_tensor(all_atoms)).data.numpy().astype(np.float32)


def embed_bfm(bfm):
    all_bonds = np.vsplit(bfm.reshape(-1, 8), bfm.shape[0] ** 2)
    all_bonds = [bond_dict[''.join(map(str, entry.reshape(-1)))] for entry in all_bonds]
    all_bonds = map(long, all_bonds)
    return bond_embed(torch.as_tensor(all_bonds)).data.numpy().astype(np.float32)


with open('./atom_dict.pickle') as f:
    atom_dict = pickle.load(f)

with open('./bond_dict.pickle') as f:
    bond_dict = pickle.load(f)

atom_embed = nn.Embedding(58, 4, max_norm=1)
atom_embed.load_state_dict(torch.load('./atom_embed.state_dict', map_location=lambda storage, loc: storage))
atom_embed.cpu().eval()
bond_embed = nn.Embedding(8, 2, max_norm=1)
bond_embed.load_state_dict(torch.load('./bond_embed.state_dict', map_location=lambda storage, loc: storage))
bond_embed.cpu().eval()


seed = 317
torch.manual_seed(seed)
data_file = sys.argv[1]

mgf = MolGraphFactory(Mol2DGraph.TYPE, AtomFeatures(), BondFeatures())
try:
    file_data = np.load(data_file+'.npz')
    data = file_data['data']
    for graph in data:
        graph.mask = np.ones(graph.afm.shape[0], dtype=np.float32).reshape(graph.afm.shape[0], 1)
        graph.afm = embed_afm(graph.afm)
        graph.adj = graph.adj.astype(np.float32)
        graph.bfm = embed_bfm(graph.bfm)
        graph.bfm = np.multiply(graph.bfm, graph.adj.reshape(-1, 1)).reshape(graph.adj.shape + (-1,)).astype(np.float32)
        graph.label = long(graph.label)
    no_labels = int(file_data['no_labels'])
    all_labels = file_data['all_labels']
    file_data.close()
except IOError:
    data, no_labels, all_labels = load_classification_dataset(data_file+'.csv',
                                              'InChI', Chem.MolFromInchi, mgf, 'target')
    graph_encoder = GraphEncoder()
    with open('basic_model_graph_encoder.pickle', 'wb') as out:
        pickle.dump(graph_encoder, out)
    np.savez_compressed(data_file, data=data, no_labels=no_labels, all_labels=all_labels)

model_attributes = {
    'afm': data[0].afm.shape[-1],
    'bfm': data[0].bfm.shape[-1],
    'mfm': 2*data[0].afm.shape[-1],
    'adj': data[0].adj.shape[-1],
    'out': 4*data[0].afm.shape[-1],
    'classification_output': no_labels
}

model = nn.Sequential(
    GraphWrapper(BasicModel(model_attributes['afm'], model_attributes['bfm'], model_attributes['mfm'],
                            model_attributes['adj'], model_attributes['out'])),
    # nn.BatchNorm1d(model_attributes['out']),
    nn.Linear(model_attributes['out'], model_attributes['classification_output'])
)

model.float()  # convert to half precision
# for layer in model.modules():
#     if isinstance(layer, nn.BatchNorm1d):
#         layer.float()

print "Model has: {} parameters".format(count_model_params(model))
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
model.train()

train, test, train_labels, test_labels = train_test_split(data, all_labels, test_size=0.1,
                                                          random_state=seed, stratify=all_labels)
del data
del all_labels
del test_labels
train, val = train_test_split(train, test_size=0.1, random_state=seed, stratify=train_labels)
del train_labels
train = GraphDataSet(train)
val = GraphDataSet(val)
test = GraphDataSet(test)
train = DataLoader(train, 16, shuffle=True, collate_fn=collate_2d_graphs)
val = DataLoader(val, 16, shuffle=True, collate_fn=collate_2d_graphs)
test = DataLoader(test, 16, shuffle=True, collate_fn=collate_2d_graphs)

losses = []
epoch_losses = []
break_con = False
for epoch in tqdm.trange(500):
    model.train()
    epoch_loss = 0
    for batch in tqdm.tqdm(train):
        model.zero_grad()
        loss = criterion(model(batch), batch['labels'])
        losses.append(loss.item())
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    acc, pre, rec = test_model(model, val)
    f1 = 2 * (pre * rec) / (pre + rec)
    tqdm.tqdm.write(
        "epoch {} training loss: {}, validation acc: {}, pre: {}, rec: {}, F1: {}".format(epoch, epoch_loss, acc,
                                                                                          pre, rec, f1))
    if not np.isnan(f1) and f1 > 0.78:
        save_model(model, 'epoch_'+str(epoch), model_attributes, {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1})
    # epoch_losses.append(epoch_loss)
    # if 0 == (epoch+1) % 50:
    #     print "epoch: {}, loss: {}".format(epoch, epoch_loss)
    # break_con = loss.item() < 0.02
    # if break_con:
    #     break

acc, pre, rec = test_model(model, test)
f1 = 2 * (pre * rec) / (pre + rec)
tqdm.tqdm.write(
    "Testing acc: {}, pre: {}, rec: {}, F1: {}".format(epoch, epoch_loss, acc, pre, rec, f1))
# save_model(model, model_attributes)

# model.eval()
# labels = []
# true_labels = []
# for batch in val:
#     labels = labels + model(batch).max(dim=-1)[1].cpu().data.numpy().tolist()
#     true_labels = true_labels + batch['labels'].cpu().data.numpy().tolist()
#
# print "accuracy: {}, precision: {}, recall: {}".format(
#     metrics.accuracy_score(true_labels, labels),
#     metrics.precision_score(true_labels, labels, average='micro'),
#     metrics.recall_score(true_labels, labels, average='micro')
# )
