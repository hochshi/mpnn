import cPickle as pickle
import sys

import numpy as np
import torch
import torch.cuda
from rdkit import Chem
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models.lipo_basic_model import BasicModel, _DEF_STEPS
from models.graph_norm_wrapper import GraphWrapper
from mol_graph import *
from mol_graph import GraphEncoder
from pre_process.data_loader import GraphDataSet, collate_2d_graphs, collate_2d_tensors
from pre_process.load_dataset import load_number_dataset, load_classification_dataset
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
    tot_loss = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            output = model(batch).squeeze()
            loss = criterion(output, batch['labels'])
            tot_loss += loss.item() * batch['afm'].shape[0]
            labels.extend(output.max(dim=-1)[1].cpu().data.numpy().tolist())
            true_labels.extend(batch['labels'].cpu().data.numpy().tolist())
    return tot_loss/len(dataset.dataset), metrics.roc_auc_score(true_labels, labels)

def test_model_class(model, dataset):
    model.eval()
    labels = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            labels = labels + model(batch).max(dim=-1)[1].cpu().data.numpy().tolist()
            true_labels = true_labels + batch['labels'].cpu().data.numpy().tolist()
    return metrics.accuracy_score(true_labels, labels)


def train_model(tmodel, ttrain, tval, ttest, toptimizer, tcriterion):
    epoch_losses = []
    break_con = False
    itr = tqdm.trange(1000)
    for epoch in itr:
        tmodel.train()
        epoch_loss = 0
        for batch in tqdm.tqdm(ttrain):
            tmodel.zero_grad()
            loss = tcriterion(tmodel(batch).squeeze(), batch['labels'])
            epoch_loss += loss.item() * batch['afm'].shape[0]
            loss.backward()
            toptimizer.step()
        epoch_losses.append(epoch_loss)
        t_mse = test_model(tmodel, ttrain)
        mse = test_model(tmodel, tval)
        tqdm.tqdm.write(
            "epoch {} loss: {} Train MSE: {} Val MSE: {}".format(epoch, epoch_loss / len(ttrain.dataset), t_mse, mse))
        if mse[1] < 0.7:
            itr.close()
            break

    vmse = test_model(tmodel, tval)
    tqdm.tqdm.write("Val MSE: {}".format(vmse))
    tmse = test_model(tmodel, ttest)
    tqdm.tqdm.write("Testing MSE: {}".format(tmse))
    return [vmse, tmse]

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
data, no_labels, all_labels = load_classification_dataset(data_file, 'smiles', Chem.MolFromSmiles, mgf, 'HIV_active')
data = np.array(data)
for graph in data:
    graph.mask = np.ones(graph.afm.shape[0], dtype=np.float32).reshape(graph.afm.shape[0], 1)
    graph.afm = graph.afm.astype(np.float32)
    graph.bfm = graph.bfm.astype(np.long)
    graph.a_bfm = graph.a_bfm.astype(np.long)
    graph.adj = graph.adj.astype(np.float32)
    graph.label = np.float32(graph.label)
graph_encoder = GraphEncoder()


bfm = int(sum(None != graph_encoder.bond_enc[0].classes_))
a_bfm = int(sum(None != graph_encoder.a_bond_enc[0].classes_))
afm = int(np.ceil(a_bfm ** 0.25))
mfm = int(np.ceil(a_bfm ** 0.25) * np.ceil(bfm ** 0.5))

model_attributes = {
    'afm': afm,
    'bfm': bfm,
    'a_bfm': a_bfm,
    'mfm': mfm,
    'adj': 1,
    'out': mfm*2,
    'classification_output': 1
}

model = nn.Sequential(
        GraphWrapper(BasicModel(model_attributes['afm'], model_attributes['bfm'], model_attributes['a_bfm'],
                                model_attributes['mfm'],
                                model_attributes['adj'], model_attributes['out'])),
        nn.BatchNorm1d(model_attributes['out']),
        nn.Linear(model_attributes['out'], model_attributes['classification_output'])
    )

print "Model has: {} parameters".format(count_model_params(model))
print model
print model_attributes

run_res = []
kf = KFold(n_splits=10, shuffle=True, random_state=seed)
for train, test in tqdm.tqdm(kf.split(data)):
    train, val = train_test_split(train, test_size=0.1, random_state=seed)
    train = GraphDataSet(data[train])
    val = GraphDataSet(data[val])
    test = GraphDataSet(data[test])
    train = DataLoader(train, 16, shuffle=True, collate_fn=collate_2d_graphs)
    val = DataLoader(val, 16, shuffle=True, collate_fn=collate_2d_graphs)
    test = DataLoader(test, 16, shuffle=True, collate_fn=collate_2d_graphs)

    model = nn.Sequential(
        GraphWrapper(BasicModel(model_attributes['afm'], model_attributes['bfm'], model_attributes['a_bfm'],
                                model_attributes['mfm'],
                                model_attributes['adj'], model_attributes['out'])),
        nn.BatchNorm1d(model_attributes['out']),
        nn.Linear(model_attributes['out'], model_attributes['classification_output'])
    )

    model.float()
    model.apply(BasicModel.init_weights)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    model.train()

    run_res.append(train_model(model, train, val, test, optimizer, criterion))




