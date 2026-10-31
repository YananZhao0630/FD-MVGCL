import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy.sparse import coo_matrix
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator, from_predefined_split, from_predefined_split_wiki, from_predefined_split_heter, from_predefined_split_heterophilous
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, Actor
from torch_geometric.datasets import WebKB
from ogb.nodeproppred import NodePropPredDataset
import numpy as np
import scipy.sparse as sp
import scipy.io
import torch
import torch.nn as nn
import time
from dataset import WikipediaNetwork, load_fixed_splits
import os
import copy
import random
import argparse
import sys
from pdemodel.GNN_pde import GNNPDE, GNN
from pdemodel.classifier import Classifier
from data import get_train_val_test_split, even_quantile_labels
import json
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from sklearn import preprocessing as sk_prep
from pdemodel.best_params import best_params_dict
from prettytable import PrettyTable
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.io as sio
from typing import List

def _now(device=None):
    if device is not None and isinstance(device, torch.device) and device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)


class CustomNodePropPredDataset:
    def __init__(self, name, custom_num_classes):
        # Load the original dataset
        self.dataset = NodePropPredDataset(name=name)
        self._num_classes = custom_num_classes

        # Set up data (your Data object from previous steps)
        self.data = None  # Initialized as None and populated later

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_features(self):
        # Dynamically fetch the number of features from dataset
        graph = self.dataset.graph  # Assuming self.dataset.graph contains 'node_feat'
        return graph['node_feat'].shape[1]

    def __getitem__(self, idx):
        # Return the single data object (since this dataset contains only one graph)
        if idx != 0:
            raise IndexError("CustomNodePropPredDataset contains only one graph.")
        return self.data

    def __len__(self):
        # The length is 1 because it contains only one graph
        return 1

    # Forward any other attribute or method calls to the original dataset
    def __getattr__(self, attr):
        return getattr(self.dataset, attr)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    ham_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def count_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb


def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim, proj_dim, num_features, a1, a2):
        super(Encoder, self).__init__()
        self.encoder1, self.encoder2 = encoder
        self.a1, self.a2 = a1, a2

    def forward(self, x, edge_index, edge_weight=None):
        z1_0, z1 = self.encoder1(x, [edge_index, edge_weight])
        z2_0, z2 = self.encoder2(x, [edge_index, edge_weight])
        return z1_0, z2_0, z1, z2

    def eval_test(self, x, edge_index, edge_weight=None):
        with torch.no_grad():
            z1_0, z1 = self.encoder1(x, [edge_index, edge_weight])
            z2_0, z2 = self.encoder2(x, [edge_index, edge_weight])
            z0 = self.a1 * z1_0 + self.a2 * z2_0
            z = self.a1 * z1 + self.a2 * z2
        return z0, z

    # def cal_reg(self, z1, z2):
    #     return self.encoder.odeblock.cal_loss(z1, z2)

class MultiEncoder(nn.Module):
    def __init__(self, encoders, hidden_dim, proj_dim, num_features, a_list):
        super(MultiEncoder, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.a_list = a_list  # list of weights [a1, a2, ..., an]

    def forward(self, x, edge_index, edge_weight=None):
        z0_list, z_list = [], []
        for encoder in self.encoders:
            z0, z = encoder(x, [edge_index, edge_weight])
            z0_list.append(z0)
            z_list.append(z)
        return z0_list, z_list

    def eval_test(self, x, edge_index, edge_weight=None):
        with torch.no_grad():
            z0_list, z_list = [], []
            for encoder in self.encoders:
                z0, z = encoder(x, [edge_index, edge_weight])
                z0_list.append(z0)
                z_list.append(z)

            # Weighted sum of all encoders
            z0 = sum(a * z0 for a, z0 in zip(self.a_list, z0_list))
            z = sum(a * z for a, z in zip(self.a_list, z_list))
        return z0, z


# class Tri_Encoder(torch.nn.Module):
#     def __init__(self, encoder, hidden_dim, proj_dim, num_features, a1, a2, a3):
#         super(Tri_Encoder, self).__init__()
#         self.encoder1, self.encoder2, self.encoder3 = encoder
#         self.a1, self.a2, self.a3 = a1, a2, a3
#
#     def forward(self, x, edge_index, edge_weight=None):
#         z1_0, z1 = self.encoder1(x, [edge_index, edge_weight])
#         z2_0, z2 = self.encoder2(x, [edge_index, edge_weight])
#         z3_0, z3 = self.encoder3(x, [edge_index, edge_weight])
#         return z1_0, z2_0, z3_0, z1, z2, z3
#
#     def eval_test(self, x, edge_index, edge_weight=None):
#         with torch.no_grad():
#             z1_0, z1 = self.encoder1(x, [edge_index, edge_weight])
#             z2_0, z2 = self.encoder2(x, [edge_index, edge_weight])
#             z3_0, z3 = self.encoder3(x, [edge_index, edge_weight])
#             z0 = self.a1 * z1_0 + self.a2 * z2_0 + self.a3 * z3_0
#             z = self.a1 * z1 + self.a2 * z2 + self.a3 * z3
#         return z0, z
#
#     def cal_reg(self, z1, z2):
#         return self.encoder.odeblock.cal_loss(z1, z2)


def loss_fn_batch(x, y):
    # add batch dimension
    x = F.normalize(x, dim=1, p=2)
    y = F.normalize(y, dim=1, p=2)
    loss = 2 - 2 * (x * y).sum(dim=-1).mean()
    return loss


def loss_fn(x, y):
    # x1 = torch.mean(x, dim=1)
    # y1 = torch.mean(y, dim=1)

    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    # loss = torch.linalg.matrix_norm(x-y)
    loss = 1 - 1 * (x * y).sum(dim=-1).mean()
    # loss = 1 - 1 * (x * y).sum(dim=-1).mean()
    cos_sim = (x * y).sum(dim=-1)
    cos_sim_var = cos_sim.var()

    return loss, cos_sim_var
    # return 2 - 2 * (x * y).sum(dim=-1)


def loss_fn_plus(x, y, lamba=0.1):
    # x1 = torch.mean(x, dim=1)
    # y1 = torch.mean(y, dim=1)

    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    pca_x = PCA()
    pca_y = PCA()

    pca_x.fit(x.detach().cpu().numpy())
    pca_y.fit(y.detach().cpu().numpy())

    main_feature_vector_x = pca_x.components_[0]
    main_feature_vector_y = pca_y.components_[0]

    main_feature_vector_x_tensor = torch.from_numpy(main_feature_vector_x).to(x.device)
    main_feature_vector_y_tensor = torch.from_numpy(main_feature_vector_y).to(y.device)

    # loss = torch.linalg.matrix_norm(x-y)
    loss_1 = 1 - 1 * (x * y).sum(dim=-1).mean()
    # loss = 1 - 1 * (x * y).sum(dim=-1).mean()
    cos_sim = (x * y).sum(dim=-1)
    cos_sim_var = cos_sim.var()

    # Compute the magnitudes
    dot_product = torch.dot(main_feature_vector_x_tensor, main_feature_vector_y_tensor)
    magnitude_x = torch.norm(main_feature_vector_x_tensor)
    magnitude_y = torch.norm(main_feature_vector_y_tensor)
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_x * magnitude_y)

    abs_cos_theta = torch.abs(cos_theta)

    loss_2 = abs_cos_theta

    loss = loss_1 + lamba * loss_2

    return loss


def vicreg_loss(z1, z2, lambd=1.0, mu=1.0, nu=1.0, gamma=1.0, epsilon=1e-4):
    """
    Computes the VICReg loss for graph feature matrices.

    Args:
        z1 (torch.Tensor): Feature matrix 1, shape (N, d), where N is the number of nodes and d is the feature dimension.
        z2 (torch.Tensor): Feature matrix 2, shape (N, d).
        lambd (float): Weight for the invariance component.
        mu (float): Weight for the variance component.
        nu (float): Weight for the covariance component.
        gamma (float): Margin for the variance component.
        epsilon (float): Small scalar for numerical stability.

    Returns:
        loss (torch.Tensor): VICReg loss value.
        loss_var (torch.Tensor): Variance of the loss contributions.
    """
    # Number of nodes and feature dimensions
    N, d = z1.size()

    # Invariance component: Mean squared error between z1 and z2
    invariance = F.mse_loss(z1, z2, reduction='none')
    invariance_contributions = invariance.mean(dim=1)  # Per-node contributions

    # Variance component: Enforce non-collapse by ensuring variance along feature dimensions
    def variance_penalty(z):
        std = torch.sqrt(z.var(dim=0, unbiased=False) + epsilon)
        return F.relu(gamma - std)

    variance_contributions = variance_penalty(z1).mean() + variance_penalty(z2).mean()

    # Covariance component: Decorrelate features
    def covariance_penalty(z):
        z = z - z.mean(dim=0)  # Center the features
        cov = (z.T @ z) / (N - 1)  # Covariance matrix
        off_diag = cov - torch.diag(torch.diag(cov))  # Zero the diagonal
        return off_diag.pow(2).sum(dim=1) / d  # Per-feature contributions

    covariance_contributions = covariance_penalty(z1).mean() + covariance_penalty(z2).mean()

    # Combine components
    loss = lambd * invariance.mean() + mu * variance_contributions + nu * covariance_contributions

    # Calculate variance of the loss contributions
    loss_contributions = (lambd * invariance_contributions).flatten()
    loss_var = loss_contributions.var()

    return loss, loss_var


def barlow_twin_loss(x, y, lambd=1.0):
    """
    Computes the Barlow Twins loss.

    Args:
        x (torch.Tensor): Input tensor 1, shape (batch_size, feature_dim).
        y (torch.Tensor): Input tensor 2, shape (batch_size, feature_dim).
        lambd (float): Weight for the redundancy reduction term.

    Returns:
        loss (torch.Tensor): Barlow Twins loss value.
    """
    # Normalize the inputs
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    # Cross-correlation matrix computation
    C = torch.einsum('bi,bj->ij', x, y) / x.size(0)

    # Compute invariance term (diagonal close to 1)
    invariance_term = (1 - torch.diagonal(C)).pow(2)

    # Compute redundancy reduction term (off-diagonals close to 0)
    off_diag = C.clone()
    off_diag[torch.eye(C.size(0), device=C.device).bool()] = 0
    redundancy_term = off_diag.pow(2)

    # Combine the two terms
    loss = invariance_term.sum() + lambd * redundancy_term.sum()
    loss_var = invariance_term.var()

    return loss


def loss_fn_cos(x1, y1):
    loss_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cossim = loss_cos(x1, y1)
    # loss_cos_dis = 2 - 2 * loss_cossim
    loss_cos_dis = 1 - loss_cossim
    loss_cos_dis_all = loss_cos_dis.sum()
    loss_var = loss_cos_dis.var()

    # loss_cossim_mean = loss_cossim.mean()
    # loss_all = 2 - 2 * loss_cossim_mean

    return loss_cos_dis_all


def loss_fn_cos_plus(x1, y1, lambd=0.1):
    loss_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cossim = loss_cos(x1, y1)
    # loss_cos_dis = 2 - 2 * loss_cossim
    loss_cos_dis = 1 - loss_cossim
    loss_cos_dis_all = loss_cos_dis.sum()

    pca_x = PCA()
    pca_y = PCA()

    pca_x.fit(x1.detach().cpu().numpy())
    pca_y.fit(y1.detach().cpu().numpy())

    main_feature_vector_x = pca_x.components_[0]
    main_feature_vector_y = pca_y.components_[0]

    main_feature_vector_x_tensor = torch.from_numpy(main_feature_vector_x).to(x1.device)
    main_feature_vector_y_tensor = torch.from_numpy(main_feature_vector_y).to(y1.device)

    dot_product = torch.dot(main_feature_vector_x_tensor, main_feature_vector_y_tensor)
    magnitude_x = torch.norm(main_feature_vector_x_tensor)
    magnitude_y = torch.norm(main_feature_vector_y_tensor)
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_x * magnitude_y)

    abs_cos_theta = torch.abs(cos_theta)

    loss_2 = abs_cos_theta

    loss_var = loss_cos_dis.var()

    # loss_cossim_mean = loss_cossim.mean()
    # loss_all = 2 - 2 * loss_cossim_mean

    loss = loss_cos_dis_all + lambd * loss_2

    return loss


def loss_fn_euclidean(x1, y1):
    loss_euclidean = torch.nn.PairwiseDistance(p=2)
    loss_euclidean_dis = loss_euclidean(x1, y1)
    loss_euclidean_dis_all = loss_euclidean_dis.sum()
    loss_var = loss_euclidean_dis.var()

    return loss_euclidean_dis_all


def loss_fn_euclidean_mean(x1, y1):
    loss_euclidean = torch.nn.PairwiseDistance(p=2)
    loss_euclidean_dis = loss_euclidean(x1, y1)
    loss_euclidean_dis_all = loss_euclidean_dis.mean()
    loss_var = loss_euclidean_dis.var()

    return loss_euclidean_dis_all


def get_rows_by_class(feature_matrix, labels, target_class):
    selected_indices = np.where(labels == target_class)[0]
    return feature_matrix[selected_indices]


def train(encoder_model, data, optimizer, optimizer_list: List[torch.optim.Optimizer]):
    encoder_model.train()
    optimizer.zero_grad()
    for optim_alpha in optimizer_list:
        optim_alpha.zero_grad()

    z0_list, z1_list = encoder_model(data.x, data.edge_index, data.edge_attr)
    if opt['loss'] == 'barlow_twin':
        if len(z1_list) == 2:
            loss = barlow_twin_loss(z1_list[0], z1_list[1], lambd=opt['lambda'])
        else:
            loss = 0
            for i in range(len(z1_list) - 1):  # adjacent pairs
                loss += barlow_twin_loss(z1_list[i], z1_list[i + 1], lambd=opt['lambda'])

            loss += barlow_twin_loss(z1_list[0], z1_list[-1], lambd=opt['lambda'])  # endpoints
    elif opt['loss'] == 'cossum_plus':
        if len(z1_list) == 2:
            loss = loss_fn_cos_plus(z1_list[0], z1_list[1], lambd=opt['lambda'])
        else:
            loss = 0
            for i in range(len(z1_list) - 1):  # adjacent pairs
                loss += loss_fn_cos_plus(z1_list[i], z1_list[i + 1], lambd=opt['lambda'])

            loss += loss_fn_cos_plus(z1_list[0], z1_list[-1], lambd=opt['lambda'])  # endpoints
    elif opt['loss'] == 'cosmean_plus':
        if len(z1_list) == 2:
            loss = loss_fn_plus(z1_list[0], z1_list[1], lamba=opt['lambda'])
        else:
            loss = 0
            for i in range(len(z1_list) - 1):  # adjacent pairs
                loss += loss_fn_plus(z1_list[i], z1_list[i + 1], lamba=opt['lambda'])

            loss += loss_fn_plus(z1_list[0], z1_list[-1], lamba=opt['lambda'])  # endpoints
    else:
        raise NotImplementedError


    loss.backward()
    optimizer.step()

    for optim_alpha in optimizer_list:
        optim_alpha.step()

    return loss.item(), z1_list


def test(encoder_model, data, opt, class_time):
    encoder_model.eval()
    z0, z = encoder_model.eval_test(data.x, data.edge_index, data.edge_attr)
    z0 = z0.detach()
    z = z.detach()

    if opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed', 'ogbn-arxiv']:
        split = from_predefined_split(data)
    elif opt['dataset'] in ['wikics']:
        split = from_predefined_split_wiki(data, class_time)
    elif opt['dataset'] in ['cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel', 'actor', 'crocodile']:
        split = from_predefined_split_heter(data, class_time)
    elif opt['dataset'] in ['wiki-cooc', 'roman-empire', 'amazon-ratings', 'minesweeper', 'workers', 'questions',
                            'arxiv-year', 'chameleon-filtered', 'squirrel-filtered']:
        split = from_predefined_split_heterophilous(data, class_time)
    else:
        split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator(num_epochs=opt['classifier_epochs'], learning_rate=opt['classifier_lr'],
                         weight_decay=opt['classifier_decay'])(z, data.y, split)
    return result


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def test_classifier(encoder_model, data, classifier, classifier_optimizer, opt):
    encoder_model.eval()
    embeds = encoder_model.eval_test(data.x, data.edge_index, data.edge_attr).detach()

    if opt['dataset'] == 'ogbn-arxiv':
        embeds = embeds.squeeze(0)
    embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")

    embeds = torch.FloatTensor(embeds).to(opt['cuda'])
    dur = []
    best_acc, best_val_acc = 0, 0
    print('Testing Phase ==== Please Wait.')
    n_classifier_epochs = 5000
    for epoch in range(n_classifier_epochs):
        classifier.train()
        if epoch >= 3:
            t0 = time.time()

        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = F.nll_loss(preds[data.train_mask], data.y[data.train_mask])
        loss.backward()
        classifier_optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        val_acc = evaluate(classifier, embeds, data.y, data.val_mask)
        if epoch > 500:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = evaluate(classifier, embeds, data.y, data.test_mask)
                if test_acc > best_acc:
                    best_acc = test_acc
    print("Valid Accuracy {:.4f}".format(best_val_acc))

    # best_acc = evaluate(classifier, embeds, labels, test_mask)
    print("Test Accuracy {:.4f}".format(best_acc))

    return best_acc


def deduplicate_alphas(alpha_values, delta=0.1):
    remaining = []
    visited = [False] * len(alpha_values)

    for i in range(len(alpha_values)):
        if not visited[i]:
            # Find log-close elements
            group = [i]
            for j in range(i + 1, len(alpha_values)):
                if not visited[j] and abs(np.log(alpha_values[i]) - np.log(alpha_values[j])) < delta:
                    group.append(j)
            # Randomly keep one from group
            keep_idx = random.choice(group)
            remaining.append(alpha_values[keep_idx])
            # Mark all in group as visited
            for j in group:
                visited[j] = True

    # Add remaining values not in any log-close group
    return sorted(remaining, reverse=True)

def main(opt):
    set_seed(opt['seed'])
    device = torch.device('cuda:' + str(opt['cuda']) if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.expanduser('~'), 'datasets')
    if opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, name=opt['dataset'], transform=T.NormalizeFeatures())
    elif opt['dataset'] in ['Computers', 'Photo']:
        dataset = Amazon(path, name=opt['dataset'])
    elif opt['dataset'] in ['Coauthorcs']:
        dataset = Coauthor(path, name='CS')
    elif opt['dataset'] in ['Coauthorphy']:
        dataset = Coauthor(path, name='Physics')
    elif opt['dataset'] in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name=opt['dataset'])
    elif opt['dataset'] in ['wikics']:
        dataset = WikiCS(path)
    elif opt['dataset'] in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, name=opt['dataset'])
    elif opt['dataset'] in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=f'../datasets/', name=opt['dataset'], geom_gcn_preprocess=True)
    elif opt['dataset'] in ['crocodile']:
        dataset = WikipediaNetwork(root=f'../dataset/', name=opt['dataset'], geom_gcn_preprocess=False)
    elif opt['dataset'] in ['actor']:
        path = os.path.join(path, opt['dataset'])
        dataset = Actor(path)
    elif opt['dataset'] in ['arxiv-year']:
        dataset = CustomNodePropPredDataset(name='ogbn-arxiv', custom_num_classes=5)
        split_dir = '../data/splits'
        split_idx = load_fixed_splits('arxiv-year', split_dir)
        for i in range(len(split_idx)):
            for key in split_idx[i]:
                if not torch.is_tensor(split_idx[i][key]):
                    split_idx[i][key] = torch.as_tensor(split_idx[i][key])

        # Extract train, valid, and test splits
        train_masks = []
        valid_masks = []
        test_masks = []

        for split in split_idx:
            train_masks.append(split['train'])
            valid_masks.append(split['valid'])
            test_masks.append(split['test'])

        # Stack masks to create the desired 2D tensor structure
        train_mask = torch.stack(train_masks)  # Each row represents a split's train mask
        valid_mask = torch.stack(valid_masks)  # Each row represents a split's valid mask
        test_mask = torch.stack(test_masks)

        graph = dataset.graph

        nclass = 5
        label = even_quantile_labels(
            graph['node_year'].flatten(), nclass, verbose=False)
        label = torch.tensor(label)
        node_features = torch.tensor(graph['node_feat'])
        edges = graph['edge_index']
        data = Data(
            x=node_features,
            edge_index=torch.LongTensor(edges),
            y=label,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=valid_mask
        )
        dataset.data = data

    if opt['dataset'] in ['Computers', 'Photo', 'CoauthorCS', 'CoauthorPHY']:
        dataset.data = get_train_val_test_split(opt['seed'], dataset.data,
                                                train_examples_per_class=20, val_examples_per_class=None,
                                                test_examples_per_class=None,
                                                train_size=None, val_size=500, test_size=1000)
    elif opt['dataset'] in ['ogbn-arxiv']:
        split_idx = dataset.get_idx_split()
        ei = to_undirected(dataset.data.edge_index)
        data = Data(
            x=dataset.data.x,
            edge_index=ei,
            y=dataset.data.y.T.squeeze(0),
            train_mask=split_idx['train'],
            test_mask=split_idx['test'],
            val_mask=split_idx['valid'])
        dataset.data = data
    elif opt['dataset'] in ['wiki-cooc', 'roman-empire', 'amazon-ratings', 'minesweeper', 'workers', 'questions', 'chameleon-filtered', 'squirrel-filtered']:
        dataset = MyOwnDataset(path, name=opt['dataset'])
        data = np.load(
            os.path.join('/home/zyn/datasets/HeterophilousDatasets/data', f'{opt["dataset"].replace("-", "_")}.npz'))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])
        edges = edges.T

        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])

        # train_idx_list = [torch.where(train_mask)[0] for train_mask in train_mask]
        # val_idx_list = [torch.where(val_mask)[0] for val_mask in val_mask]
        # test_idx_list = [torch.where(test_mask)[0] for test_mask in test_mask]
        # train_mask = train_masks[split, :]
        # val_mask = val_masks[split, :]
        # test_mask = test_masks[split, :]
        #
        # print("fixed_splits with splits number: ", split)

        data = Data(
            x=node_features,
            edge_index=torch.LongTensor(edges),
            y=labels,
            train_mask=train_masks,
            test_mask=test_masks,
            val_mask=val_masks
        )

        # use_lcc = False
        dataset.data = data

    # else:
    #     dataset = get_dataset(opt, './data')
    data = dataset[0].to(device)

    n_classes = dataset.num_classes

    num_nodes = data.x.shape[0]
    opt['num_nodes'] = num_nodes

    # opt_gconv1 = copy.deepcopy(opt)
    # opt_gconv2 = copy.deepcopy(opt)
    # if opt['block'] == 'constant_alpha':
    #     opt_gconv1['block'] = 'constant_frac'
    #
    # opt_gconv1['alpha_ode'] = opt['alpha_ode_1']
    # opt_gconv2['alpha_ode'] = opt['alpha_ode_2']
    #
    # gconv1 = GNNPDE(opt_gconv1, dataset.num_features, data, device)
    # gconv2 = GNNPDE(opt_gconv2, dataset.num_features, data, device)
    #
    # count_parameters(gconv1)
    # count_model_size(gconv1)
    # count_parameters(gconv2)
    # count_model_size(gconv2)
    repeat = True

    while repeat:
        gconvs = []
        # make the encoder more scalable and flexible
        for i in range(opt['num_encoders']):
            opt_gconv = copy.deepcopy(opt)

            # Use 'constant_frac' block if specified for the first encoder
            if i == 0 and opt['block'] == 'constant_alpha':
                opt_gconv['block'] = 'constant_frac'

            opt_gconv['alpha_ode'] = opt[f'alpha_ode_{i + 1}']
            gconv = GNNPDE(opt_gconv, dataset.num_features, data, device)

            count_parameters(gconv)
            count_model_size(gconv)

            gconvs.append(gconv)


        # if opt['num_encoders'] == 3:
        #     opt_gconv3 = copy.deepcopy(opt)
        #     opt_gconv3['alpha_ode'] = opt['alpha_ode_3']
        #     gconv3 = GNNPDE(opt_gconv3, dataset.num_features, data, device)
        #     count_parameters(gconv3)
        #     count_model_size(gconv3)

        # if opt['num_encoders'] == 3:
        encoder_model = MultiEncoder(encoders=gconvs, hidden_dim=opt['hidden_dim'], proj_dim=opt['proj_dim'], num_features=dataset.num_features, a_list=[args.a1, args.a2, args.a3, args.a4, args.a5, args.a6, args.a7, args.a8, args.a9, args.a10][:opt['num_encoders']]).to(device)
        parameters = [
            {'params': p, 'name': n} for n, p in encoder_model.named_parameters() if p.requires_grad
        ]

        # Step 1: Separate alpha_params and other params
        alpha_optimizers = []
        alpha_param_names = []
        alpha_lr_list = [opt[f'alpha_lr_{i + 1}'] for i in range(opt['num_encoders'])]

        for i, (encoder, alpha_lr) in enumerate(zip(encoder_model.encoders[1:], alpha_lr_list[1:]), start=1):
            alpha_param = encoder.odeblock.alpha_param
            alpha_optimizers.append(torch.optim.Adam([alpha_param], lr=alpha_lr))
            alpha_param_names.append(f'encoders.{i}.odeblock.alpha_param')

        # Step 2: Gather all non-alpha parameters
        optimizer_list = alpha_optimizers
        other_params = [
            p for n, p in encoder_model.named_parameters()
            if 'alpha_param' not in n and p.requires_grad
        ]

        # Step 3: Define optimizer for non-alpha parameters
        optimizer = get_optimizer(
            opt['optimizer'], other_params,
            lr=opt['lr'], weight_decay=opt['decay']
        )



        classifier = Classifier(opt['hidden_dim'], n_classes).to(device)
        classifier_optimizer = torch.optim.AdamW(classifier.parameters(),
                                                 lr=opt['classifier_lr'],
                                                 weight_decay=opt['classifier_decay'])


        loss_values = []
        cnt_wait = 0
        best_loss = 1e9
        best_t = 0

        epoch_values = list(range(1, opt['epoch'] + 1))
        # ep_train_times, ep_infer_times = [], []
        with tqdm(total=opt['epoch'], desc='(T)') as pbar:
            for epoch in range(1, opt['epoch'] + 1):
                # t_train0 = _now(device)
                loss, z1_list = train(encoder_model, data, optimizer, optimizer_list)
                # t_train1 = _now(device)
                # train_s = t_train1 - t_train0
                # ep_train_times.append(train_s)
                
                loss_values.append(loss)
                if loss < best_loss:
                    best_loss = loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(encoder_model.state_dict(),
                               'pkl/modelmulti_' + str(opt['hidden_dim']) + '_' + opt['dataset'] + opt['function'] + opt[
                                   'block'] + '.pkl')
                else:
                    cnt_wait += 1
                # pbar.set_postfix({'loss': loss, 'loss_var': loss_var})
                pbar.set_postfix({'loss': loss})
                pbar.update()



                if cnt_wait == opt['patience']:
                    print('Early stopping!')
                    break


        encoder_model.load_state_dict(torch.load(
            'pkl/modelmulti_' + str(opt['hidden_dim']) + '_' + opt['dataset'] + opt['function'] + opt['block'] + '.pkl',
            map_location='cpu'))
        encoder_model = encoder_model.to(device)
        encoder_model.eval()
        # with torch.no_grad():
        #         t_inf0 = _now(device)
        #         _ = encoder_model.eval_test(data.x, data.edge_index, data.edge_attr)
        #         t_inf1 = _now(device)
        # infer_s = t_inf1 - t_inf0
        # ep_infer_times.append(infer_s)
        print("Learned alpha values from best model:")
        print(f"Encoder 1: alpha = 1")
        alpha_values = []
        for i, encoder in enumerate(encoder_model.encoders[1:], start=1):
            alpha_value = encoder.odeblock.alpha_param.item()
            alpha_value = torch.clamp(torch.tensor(alpha_value), 0.0001, 0.99).item()
            alpha_values.append(alpha_value)
            print(f"Encoder {i+1}: alpha = {alpha_value:.6f}")

        # check repeat alpha values

        # unique_alphas = sorted(set(alpha_values),reverse=True)
        unique_alphas = deduplicate_alphas(alpha_values, delta=0.001)
        if len(unique_alphas) < len(alpha_values):
            print(f"Detected duplicates in alphas → unique = {unique_alphas}")
            opt['num_encoders'] = len(unique_alphas) + 1 # plus the ODE encoder

            for idx, a in enumerate(unique_alphas, start=2):
                opt[f'alpha_ode_{idx}'] = a
                orig_lr = opt[f'alpha_lr_{idx}']
                opt[f'alpha_lr_{idx}'] = orig_lr * 0.01
            print(f"Re-training with num_encoders={opt['num_encoders']} and alphas={1, unique_alphas}")
            repeat = True

        else:
            repeat = False


    results_lg = []
    # if ep_train_times and ep_infer_times:
    #     print(
    #         f"[Timing/epoch] "
    #         f"train_avg={np.mean(ep_train_times):.4f}s "
    #         f"(median={np.median(ep_train_times):.4f}s), "
    #         f"infer_avg={np.mean(ep_infer_times):.4f}s "
    #         f"(median={np.median(ep_infer_times):.4f}s)"
    #     )
    classifier_seed = opt['classifier_seed']
    if opt['classifier'] == 'linear':

        for _ in range(opt['classifier_time']):
            set_seed(seed=classifier_seed)
            test_result = test_classifier(encoder_model, data, classifier, classifier_optimizer, opt)
            results_lg.append(test_result)
            classifier_seed += 1
    else:

        for class_time in range(opt['classifier_time']):
            set_seed(seed=classifier_seed)
            test_result = test(encoder_model, data, opt, class_time)
            classifier_seed += 1
            # print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
            results_lg.append(test_result['acc'])
            # results_lg_class.append(test_result['class_accuracies'])

    test_mean = np.mean(np.array(results_lg))
    test_std = np.std(np.array(results_lg))
    print(f'(E): Best test_mean={test_mean:.4f}, std={test_std:.4f}')
    best_opt = opt
    return test_mean, test_std, best_opt, alpha_values


def merge_cmd_args(cmd_opt, opt):
    if cmd_opt['beltrami']:
        opt['beltrami'] = True
    if cmd_opt['function'] is not None:
        opt['function'] = cmd_opt['function']
    if cmd_opt['block'] is not None:
        opt['block'] = cmd_opt['block']
    if cmd_opt['attention_type'] != 'scaled_dot':
        opt['attention_type'] = cmd_opt['attention_type']
    if cmd_opt['self_loop_weight'] is not None:
        opt['self_loop_weight'] = cmd_opt['self_loop_weight']
    if cmd_opt['method'] is not None:
        opt['method'] = cmd_opt['method']
    if cmd_opt['step_size'] != 1:
        opt['step_size'] = cmd_opt['step_size']
    if cmd_opt['time'] is not None:
        opt['time'] = cmd_opt['time']
    if cmd_opt['epoch'] != 100:
        opt['epoch'] = cmd_opt['epoch']
    if not cmd_opt['not_lcc']:
        opt['not_lcc'] = False
    if cmd_opt['num_splits'] != 1:
        opt['num_splits'] = cmd_opt['num_splits']
    if cmd_opt['dropout'] is not None:
        opt['dropout'] = cmd_opt['dropout']
    if cmd_opt['hidden_dim'] is not None:
        opt['hidden_dim'] = cmd_opt['hidden_dim']
    if cmd_opt['decay'] is not None:
        opt['decay'] = cmd_opt['decay']
    if cmd_opt['self_loop_weight'] is not None:
        opt['self_loop_weight'] = cmd_opt['self_loop_weight']
    if cmd_opt['edge_homo'] != 0:
        opt['edge_homo'] = cmd_opt['edge_homo']

    if cmd_opt['lr'] is not None:
        opt['lr'] = cmd_opt['lr']
    if cmd_opt['input_dropout'] is not None:
        opt['input_dropout'] = cmd_opt['input_dropout']
    if cmd_opt['heads'] is not None:
        opt['heads'] = cmd_opt['heads']


if __name__ == '__main__':

    import warnings

    warnings.filterwarnings("ignore")

    # setting arguments
    parser = argparse.ArgumentParser('GGD')
    parser.add_argument('--classifier_epochs', type=int, default=1000, help='classifier epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--patience', type=int, default=500, help='Patience')
    parser.add_argument('--l2_coef', type=float, default=0.0, help='l2 coef')
    parser.add_argument('--drop_prob', type=float, default=0.0, help='Tau value')
    parser.add_argument('--dataset', type=str, default='wikics', help='Dataset name: cora, citeseer, pubmed, cs, phy')
    parser.add_argument('--num_hop', type=int, default=0, help='graph power')
    parser.add_argument('--cuda', type=int, default=3, help='cuda')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--classifier_lr', type=float, default=0.005, help='classifier_lr.')
    parser.add_argument('--classifier_decay', type=float, default=0.0005, help='classifier_decay.')
    parser.add_argument('--classifier_seed', type=int, default=10, help='classifier_seed.')
    parser.add_argument('--classifier_time', type=int, default=2, help='classifier_time.')
    parser.add_argument('--classifier', type=str, default=None, help='classifier_time.')

    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension.')
    parser.add_argument('--proj_dim', type=int, default=256, help='proj_dim dimension.')
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                        help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.4, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate.')
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')

    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT')
    parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                        help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                        help='If try get rid of alpha param and the beta*x0 source term')

    # ODE args
    parser.add_argument('--time', type=float, default=3.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, default='euler',
                        help="set the numerical solver: dopri5, euler, rk4, midpoint, predictor")
    parser.add_argument('--step_size', type=float, default=1.0,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                             "used if getting OOM errors at test time")

    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')

    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                        help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=16,
                        help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                        help="multiply attention scores by edge weights before softmax")
    parser.add_argument('--attention_type', type=str, default="scaled_dot",
                        help="scaled_dot,cosine_sim,pearson, exp_kernel")
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

    parser.add_argument('--data_norm', type=str, default='gcn',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')

    parser.add_argument('--runtime', type=int, default=1, help='run time')
    parser.add_argument('--loss', type=str, default='cossum', help='loss function,cosmean,cossum,eucmean,eucsum')
    parser.add_argument('--random_splits', dest='random_splits', action='store_true',
                        help='random_splits')

    # FDE args
    parser.add_argument('--num_encoders', type=int, default=2, help='Number of gconvs.')
    parser.add_argument('--alpha_ode', type=float, default=1.0, help='alpha value')
    # parser.add_argument('--alpha_ode_1', type=float, default=1.0, help='alpha value')
    # parser.add_argument('--alpha_ode_2', type=float, default=0.1, help='alpha value')
    # parser.add_argument('--alpha_ode_3', type=float, default=0.5, help='alpha value')
    # parser.add_argument('--alpha_ode_4', type=float, default=0.5, help='alpha value')
    # parser.add_argument('--alpha_ode_5', type=float, default=1.0, help='alpha value')
    # parser.add_argument('--alpha_ode_6', type=float, default=0.1, help='alpha value')
    # parser.add_argument('--alpha_ode_7', type=float, default=0.5, help='alpha value')
    # parser.add_argument('--alpha_ode_8', type=float, default=0.5, help='alpha value')
    # parser.add_argument('--alpha_ode_9', type=float, default=0.5, help='alpha value')
    # parser.add_argument('--alpha_ode_10', type=float, default=0.5, help='alpha value')

    parser.add_argument('--alpha_lr_1', type=float, default=1e-4, help='Alpha Learning rate.')
    parser.add_argument('--alpha_lr_2', type=float, default=1e-4, help='Alpha Learning rate.')
    parser.add_argument('--alpha_lr_3', type=float, default=1e-4, help='Alpha Learning rate.')
    parser.add_argument('--alpha_lr_4', type=float, default=1e-4, help='Alpha Learning rate.')
    parser.add_argument('--alpha_lr_5', type=float, default=1e-4, help='Alpha Learning rate.')
    parser.add_argument('--alpha_lr_6', type=float, default=1e-4, help='Alpha Learning rate.')
    parser.add_argument('--alpha_lr_7', type=float, default=1e-4, help='Alpha Learning rate.')
    parser.add_argument('--alpha_lr_8', type=float, default=1e-4, help='Alpha Learning rate.')
    parser.add_argument('--alpha_lr_9', type=float, default=1e-4, help='Alpha Learning rate.')
    parser.add_argument('--alpha_lr_10', type=float, default=1e-4, help='Alpha Learning rate.')


    # parser.add_argument('--fractional', action='store_true', help="Use fractional GNNs")
    parser.add_argument('--a1', type=float, default=0.5, help='weighted of FDE')
    parser.add_argument('--a2', type=float, default=0.5, help='weighted of FDE')
    parser.add_argument('--a3', type=float, default=0.5, help='weighted of FDE')
    parser.add_argument('--a4', type=float, default=0.5, help='weighted of FDE')
    parser.add_argument('--a5', type=float, default=0.5, help='weighted of FDE')
    parser.add_argument('--a6', type=float, default=0.5, help='weighted of FDE')
    parser.add_argument('--a7', type=float, default=0.5, help='weighted of FDE')
    parser.add_argument('--a8', type=float, default=0.5, help='weighted of FDE')
    parser.add_argument('--a9', type=float, default=0.5, help='weighted of FDE')
    parser.add_argument('--a10', type=float, default=0.5, help='weighted of FDE')

    # loss function regularization

    parser.add_argument('--lambda', type=float, default=0.1, help='regularization parameter')

    # gread args
    parser.add_argument('--reaction_term', type=str, default='bspm', help='bspm, fisher, allen-cahn')
    parser.add_argument('--beta_diag', type=eval, default=False)

    # equal distance for setting
    args, _ = parser.parse_known_args()
    num_encoders = args.num_encoders
    alphas = np.linspace(0.01, 1.0, num=num_encoders)[::-1]
    for i, alpha in enumerate(alphas):
        parser.add_argument(f'--alpha_ode_{i+1}', type=float, default=alpha, help='alpha value')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    cmd_opt = vars(args)
    try:
        best_opt = best_params_dict[cmd_opt['dataset']]
        opt = {**cmd_opt, **best_opt}
        merge_cmd_args(cmd_opt, opt)
    except KeyError:
        opt = cmd_opt
    #
    # opt = cmd_opt
    test_result = []

    timestr = time.strftime("%H%M%S")
    # create a folder to store the log
    if not os.path.exists("log_multi"):
        os.makedirs("log_multi")
    filename = "log_multi/" + str(args.dataset) + str(args.function) + str(args.block)+ "num_encoders_" + str(args.num_encoders) + "_" + str(args.loss) + "_" + str(args.time) + timestr + ".txt"

    command_args = " ".join(sys.argv)
    with open(filename, 'a') as f:
        json.dump(command_args, f)
        f.write("\n")

    for _ in range(args.runtime):
        test_acc, test_std, opt_final, alpha_values = main(opt)
        test_result.append(test_acc)
        with open(filename, 'a') as f:
            f.write("test_acc: " + str(test_acc) + "\n")
            f.write("test_std: " + str(test_std) + "\n")
            f.write("alpha_values: " + ", ".join(f"{α:.6f}" for α in alpha_values) + "\n\n")

    acc_mean = np.mean(np.array(test_result))
    acc_std = np.std(np.array(test_result))
    print(f'(Final): Best test mean={acc_mean:.4f}, std={acc_std:.4f}')
    with open(filename, 'a') as f:
        f.write("test_acc_mean: " + str(acc_mean) + "\n")
        f.write("test_acc_std: " + str(acc_std) + "\n")
        json.dump(opt_final, f, indent=2)
        f.write("\n")
    # change saved filename to the include the best test acc
    os.rename(filename, filename[:-4] + str(acc_mean) + ".txt")


