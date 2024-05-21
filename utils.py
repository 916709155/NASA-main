import argparse
import numpy as np
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv, expm
import torch
import random
import networkx as nx
import dgl
from dgl import DGLGraph
from dgl.data import *
import networkx
#from torch_geometric.datasets import Planetoid, Coauthor, Amazon, DBLP, CoraFull

device = 'cuda:0'


def load_npz_to_sparse_graph(file_name):
    with np.load('dataset/' + file_name + '.npz', allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            labels = loader['labels']
        else:
            labels = None

        adj = adj_matrix.todense()
        col, row = np.where(adj > 0)

        feat = attr_matrix.todense()
        num_class = len(set(labels))

        graph = dgl.graph((col, row), num_nodes=adj.shape[0])
        graph.ndata['feat'] = torch.FloatTensor(feat)
        graph.ndata['label'] = torch.LongTensor(labels)

    return graph, num_class


def get_split(g, nclass, train=20, valid=30):
    label = g.ndata['label'].numpy().tolist()

    class_ind = [[] for i in range(nclass)]
    for ind, lab in enumerate(label):
        class_ind[lab].append(ind)

    train_ind = []
    val_ind = []
    test_ind = []

    for i in range(nclass):
        inds = class_ind[i]
        random.shuffle(inds)
        train_ind.extend(inds[:train])
        val_ind.extend(inds[train:train + valid])
        test_ind.extend(inds[train + valid:])

    train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)

    train_mask[torch.LongTensor(train_ind)] = 1
    val_mask[torch.LongTensor(val_ind)] = 1
    test_mask[torch.LongTensor(test_ind)] = 1

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    return g





def load_data(name, split='fix', seed=0, train_percent=0.3, **kwargs):
    random.seed(seed)

    '''
    if name == 'cs':
        dataset = CoauthorCSDataset()
    if name == 'phy':
        dataset = CoauthorPhysicsDataset()
    if name == 'computer':
        dataset = AmazonCoBuyComputerDataset()
    if name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    '''

    if name in ['cora', 'citeseer', 'pubmed']:
        if name == 'cora':
            dataset = CoraGraphDataset(verbose=False)
        if name == 'citeseer':
            dataset = CiteseerGraphDataset(verbose=False)
        if name == 'pubmed':
            dataset = PubmedGraphDataset(verbose=False)
        # if name == 'corafull':
        #     dataset = CoraFull(root='./data/corafull', transform=T.NormalizeFeatures())
        #     data = dataset[0]


        nclass = dataset.num_classes
        graph = dataset[0]

        if split == 'random':
            inds = torch.where(graph.ndata['train_mask'] > 0.)[0].tolist()
            labels = graph.ndata['label']
            bucket = [[] for _ in range(nclass)]

            random.shuffle(inds)
            for ind in inds:
                lab = labels[ind]
                if len(bucket[lab]) >= kwargs['train']:
                    pass
                else:
                    bucket[lab].append(ind)

            new_mask_ind = []
            for i in range(nclass):
                new_mask_ind += bucket[i]

            train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
            train_mask[torch.LongTensor(new_mask_ind)] = 1
            graph.ndata['train_mask'] = train_mask


    if name in ['cs', 'phy', 'computer', 'photo']:
        graph, nclass = load_npz_to_sparse_graph(name)
        graph = get_split(graph, nclass, train=kwargs['train'], valid=kwargs['valid'])

    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    graph = dgl.to_simple(graph, copy_ndata=True)

    return graph, nclass


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


class augmentation:
    def __init__(self, g, feat):
        self.g = g
        self.feat = feat
        self.adj = g.adj_external(scipy_fmt='csr')

    def init(self, diff='', **kwargs):
        adj = self.adj.copy()

        # sparse version
        d = np.array(adj.sum(1))
        dinv = np.power(d, -0.5).flatten()
        dinv = sp.diags(dinv, format='coo')
        at = (dinv.dot(adj)).dot(dinv)

        self.bern = torch.distributions.Bernoulli(torch.tensor([0.5]))

        # iterately calculate ppr
        if diff == 'ppr':
            alpha = kwargs['alpha']
            temp = sp.eye(self.g.num_nodes(), dtype=np.float)
            sadj = temp.copy()
            for i in range(4):
                temp = (1 - alpha) * temp.dot(at)
                sadj = sadj + temp
            sadj = alpha * sadj + (1 - alpha) * temp.dot(at)

        if diff == 'heat':
            sadj = expm(-5.0 * (sp.eye(self.g.num_nodes()) - at))

        if diff == 'order':
            sadj = at.dot(at)

        if diff == 'ar':
            sadj = (sp.eye(self.g.num_nodes()) - kwargs['alpha'] * (sp.eye(self.g.num_nodes()) - at))
            sadj = sadj.todense().I

        '''
        self.add_idx = []
        for i in range(self.g.num_nodes()):
            idx = np.argpartition(a, -4)[-4:]
            idx = np.where(sadj[i, :].toarray()[0] > 0.)[0].tolist()
            self.add_idx.append(idx)
        '''

        self.deg = self.g.in_degrees().tolist()
        self.del_idx = []
        for i in range(self.g.num_nodes()):
            idx = np.where(at[i, :].toarray()[0] > 0.)[0].tolist()
            idx.remove(i)
            self.del_idx.append(idx)

    def random_distribution_choice(self, dist):
        val = random.random()
        cum = 0.
        for ind in range(self.num):
            cum += dist[ind]
            if cum > val:
                return ind

    def normalize(self, adj):
        rowsum = np.array(adj.sum(1))
        r_inv = np.power(rowsum, -1, where=rowsum > 0.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        nadj = r_mat_inv.dot(adj)
        return nadj

    def generate(self, strategy='NRGE', **kwargs):
        with self.g.local_scope():
            if strategy == 'noau':
                return self.g, self.feat

            if strategy == 'dropedge':
                ratio = kwargs['ratio']
                eids = random.sample(range(self.g.num_edges()), int(ratio * self.g.num_edges()))
                aug_g = dgl.remove_edges(self.g, torch.LongTensor(eids))
                return aug_g, self.feat

            if strategy == 'dropnode':
                ratio = kwargs['ratio']
                nids = random.sample(range(self.g.num_nodes()), int(ratio * self.g.num_nodes()))
                aug_feat = self.feat.clone()
                aug_feat[torch.LongTensor(nids), :] = 0.
                return self.g, aug_feat

            if strategy == 'dropout':
                ratio = kwargs['ratio']
                mask = torch.FloatTensor(self.feat.size()).uniform_() > ratio
                aug_feat = self.feat * mask
                return self.g, aug_feat

            if strategy == 'rand_neighbor':
                adj = self.adj.todense()
                aug_ind = []
                counter = 0
                for u in range(adj.shape[0]):
                    inds = np.where(adj[u, :] > 0)[1].tolist()
                    ind = random.choice(inds)
                    if self.g.ndata['label'][u] == self.g.ndata['label'][ind]:
                        counter += 1
                    aug_ind.append(ind)
                return torch.tensor(aug_ind)

            if strategy == 'sample_neighbor':
                non = kwargs['num_neighbor']
                aug_u = []
                aug_v = []
                adj = self.adj.todense()
                for u in range(adj.shape[0]):
                    inds = np.where(adj[u, :] > 0)[1].tolist()
                    ind = random.sample(inds, non)
                    aug_u += [u] * non
                    aug_v += ind
                aug_g = dgl.graph((aug_u, aug_v), num_nodes=self.g.num_nodes())
                return aug_g, self.feat

            if strategy == 'replace':
                add_u = []
                add_v = []
                del_u = []
                del_v = []

                ratio = kwargs['ratio']
                for u in range(self.g.num_nodes()):
                    for v in self.del_idx[u]:
                        if self.bern.sample():
                            # if random.uniform(0, 1) < ratio:
                            del_u += [v, u]
                            del_v += [u, v]

                            vv = random.choice(self.del_idx[v])
                            add_u += [vv, u]
                            add_v += [u, vv]

                if del_u:
                    eids = self.g.edge_ids(torch.tensor(del_u), torch.tensor(del_v))
                    aug_g = dgl.remove_edges(self.g, eids)
                    aug_g.add_edges(torch.tensor(add_u), torch.tensor(add_v))
                    aug_g = dgl.to_simple(aug_g)

                    return aug_g, self.feat
                else:
                    return self.g, self.feat

            if strategy == 'NRGE':
                add_u = []
                add_v = []
                del_u = []
                del_v = []

                G = dgl.to_networkx(self.g)
                motifs = mine_triangle_motifs(G)
                ratio = kwargs['ratio']
                for u in range(self.g.num_nodes()):
                    if u in motifs:
                        pass
                    else:
                        for v in self.del_idx[u]:
                            if self.bern.sample():
                                # if random.uniform(0, 1) < ratio:
                                del_u += [v, u]
                                del_v += [u, v]

                                vv = random.choice(self.del_idx[v])
                                add_u += [vv, u]
                                add_v += [u, vv]

                if del_u:
                    eids = self.g.edge_ids(torch.tensor(del_u), torch.tensor(del_v))
                    aug_g = dgl.remove_edges(self.g, eids)
                    aug_g.add_edges(torch.tensor(add_u), torch.tensor(add_v))
                    aug_g = dgl.to_simple(aug_g)

                    return aug_g, self.feat
                else:
                    return self.g, self.feat

            if strategy == 'NRGE_noNR':
                del_u = []
                del_v = []

                G = dgl.to_networkx(self.g)
                motifs = mine_triangle_motifs(G)
                ratio = kwargs['ratio']
                for u in range(self.g.num_nodes()):
                    if u in motifs:
                        pass
                    else:
                        for v in self.del_idx[u]:
                            if self.bern.sample():
                                # if random.uniform(0, 1) < ratio:
                                del_u += [v, u]
                                del_v += [u, v]

                if del_u:
                    eids = self.g.edge_ids(torch.tensor(del_u), torch.tensor(del_v))
                    aug_g = dgl.remove_edges(self.g, eids)
                    aug_g = dgl.to_simple(aug_g)
                    return aug_g, self.feat
                else:
                    return self.g, self.feat


def mine_triangle_motifs(G):
    S = set()  # 初始化子图集合为空
    for v1, v2 in G.edges():
        # 挖掘节点 v1 和 v2 的邻居节点集合
        Nv1 = set(G.neighbors(v1))
        Nv2 = set(G.neighbors(v2))
        # 如果邻居节点集合有交集，则说明存在三角形子图
        if Nv1.intersection(Nv2):
            # 遍历交集中的每个节点，构建三角形子图并添加到集合 S 中
            for v3 in Nv1.intersection(Nv2):
                S.add(frozenset([v1, v2, v3]))  # 将三个节点组成一个不可变集合
    return S


