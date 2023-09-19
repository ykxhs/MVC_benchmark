import argparse
import random

import numpy as np
import qpsolvers
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from qpsolvers import solve_qp
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.metrics import normalized_mutual_info_score

from dataloader import load_data

SERVERDATAPATH = "D:/cyy/dataset/MVC_data/"
seed = 10
# BDGP
# MNIST-USPS
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
Dataname = "Caltech-2V"
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset,datapath=SERVERDATAPATH)
full_data,_ = dataset.full_data()
for v in range(view):
    full_data[v] = ((full_data[v] - full_data[v].min())/(full_data[v].max() - full_data[v].min())).numpy()
    print(full_data[v].shape)

k = class_num # k is the number of clusters
alpha = 10000 # alpha is the regularisation term in the convex optimisation problem
m = 100 # here m is the number of anchors for each view
n = full_data[0].shape[0]

A=[]
for v in range(view):
    k_means = KMeans(random_state=25, n_clusters=m, max_iter=1000, n_init = 20)
    k_means.fit(full_data[v])
    A.append(k_means.cluster_centers_.T)

for v in range(view):
    print(f'View {v}')
    AA = 2 * alpha * np.eye(m) + 2 * A[v].T @ A[v]
    AA = (AA + AA.T) / 2
    B = full_data[v].T

    d = B.shape[0]

    ff = -2 * (B[:, 0].reshape(d, 1)).T @ A[v]
    q = (ff.T).reshape((m,))
    G = -1 * np.eye(m)
    h = np.zeros((m, 1)).reshape((m,))
    AI = np.ones((m, 1)).reshape((m,))
    b = np.array([1.])

    Z = solve_qp(AA, q, G, h, AI, b,solver="osqp").reshape(m, 1)
    for j in range(1, n):
        ff = -2 * (B[:, j].reshape(d, 1)).T @ A[v]
        q = (ff.T).reshape((m,))

        z = solve_qp(AA, q, G, h, AI, b,solver="osqp").reshape(m, 1)
        Z = np.concatenate((Z, z), axis=1)

    D = np.diag(np.divide(1, np.sqrt(np.sum(Z, axis=1))))
    Zc = (Z.T @ D).T

    if v == 0:
        Sbar = Zc / np.sqrt(view)
    else:
        Sbar = np.concatenate((Sbar, (1 / np.sqrt(view)) * Zc), axis=0)

    print(Sbar.shape)

U, _, _ = randomized_svd(Sbar.T, n_components = k)
k_means2 = KMeans(random_state=25, n_clusters=k, max_iter=1000, n_init=20)
k_means2.fit(U)
k_means2.labels_ + 1
pred = k_means2.labels_
_,orig = dataset.full_data()
true_labels = orig.flatten()
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    if type(label) != np.ndarray:
        label = label.numpy()
    # pred = pred.numpy()
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur

NMI,ARI,ACC,PUR = evaluate(true_labels,pred)
print("{:.4f}\t{:.4f}\t{:.4f}".format(ACC,NMI,PUR))