import numpy as np
import sklearn
from sklearn import decomposition
import torch
from scipy import linalg
import scipy.sparse as sp
import math
import torch
import tensorflow as tf
import torch.nn as nn
from torch.nn.parameter import Parameter
import heapq
import sys
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def convert_to_64bit_indices(A):
    A.indptr = np.array(A.indptr, copy=False, dtype=np.int64)
    A.indices = np.array(A.indices, copy=False, dtype=np.int64)
    return A
    
def simple_randomized_torch_svd(B, k):
    _, n = B.size()
    rand_matrix = torch.rand((n,k), dtype=torch.float64).to(device)   
    Q, _ = torch.qr(B @ rand_matrix)                                # qr decomposition
    Q.to(device)
    smaller_matrix = (Q.transpose(0, 1) @ B).to(device)
    U_hat, s, V = torch.svd(smaller_matrix, True)                   # matrix decompostion
    U_hat.to(device)
    U = (Q @ U_hat)

    return U @ (s.pow(0.5).diag()), V @ (s.pow(0.5).diag())     # for link prediction


def computeP4svd(prob, hi, threshold=1e-5, niter=8,alpha=0.5):
    # hi = torch.tensor(identity,dtype=torch.float)
    # prob = torch.tensor(prob,dtype=torch.float)
    prx_mat = hi * alpha
    print("begin SVD iter...")
    for i in range(niter):
        # hi = (prob @ hi) * (1 - alpha)
        hi = (prob@hi) * (1 - alpha)
        prx_mat += hi * alpha
        print(f"before SVD iter{i}")
    prx_mat /= threshold
    prx_mat = prx_mat.todense()
    prx_mat[prx_mat < 1] = 1.
    # prx_mat_log = torch.from_numpy(prx_mat)
    prx_mat = torch.from_numpy(prx_mat)
    prx_mat_log = prx_mat.log().to_sparse().requires_grad_(False)
    # U, V = simple_randomized_torch_svd(prx_mat_log, 128)
    print("begin torch SVD...")
    U, sigma, V = torch.svd_lowrank(prx_mat_log, q=prx_mat.shape[0]*0.01)
    U = U @ (sigma.pow(0.5).diag())

    return U.numpy()
    
def computeP4svd_topk(prob, hi, threshold=1e-5, niter=8,alpha=0.5):
    # hi = torch.tensor(identity,dtype=torch.float)
    # prob = torch.tensor(prob,dtype=torch.float)
    prx_mat = hi * alpha
    prx_mat = convert_to_64bit_indices(prx_mat)
    print("begin SVD iter...")
    for i in range(niter):
        # hi = (prob @ hi) * (1 - alpha)
        hi = (prob@hi) * (1 - alpha)
        prx_mat += hi * alpha
        print(f"before SVD iter{i}")
    prx_mat /= threshold
    prx_mat = prx_mat.todense()
    prx_mat[prx_mat < 1] = 1.
    #prx_mat[prx_mat < threshold] = 0.
    # prx_mat = torch.from_numpy(prx_mat)
    prx_mat -= 1.
    prx_mat = sp.coo_matrix(prx_mat)
    prx_mat = prx_mat.log1p()
    values = prx_mat.data
    indices = np.vstack((prx_mat.row, prx_mat.col))   
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = prx_mat.shape    
    prx_mat = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    #prx_mat_log = prx_mat.log().to_sparse().requires_grad_(False)
    # U, V = simple_randomized_torch_svd(prx_mat_log, 128)
    print("begin torch SVD...")
    U, sigma, V = torch.svd_lowrank(prx_mat, q=128)
    U = U @ (sigma.pow(0.5).diag())

    return U.numpy()
    
def computeP4svd_sparse(ppr, threshold=1e-5, niter=8,alpha=0.5):
    prx_mat = ppr/threshold
    prx_mat = prx_mat.todense()
    prx_mat[prx_mat < 1] = 1.
    # prx_mat_log = torch.from_numpy(prx_mat)
    # prx_mat = torch.from_numpy(prx_mat)
    prx_mat -= 1.
    prx_mat = sp.coo_matrix(prx_mat)
    prx_mat = prx_mat.log1p()
    values = prx_mat.data
    indices = np.vstack((prx_mat.row, prx_mat.col))   
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = prx_mat.shape    
    prx_mat = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    print("begin torch SVD...")
    U, sigma, V = torch.svd_lowrank(prx_mat, q=128)
    U = U @ (sigma.pow(0.5).diag())

    return U.numpy()


def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file_name,'r'):
        head,r,tail = [int(item) for item in line.split()]
        entity.add(head); entity.add(tail); rel.add(r+1)
        triples.append((head,r+1,tail))
    return entity,rel,triples


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj)

def get_adj_p_matrix(triples, entity, rel):
    if triples[0][0]%2 == 0:
        ent_size = int(max(entity)/2 + 1)
        adj_p = sp.lil_matrix((ent_size, ent_size))
        for h, r, t in triples:    
            # adj_matrix[h, t] = 1;
            # adj_matrix[t, h] = 1;
            adj_p[h/2, t/2] = 1;
            adj_p[t/2, h/2] = 1;
    else:
        ent_size = int((max(entity)-1)/2 + 1)
        adj_p = sp.lil_matrix((ent_size, ent_size))
        for h, r, t in triples:    
            # adj_matrix[h, t] = 1;
            # adj_matrix[t, h] = 1;
            adj_p[(h-1)/2, (t-1)/2] = 1;
            adj_p[(t-1)/2, (h-1)/2] = 1;
    #adj_matrix = sp.lil_matrix((ent_size, ent_size))

    adj_p = normalize_adj(adj_p)
    return adj_p

def compute_adj_p(lang):
    entity1, rel1, triples1 = load_triples(lang + 'triples_1')
    entity2, rel2, triples2 = load_triples(lang + 'triples_2')
    adj_p_1 = get_adj_p_matrix(triples1,entity1,rel1)
    adj_p_2 = get_adj_p_matrix(triples2,entity2,rel2)
    return adj_p_1,adj_p_2

def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # dot
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def get_k_max(array,k):
    _k_sort = np.argpartition(array,-k)[-k:]
    return array[_k_sort]

def top_k_matrix(ori_matrix, k):
    for i in range(len(ori_matrix)):
        threshold = np.min(get_k_max(ori_matrix[i],k))
        if threshold<0.5:
            # print("The threshold is:",threshold)
            threshold = np.min(get_k_max(ori_matrix[i],k+1))
        ori_matrix[i][ori_matrix[i]<threshold] = 0

    ori_matrix = normalize_adj(ori_matrix)
    # print(ori_matrix.todense())
        
    return ori_matrix.todense()
    
def top_k_matrix_sparse(ori_matrix, k):
    for i in range(len(ori_matrix)):
        threshold = torch.min(ori_matrix[i].topk(k, sorted=True)[0])
        ori_matrix[i][ori_matrix[i]<threshold] = 0
    ori_matrix = torch.nn.functional.normalize(ori_matrix, p=2, dim=1)
    # print(ori_matrix.todense())
        
    return ori_matrix


def compute_P(adj_1, adj_2, E1, E2, alpha=0.5, k=5):
    # get original KG1 to KG2 matrix
    # cos_12 = get_cos_similar_matrix(E1, E2)

    cos_12 = np.maximum(0,sklearn.metrics.pairwise.cosine_similarity(E1, E2))
    cos_21 = cos_12.T

    # get topK and normalize
    cos_12 = top_k_matrix(cos_12,k)
    cos_21 = top_k_matrix(cos_21,k)

    # stack 4 parts
    adj_1 = np.concatenate((adj_1*alpha,cos_12*(1-alpha)),axis=1)
    adj_2 = np.concatenate((cos_21*(1-alpha), adj_2*alpha),axis=1)
    adj_1 = sp.csr_matrix(adj_1)
    adj_2 = sp.csr_matrix(adj_2)
    # P = np.concatenate((adj_1,adj_2),axis=0)
    P = sp.vstack((adj_1,adj_2))

    return P

def reshape_P(P):
    p1 = P[:int(P.shape[0]/2),]
    p2 = P[int(P.shape[0]/2):,]
    new_p = np.zeros_like(P)
    new_p[::2,] = p1
    new_p[1::2,] = p2

    return new_p

def refina(a1, a2, M, train_pair=None ,k=8):
    a1, a2 = torch.tensor(a1, dtype=torch.float32).to_sparse().requires_grad_(False), torch.tensor(a2,dtype=torch.float32).to_sparse().requires_grad_(False)
    print(a1.dtype,a2.dtype,M.dtype)
    M = torch.tensor(M, dtype=torch.float32)
    for i in range(k):
        M = torch.mul(M.requires_grad_(False), torch.sparse.mm(torch.sparse.mm(a1,M).to_sparse(),a2).to_dense())
        M = M + 1e-5
        M = torch.nn.functional.normalize(M, p=2, dim=1)
        M = torch.nn.functional.normalize(M, p=2, dim=0)
        M = train_sims(pair, M)
        print("Refina in iter {}".format(i))
    return M

def csr_to_torch_sparse(csr_matrix):
    csr_matrix = csr_matrix.tocoo()
    values = csr_matrix.data
    indices = np.vstack((csr_matrix.row, csr_matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = csr_matrix.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).requires_grad_(False)

def refina_batch(a1, a2, M, k=8, batch_size=5000):
    a1, a2 = csr_to_torch_sparse(a1), csr_to_torch_sparse(a2).to_dense()
    print(a1.dtype,a2.dtype,M.dtype)
    M = torch.tensor(M, dtype=torch.float32).requires_grad_(False)

    # 计算批处理的次数（列方向）
    n_batches = M.size(1) // batch_size
    if M.size(1) % batch_size != 0:
        n_batches += 1

    for i in range(k):
        M_new = torch.zeros_like(M)
        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, M.size(1))

            # 处理每个批次
            AMA = torch.sparse.mm(a1, M).to_sparse();
            AMA = torch.mm(AMA, a2[:, start_idx:end_idx])
            M = torch.mul(M[:, start_idx:end_idx].requires_grad_(False), AMA)
            M += 1e-5
            M_new[:, start_idx:end_idx] = M

        M = M_new
        M = torch.nn.functional.normalize(M, p=2, dim=1)
        M = torch.nn.functional.normalize(M, p=2, dim=0)
        print("Refina in iter {}".format(i))

    return M

def remake_adj_1(pair, adj):
    head = []
    tail = []
    for h,t in pair:
        head.append(int(h/2))
        tail.append(int((t-1)/2))
    adj = np.delete(adj,head,axis=0)
    adj = np.delete(adj,tail,axis=1)
    return adj

def remake_adj_2(pair, adj):
    head = []
    tail = []
    for h,t in pair:
        head.append(int(h/2))
        tail.append(int((t-1)/2))
    adj = np.delete(adj,tail,axis=0)
    adj = np.delete(adj,head,axis=1)
    return adj

def train_sims(pair, sims):
    for h,t in pair:
        sims[int(h/2)] = 0.
        sims[:,int((t-1)/2)] = 0.
        sims[int(h/2),int((t-1)/2)] = 1.
    return sims


def get_dev_sims(test_pair, sims):
    print("Begin get dev sims ...")
    sims = tf.gather(indices=test_pair[:,0],params=sims, axis=0)
    sims = tf.gather(indices=test_pair[:,1],params=sims, axis=1)
    return sims

def get_dev_torch_sims(test_pair, sims):
    row = torch.tensor(test_pair[:,0])
    col = torch.tensor(test_pair[:,1])
    sims_row = tf.gather(indices=test_pair[:,0],params=sims, axis=0)
    dev_sims = tf.gather(indices=test_pair[:,1],params=sims_row, axis=1)
    return dev_sims
