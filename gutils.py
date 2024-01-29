import numpy as np
import sklearn
from sklearn import decomposition
from scipy import linalg
import scipy.sparse as sp
import math
import torch
import tensorflow as tf



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
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
    U, sigma, V = torch.svd_lowrank(prx_mat_log, q=128)
    U = U @ (sigma.pow(0.5).diag())

    return U.numpy()
    



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



def compute_P(adj_1, adj_2, sims, alpha=0.5, k=5):
    # get original KG1 to KG2 matrix
    # cos_12 = get_cos_similar_matrix(E1, E2)

    cos_12 = sims
    cos_21 = cos_12.T

    # get topK and normalize
    cos_12 = sparse_top_k(cos_12,k)
    cos_21 = sparse_top_k(cos_21,k)

    # stack 4 parts
    adj_1 = np.concatenate((adj_1*alpha,cos_12*(1-alpha)),axis=1)
    adj_2 = np.concatenate((cos_21*(1-alpha), adj_2*alpha),axis=1)
    adj_1 = sp.csr_matrix(adj_1)
    adj_2 = sp.csr_matrix(adj_2)
    # P = np.concatenate((adj_1,adj_2),axis=0)
    P = sp.vstack((adj_1,adj_2))

    return P

def sparse_top_k(sim, k):
   # numpy matrix convert to tensorflow
   sim_tensor = tf.constant(sim, dtype=tf.float32)

   # find the max k value and indices
   _, indices = tf.math.top_k(sim_tensor, k)

   # initial bool mask as False
   mask = tf.fill(tf.shape(sim_tensor), False)

   # update bool mask
   row_indices = tf.range(tf.shape(sim_tensor)[0])[:, tf.newaxis]
   full_indices = tf.reshape(tf.concat([tf.repeat(row_indices, k, axis=1), indices], axis=-1), [-1, 2])
   updates = tf.fill([k * tf.shape(sim_tensor)[0]], True)
   mask = tf.tensor_scatter_nd_update(mask, full_indices, updates)

   # utilize bool mask to the original matrix
   result = tf.where(mask, sim_tensor, tf.zeros_like(sim_tensor))

   # convert to sparse matrix
   # sparse_matrix = tf.sparse.from_dense(result)

   return result.numpy()





def reshape_P(P):
    p1 = P[:int(P.shape[0]/2),]
    p2 = P[int(P.shape[0]/2):,]
    new_p = np.zeros_like(P)
    new_p[::2,] = p1
    new_p[1::2,] = p2

    return new_p

def refina(a1, a2, M, k=8):
    a1, a2 = torch.tensor(a1, dtype=torch.float32).to_sparse().requires_grad_(False), torch.tensor(a2,dtype=torch.float32).to_sparse().requires_grad_(False)
    print(a1.dtype,a2.dtype,M.dtype)
    M = torch.tensor(M, dtype=torch.float32)
    for i in range(k):
        M = torch.mul(M.requires_grad_(False), torch.sparse.mm(torch.sparse.mm(a1,M).to_sparse(),a2).to_dense())
        M = M + 1e-5
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


# if __name__ == '__main__':
#     sim=np.random.rand(4,5)
#     print(sim)
#     k=2
#     sparse_mat=sparse_top_k(sim,k)
#
#     print(sparse_mat)
