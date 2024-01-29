import numpy as np
import sklearn
from sklearn import decomposition
from scipy import linalg
import scipy.sparse as sp
import math
import torch
import tensorflow as tf
import tensorly



def computeP4svd(prob, threshold=1e-5, niter=8,alpha=0.5):
    hi = tf.eye(prob.shape[0], dtype="float32")
    prx_mat = hi * alpha
    print("begin SVD iter...")
    for i in range(niter):
        hi = tf.sparse.sparse_dense_matmul(prob, hi) * (1 - alpha)
        prx_mat += hi * alpha
        print(f"before SVD iter{i}")
    prx_mat /= threshold
    prx_mat[prx_mat < 1] = 1.
    # prx_mat_log = torch.from_numpy(prx_mat)
    prx_mat = torch.from_numpy(prx_mat)
    prx_mat_log = prx_mat.log().to_sparse().requires_grad_(False)
    # U, V = simple_randomized_torch_svd(prx_mat_log, 128)
    print("begin torch SVD...")
    U, sigma, _ = tensorly.truncated_svd(prx_mat_log, n_eigenvecs=512)
    # U, sigma, V = torch.svd_lowrank(prx_mat_log, q=128)
    U = U @ (sigma.pow(0.5).diag())

    return U.numpy()
    

def convert_scipy_to_sparsetensor(matrix):
    data = matrix.data.astype('float32')
    row = matrix.nonzero()[0]
    col = matrix.nonzero()[1]
    indices = np.vstack((row,col)).T

    return tf.SparseTensor(indices=indices, values=data, dense_shape=matrix.shape)

def compute_P(adj_1, adj_2, sims, alpha=0.5, k=5):
    # get original KG1 to KG2 matrix
    # cos_12 = get_cos_similar_matrix(E1, E2)

    cos_12 = sims
    cos_21 = cos_12.T

    # get topK and normalize
    cos_12 = sparse_top_k(cos_12,k)
    cos_21 = sparse_top_k(cos_21,k)
    adj_1 = convert_scipy_to_sparsetensor(adj_1)
    adj_2 = convert_scipy_to_sparsetensor(adj_2)

    cos_12 = tf.sparse.map_values(lambda x: x * (1-alpha), cos_12)
    adj_1 = tf.sparse.map_values(lambda x: x * alpha, adj_1)
    cos_21 = tf.sparse.map_values(lambda x: x * (1 - alpha), cos_21)
    adj_2 = tf.sparse.map_values(lambda x: x * (1 - alpha), adj_2)

    adj_1 = tf.sparse.concat(axis=-1, sp_inputs=[adj_1, cos_12])
    adj_2 = tf.sparse.concat(axis=-1, sp_inputs=[cos_21, adj_2])

    return tf.sparse.concat(axis=0, sp_inputs=[adj_1, adj_2])

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
   sparse_matrix = tf.sparse.from_dense(result)

   return sparse_matrix


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

