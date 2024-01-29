# %%

import warnings

import scipy.sparse
import scipy.sparse as sp
warnings.filterwarnings('ignore')

import os
import keras
import numpy as np
import numba as nb
import sklearn
from utils import *
from gutils import compute_P, computeP4svd, reshape_P, refina, train_sims
from tqdm import *
from evaluate import evaluate
import tensorflow as tf
import keras.backend as K
from keras.layers import *
import random
import numpy as np
from scipy import optimize
from sklearn.metrics.pairwise import cosine_distances


seed = 12306
np.random.seed(seed)
top_k=500
train_pair, dev_pair, adj_matrix, r_index, r_val, adj_features, rel_features,rwr_features,rel_adj_matrix, tools,log_sparse_rel_matrix,adj_p_1, adj_p_2 = load_data("./data/D_W_15_V2/",train_ratio=0.01)
features = np.load('feature.npy')

candidates_x, candidates_y = set([x for x, y in dev_pair]), set([y for x, y in dev_pair])
left, right = list(candidates_x), list(candidates_y)

sims = tf.math.maximum(0,sklearn.metrics.pairwise.cosine_similarity(features[::2,], features[1::2,])).numpy()
P = compute_P(adj_p_1, adj_p_2, sims, alpha=0.7,k=2)
p_features = computeP4svd(P,threshold=1e-5,alpha=0.5)

right_list, wrong_list = test(dev_pair, features, top_k)
# E1 = features[::2,]
# E2 = features[1::2,]
# P = compute_P(adj_p_1, adj_p_2, E1, E2, alpha=0.7,k=2)
# p_r = P.shape[0]
# identity = sp.identity(p_r,dtype="float32")
# p_features = computeP4svd(P,identity,threshold=1e-5,alpha=0.5)
# p_features = reshape_P(p_features)
# p_features = p_features / (np.linalg.norm(p_features, axis=-1, keepdims=True) + 1e-12)
#
# sims = cal_sims(dev_pair,features)
# sims_p = cal_sims(dev_pair,p_features)
# print("Calculate sims")
# sims = np.maximum(0, sklearn.metrics.pairwise.cosine_similarity(E1, E2))
# sims_p = np.maximum(0, sklearn.metrics.pairwise.cosine_similarity(p_features[::2,], p_features[1::2,]))


# sims = torch.tensor(sims)
# sims_p = torch.tensor(sims_p)
# sims = torch.mul(sims, sims_p)
# print(adj_p_1.shape)
# print(adj_p_2.shape)
# sims = train_sims(train_pair, sims)
# print(type(sims))
# print(sims[0][0])
# sims = refina(adj_p_1, adj_p_2, sims, k=8)
#
# sims = sims[dev_pair[:,0]*0.5][:,(dev_pair[:,1]-1)*0.5]
