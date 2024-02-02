# %%

import warnings
import scipy.sparse as sp

warnings.filterwarnings('ignore')

import os
import numpy as np
from utils import *
from tqdm import *
from gutils import refina_batch, refina_tf_batch, refina, compute_P, reshape_P, compute_tf_P, computeP4_tf_svd
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# session = tf.Session(config=config)

seed = 12306
np.random.seed(seed)

train_pair, dev_pair, adj_p_1, adj_p_2 = load_data_deal("./data/EN_FR_15K_V2/",train_ratio=0.01)

features = np.load('E:\wyy\Code\Dual-AMN-main/feature.npy')
features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-5)

# features = torch.tensor(features, dtype=torch.float32)
# sims = torch.matmul(features[::2, ], features[1::2, ].transpose(1, 0)).numpy()
sims = tf.matmul(features[::2, ],tf.transpose(features[1::2, ],[1,0])).numpy()

# adj_p_1 = adj_p_1.todense()
# adj_p_2 = adj_p_2.todense()
# P = compute_P(adj_p_1, adj_p_2, features[::2,], features[1::2,], alpha=0.7,k=2)
P = compute_tf_P(adj_p_1, adj_p_2, sims, alpha=0.7,k=2)

del features

print(adj_p_1.shape)
print(adj_p_2.shape)

# ###### Propagation Strategy

p_r = P.shape[0]
identity = sp.identity(p_r,dtype="float32")
# p_features = computeP4svd(P,identity,threshold=1e-5,alpha=0.5)
p_features = computeP4_tf_svd(P,threshold=1e-5,alpha=0.5)
p_features = reshape_P(p_features)
p_features = p_features / (np.linalg.norm(p_features, axis=-1, keepdims=True) + 1e-12)

sims_p = tf.matmul(p_features[::2,],tf.transpose(p_features[1::2,],[1,0]))
sims_p = tf.cast(sims_p, dtype=tf.float32)
del p_features, identity

sims = tf.multiply(sims, sims_p)
del sims_p

# ###### Refine Strategy
# sims = refina_batch(adj_p_1, adj_p_2, sims, k=8)
sims = refina_tf_batch(adj_p_1, adj_p_2, sims, k=10).numpy()

sims = sims[(dev_pair[:, 0] * 0.5).astype('int')][:, ((dev_pair[:, 1] - 1) * 0.5).astype('int')]
# sims = sims.numpy()
# print(sims.shape)

print("------------------------------------")
print("Begin test align ...")


def np_test(sims, mode="sinkhorn", batch_size=1024):
    if mode == "sinkhorn":
        results = []
        for epoch in range(len(sims) // batch_size + 1):
            sim = sims[epoch * batch_size:(epoch + 1) * batch_size]
            rank = tf.argsort(-sim, axis=-1)
            ans_rank = np.array([i for i in range(epoch * batch_size, min((epoch + 1) * batch_size, len(sims)))])
            results.append(tf.where(tf.equal(tf.cast(rank, ans_rank.dtype),
                                             tf.tile(np.expand_dims(ans_rank, axis=1), [1, len(sims)]))).numpy())
        results = np.concatenate(results, axis=0)

        @nb.jit(nopython=True)
        def cal(results):
            hits1, hits10, mrr = 0, 0, 0
            for x in results[:, 1]:
                if x < 1:
                    hits1 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1 / (x + 1)
            return hits1, hits10, mrr

        hits1, hits10, mrr = cal(results)
        print("hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (
        hits1 / len(sims) * 100, hits10 / len(sims) * 100, mrr / len(sims) * 100))
        # return [hits1/len(sims)*100,hits10/len(sims)*100,mrr/len(sims)*100]
    else:
        c = 0
        for i, j in enumerate(sims[1]):
            if i == j:
                c += 1
        print("hits@1 : %.2f%%" % (100 * c / len(sims[0])))

sims = np.exp(sims * 50)
for k in range(10):
    sims = sims / np.sum(sims, axis=1, keepdims=True)
    sims = sims / np.sum(sims, axis=0, keepdims=True)
np_test(sims, "sinkhorn")
