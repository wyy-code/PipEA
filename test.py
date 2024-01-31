# %%

import warnings
import scipy.sparse as sp
warnings.filterwarnings('ignore')

import os
import numpy as np
from utils import *
from tqdm import *
from gutils import refina_batch
import torch
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7  
# session = tf.Session(config=config)

seed = 12306
np.random.seed(seed)



train_pair, dev_pair,adj_p_1, adj_p_2 = load_data_deal("./data/EN_FR_100K_V2/",train_ratio=0.01)

features = torch.tensor(np.load("/data/user/wyy/Dual-AMN/EN_FR_100K_V2_feature.npy"), dtype=torch.float32)

sims = torch.matmul(features[::2,],features[1::2,].transpose(1,0)).numpy()
del features

print(adj_p_1.shape)
print(adj_p_2.shape)
sims = refina_batch(adj_p_1, adj_p_2, sims, k=8) 

sims = sims[dev_pair[:,0]*0.5][:,(dev_pair[:,1]-1)*0.5]
sims = sims.numpy()
# print(sims.shape)
print("------------------------------------")
print("Begin test align ...")
    

sims = np.exp(sims*50)
for k in range(10):
    sims = sims / np.sum(sims,axis=1,keepdims=True)
    sims = sims / np.sum(sims,axis=0,keepdims=True)
test(sims,"sinkhorn")
