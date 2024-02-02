import numpy as np
import sklearn
import scipy.sparse as sp
import torch
import tensorflow as tf
import tensorflow.keras.backend as K


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
    U, sigma, V = torch.svd_lowrank(prx_mat_log, q=300)
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


def normalize_adj_tf(adj):
    adj = tf.convert_to_tensor(adj)
    rowsum = tf.reduce_sum(adj, axis=1)
    d_inv_sqrt = tf.pow(rowsum, -0.5)
    d_inv_sqrt = tf.where(tf.math.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)
    d_mat_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
    return tf.matmul(tf.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


def top_k_matrix_tf(ori_matrix, k):
    ori_matrix = tf.convert_to_tensor(ori_matrix, dtype=tf.float32)

    # 新逻辑: 使用tf.where和tf.nn.top_k直接更新矩阵值
    def update_row(row):
        values, _ = tf.math.top_k(row, k=k, sorted=True)
        min_val = tf.reduce_min(values)
        # 更新条件：值小于第k大的值将被设置为0
        return tf.where(row >= min_val, row, tf.zeros_like(row))
    # 对ori_matrix的每一行应用update_row函数
    updated_matrix = tf.map_fn(update_row, ori_matrix)
    # 归一化处理
    normalized_matrix = normalize_adj_tf(updated_matrix)
    normalized_matrix = tf.sparse.from_dense(normalized_matrix)
    return normalized_matrix


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

def compute_P_sims(adj_1, adj_2, sims, alpha=0.5, k=5):
    # get original KG1 to KG2 matrix
    # cos_12 = get_cos_similar_matrix(E1, E2)

    cos_12 = sims
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

   result = tf.nn.l2_normalize(result, axis=-1)

   # convert to sparse matrix
   sparse_matrix = tf.sparse.from_dense(result)

   return sparse_matrix

def compute_tf_P(adj_1, adj_2, sims, alpha=0.5, k=1):
    # get original KG1 to KG2 matrix
    # cos_12 = get_cos_similar_matrix(E1, E2)

    cos_12 = sims
    cos_21 = cos_12.T

    # get topK and normalize
    cos_12 = top_k_matrix_tf(cos_12,k)
    cos_21 = top_k_matrix_tf(cos_21,k)

    adj_1 = convert_scipy_to_tf_sparsetensor(adj_1)
    adj_2 = convert_scipy_to_tf_sparsetensor(adj_2)

    cos_12 = tf.sparse.map_values(lambda x: x * (1-alpha), cos_12)
    adj_1 = tf.sparse.map_values(lambda x: x * alpha, adj_1)
    cos_21 = tf.sparse.map_values(lambda x: x * (1 - alpha), cos_21)
    adj_2 = tf.sparse.map_values(lambda x: x * alpha, adj_2)

    adj_1 = tf.sparse.concat(axis=-1, sp_inputs=[adj_1, cos_12])
    adj_2 = tf.sparse.concat(axis=-1, sp_inputs=[cos_21, adj_2])

    P = tf.sparse.concat(axis=0, sp_inputs=[adj_1, adj_2])
    P = tf_sparse_to_csr(P)

    return P

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


def tf_sparse_to_csr(sparse_tensor):
    """
    将 TensorFlow 稀疏张量转换为 SciPy CSR 矩阵。

    参数:
    - sparse_tensor: 一个 tf.sparse.SparseTensor 对象。

    返回:
    - 一个 SciPy 的 csr_matrix。
    """

    if not tf.executing_eagerly():
        raise RuntimeError('This function need TensorFlow 2.x under eager execution')

    indices = sparse_tensor.indices.numpy()
    values = sparse_tensor.values.numpy()
    dense_shape = sparse_tensor.dense_shape.numpy()

    csr_mat = sp.csr_matrix((values, (indices[:, 0], indices[:, 1])), shape=dense_shape)

    return csr_mat

def csr_to_torch_sparse(csr_matrix):
    csr_matrix = csr_matrix.tocoo()
    values = csr_matrix.data
    indices = np.vstack((csr_matrix.row, csr_matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = csr_matrix.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).requires_grad_(False)

def convert_scipy_to_tf_sparsetensor(matrix):
    data = matrix.data.astype('float32')
    row = matrix.nonzero()[0]
    col = matrix.nonzero()[1]
    indices = np.vstack((row,col)).T

    return tf.SparseTensor(indices=indices, values=data, dense_shape=matrix.shape)

def batch_sparse_matmul(sparse_tensor,dense_tensor,batch_size = 1000,save_mem = False):
    results = []
    for i in range(dense_tensor.shape[-1]//batch_size + 1):
        temp_result = tf.sparse.sparse_dense_matmul(sparse_tensor,dense_tensor[:, i*batch_size:(i+1)*batch_size])
        if save_mem:
            temp_result = temp_result.numpy()
        results.append(temp_result)
    if save_mem:
        return np.concatenate(results,-1)
    else:
        return K.concatenate(results,-1)

def refina_tf_batch(a1, a2, M, k=8, batch_size=5000):
    a1, a2 = convert_scipy_to_tf_sparsetensor(a1), a2.todense()
    a2 = tf.cast(a2, "float32")
    print(a1.dtype, a2.dtype, M.dtype)

    for i in range(k):
        AMA = batch_sparse_matmul(a1,M)
        AMA = tf.matmul(AMA, a2)
        print(AMA.shape)
        M = tf.math.multiply(M, AMA)
        print(f"M shape is {AMA.shape}")
        M += 1e-5
        M = K.l2_normalize(M,-1)
        M = K.l2_normalize(M, 0)
        print("Refina in iter {}".format(i))

    return M


def refina_batch(a1, a2, M, k=8, batch_size=5000):
    a1, a2 = csr_to_torch_sparse(a1), csr_to_torch_sparse(a2).to_dense()
    print(a1.dtype,a2.dtype,M.dtype)
    M = torch.tensor(M, dtype=torch.float32).requires_grad_(False)

    # 计算批处理的次数（列方向）
    n_batches = M.size(1) // batch_size
    if M.size(1) % batch_size != 0:
        n_batches += 1

    for i in range(k):
        M_new = []
        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, M.size(1))

            # 处理每个批次
            a2_batch = a2[:, start_idx:end_idx]
            print(a2_batch.shape)
            M_batch = M[:, start_idx:end_idx].requires_grad_(False)
            AMA = torch.sparse.mm(a1, M).to_sparse();
            print(AMA.shape)
            AMA = torch.sparse.mm(AMA, a2_batch)
            print(AMA.shape)
            M = torch.mul(M_batch, AMA)
            print(M.shape)
            M += 1e-5
            M_new.append(M)

        M = torch.cat(M_new, dim=-1)
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

if __name__ == "__main__":
    a1 = [[1,0,0],
         [0,1,0],
         [0,0,1]]
    a2 = [[0, 0, 1],
          [0, 1, 0],
          [1, 0, 0]]
    a1 = np.array(a1)
    a2 = np.array(a2)

    a1 = sp.csr_matrix(a1)
    a2 = sp.csr_matrix(a2)
    sims = np.random.rand(3,3)
    print(sims)
    P = compute_tf_P(a1,a2,sims,k=2)
    print(P.todense())

    P = compute_P_sims(a1.todense(), a2.todense(), sims, k=2)
    print(P.todense())
