import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import all the requirements

from utils import *
import faiss
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import networkx as nx
import random
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

SEED = 12306
tf.random.set_seed(SEED)  # 为TensorFlow设置随机种子 [citation:3]
np.random.seed(SEED)      # 为NumPy设置随机种子 [citation:2][citation:5][citation:8]
random.seed(SEED)         # 为Python内置random模块设置随机种子 [citation:4]
os.environ['PYTHONHASHSEED'] = str(SEED)  # 固定Python哈希种子 [citation:4]
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

dataset = ["DBP_ZH_EN/", "DBP_JA_EN/", "DBP_FR_EN/", "SRPRS_FR_EN/", "SRPRS_DE_EN/", "DBP_WD/", "DBP_YG/"][0]
path = "./EA_datasets/" + dataset

# set hyper-parameters, load graphs and pre-aligned entity pairs
# if your GPU is out of memory, try to reduce the ent_dim
# new generated pairs = 6962
# rest pairs = 3538
# Hits@1: 0.774 Hits@10: 0.919 MRR: 0.826

# new generated pairs = 6962
# rest pairs = 3538
# Hits@1: 0.773 Hits@10: 0.919 MRR: 0.826

ent_dim, depth, top_k = 1024, 2, 500
if "EN" in dataset:
    rel_dim, mini_dim = ent_dim // 2, 16
else:
    rel_dim, mini_dim = ent_dim // 3, 16

node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel = load_graph(path)

# G = nx.Graph()
# print(ent_ent)
# for edge in ent_ent:
#     node1, node2 = edge
#     G.add_edge(node1, node2)
#     G.add_edge(node2, node1)

G = nx.Graph()
G.add_nodes_from(range(node_size))
G.add_edges_from(ent_ent)


train_pair, test_pair = load_aligned_pair(path, ratio=0.3)
candidates_x, candidates_y = set([x for x, y in test_pair]), set([y for x, y in test_pair])

#################
features1 = np.load("original.npy")
features2 = np.load("ours.npy")
test2(G, train_pair,test_pair, features1 ,features2, top_k)
################
# main functions of LightEA

def random_projection(x, out_dim):
    random_vec = K.l2_normalize(tf.random.normal((x.shape[-1], out_dim)), axis=-1)
    return K.dot(x, random_vec)


def batch_sparse_matmul(sparse_tensor, dense_tensor, batch_size=128, save_mem=False):
    results = []
    for i in range(dense_tensor.shape[-1] // batch_size + 1):
        temp_result = tf.sparse.sparse_dense_matmul(sparse_tensor, dense_tensor[:, i * batch_size:(i + 1) * batch_size])
        if save_mem:
            temp_result = temp_result.numpy()
        results.append(temp_result)
    if save_mem:
        return np.concatenate(results, -1)
    else:
        return K.concatenate(results, -1)

def get_seed_weight(d1,d2,gamma):
    return gamma ** abs(d1-d2)


def get_features(train_pair, weight_seed, multipliers):

    random_vec = K.l2_normalize(tf.random.normal((len(train_pair), ent_dim)), axis=-1) * weight_seed * multipliers
    ent_feature = tf.tensor_scatter_nd_update(tf.zeros((node_size, ent_dim)), train_pair.reshape((-1, 1)),
                                              tf.repeat(random_vec, 2, axis=0))


    rel_feature = tf.zeros((rel_size, ent_feature.shape[-1]))

    ent_ent_graph = tf.SparseTensor(indices=ent_ent, values=ent_ent_val, dense_shape=(node_size, node_size))
    rel_ent_graph = tf.SparseTensor(indices=rel_ent, values=K.ones(rel_ent.shape[0]), dense_shape=(rel_size, node_size))
    ent_rel_graph = tf.SparseTensor(indices=ent_rel, values=K.ones(ent_rel.shape[0]), dense_shape=(node_size, rel_size))

    ent_list, rel_list = [ent_feature], [rel_feature]
    for i in range(2):
        new_rel_feature = batch_sparse_matmul(rel_ent_graph, ent_feature)
        new_rel_feature = tf.nn.l2_normalize(new_rel_feature, axis=-1)

        new_ent_feature = batch_sparse_matmul(ent_ent_graph, ent_feature)
        new_ent_feature += batch_sparse_matmul(ent_rel_graph, rel_feature)
        new_ent_feature = tf.nn.l2_normalize(new_ent_feature, axis=-1)

        ent_feature = new_ent_feature;
        rel_feature = new_rel_feature
        ent_list.append(ent_feature);
        rel_list.append(rel_feature)

    ent_feature = K.l2_normalize(K.concatenate(ent_list, 1), -1)
    rel_feature = K.l2_normalize(K.concatenate(rel_list, 1), -1)
    rel_feature = random_projection(rel_feature, rel_dim)

    batch_size = ent_feature.shape[-1] // mini_dim
    sparse_graph = tf.SparseTensor(indices=triples_idx, values=K.ones(triples_idx.shape[0]),
                                   dense_shape=(np.max(triples_idx) + 1, rel_size))
    adj_value = batch_sparse_matmul(sparse_graph, rel_feature)

    features_list = []
    for batch in range(rel_dim // batch_size + 1):
        temp_list = []
        for head in range(batch_size):
            if batch * batch_size + head >= rel_dim:
                break
            sparse_graph = tf.SparseTensor(indices=ent_tuple, values=adj_value[:, batch * batch_size + head],
                                           dense_shape=(node_size, node_size))
            feature = batch_sparse_matmul(sparse_graph, random_projection(ent_feature, mini_dim))
            temp_list.append(feature)
        if len(temp_list):
            features_list.append(K.concatenate(temp_list, -1).numpy())
    features = np.concatenate(features_list, axis=-1)

    faiss.normalize_L2(features)

    return features





epochs = 10
s_features = 0
gamma = 0.9
beta = 0.95
T = 20
for epoch in range(epochs):
    print("Round %d start:" % (epoch + 1))
    for x in range(T):
        random_numbers = tf.random.uniform((len(train_pair), 1), minval=0, maxval=1)
        threshold = beta ** x


        multipliers = tf.where(random_numbers < threshold, 1.0, 0.0)
        weight_seed = tf.convert_to_tensor([get_seed_weight(G.degree[item[0]],G.degree[item[1]], gamma) for item in train_pair])

        weight_seed = tf.convert_to_tensor(np.array(weight_seed), dtype=random_numbers.dtype)
        weight_seed = tf.reshape(weight_seed, multipliers.shape)
        s_features += get_features(train_pair,weight_seed,multipliers)

    features = s_features
    if epoch < epochs - 1:
        left, right = list(candidates_x), list(candidates_y)
        index, sims = sparse_sinkhorn_sims(left, right, features, top_k)
        ranks = tf.argsort(-sims, -1).numpy()
        sims = sims.numpy();
        index = index.numpy()

        temp_pair = []
        x_list, y_list = list(candidates_x), list(candidates_y)
        for i in range(ranks.shape[0]):
            if sims[i, ranks[i, 0]] > 0.5:
                x = x_list[i]
                y = y_list[index[i, ranks[i, 0]]]
                temp_pair.append((x, y))

        for x, y in temp_pair:
            if x in candidates_x:
                candidates_x.remove(x);
            if y in candidates_y:
                candidates_y.remove(y);

        print("new generated pairs = %d" % (len(temp_pair)))
        print("rest pairs = %d" % (len(candidates_x)))

        if not len(temp_pair):
            break
        train_pair = np.concatenate([train_pair, np.array(temp_pair)])
        np.save("ours.npy",features)
        raise OSError
    right_list, wrong_list = test(test_pair, features, top_k)