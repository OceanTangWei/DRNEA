import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import faiss
import pickle
import json
from tqdm import tqdm
import tensorflow.keras.backend as K
import random
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")


SEED = 12306
tf.random.set_seed(SEED)  # 为TensorFlow设置随机种子 [citation:3]
np.random.seed(SEED)      # 为NumPy设置随机种子 [citation:2][citation:5][citation:8]
random.seed(SEED)         # 为Python内置random模块设置随机种子 [citation:4]
os.environ['PYTHONHASHSEED'] = str(SEED)  # 固定Python哈希种子 [citation:4]
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def load_graph(path):
    
    if os.path.exists(path+"graph_cache.pkl"):
        return pickle.load(open(path+"graph_cache.pkl","rb"))
    
    triples = []
    with open(path + "triples_1") as f:
        for line in f.readlines():
            h,r,t = [int(x) for x in line.strip().split("\t")]
            triples.append([h,t,2*r])
            triples.append([t,h,2*r+1])
    with open(path + "triples_2") as f:
        for line in f.readlines():
            h,r,t = [int(x) for x in line.strip().split("\t")]
            triples.append([h,t,2*r])
            triples.append([t,h,2*r+1])
    triples = np.unique(triples,axis=0)
    node_size,rel_size = np.max(triples)+1 , np.max(triples[:,2])+1
    
    ent_tuple,triples_idx = [],[]
    ent_ent_s,rel_ent_s,ent_rel_s = {},set(),set()
    last,index = (-1,-1), -1

    for i in range(node_size):
        ent_ent_s[(i,i)] = 0

    for h,t,r in triples:
        ent_ent_s[(h,h)] += 1
        ent_ent_s[(t,t)] += 1

        if (h,t) != last:
            last = (h,t)
            index += 1
            ent_tuple.append([h,t])
            ent_ent_s[(h,t)] = 0

        triples_idx.append([index,r])
        ent_ent_s[(h,t)] += 1
        rel_ent_s.add((r,h))
        ent_rel_s.add((t,r))

    ent_tuple = np.array(ent_tuple)
    triples_idx = np.unique(np.array(triples_idx),axis=0)

    ent_ent = np.unique(np.array(list(ent_ent_s.keys())),axis=0)
    ent_ent_val = np.array([ent_ent_s[(x,y)] for x,y in ent_ent]).astype("float32")
    rel_ent = np.unique(np.array(list(rel_ent_s)),axis=0)
    ent_rel = np.unique(np.array(list(ent_rel_s)),axis=0)
    
    graph_data = [node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel]
    pickle.dump(graph_data, open(path+"graph_cache.pkl","wb"))
    return graph_data

def load_aligned_pair(file_path,ratio = 0.3):
    with open(file_path + "ref_ent_ids") as f:
        ref = f.readlines()
    try:
        with open(file_path + "sup_ent_ids") as f:
            sup = f.readlines()
    except:
        sup = None
    
    ref = np.array([line.replace("\n","").split("\t") for line in ref]).astype(np.int64)
    if sup:
        sup = np.array([line.replace("\n","").split("\t") for line in sup]).astype(np.int64)
        ref = np.concatenate([ref,sup])
    np.random.shuffle(ref)
    train_size = int(ref.shape[0]*ratio)
    return ref[:train_size],ref[train_size:]

def load_name_features(dataset,vector_path,mode = "word-level"):
    
    try:
        word_vecs = pickle.load(open("./word_vectors.pkl","rb"))
    except:
        word_vecs = {}
        with open(vector_path,encoding='UTF-8') as f:
            for line in tqdm(f.readlines()):
                line = line.split()
                word_vecs[line[0]] = [float(x) for x in line[1:]]
        pickle.dump(word_vecs,open("./word_vectors.pkl","wb"))

    if "EN" in dataset:
        ent_names = json.load(open("translated_ent_name/%s.json"%dataset[:-1].lower(),"r"))

    d = {}
    count = 0
    for _,name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word)-1):
                if word[idx:idx+2] not in d:
                    d[word[idx:idx+2]] = count
                    count += 1

    ent_vec = np.zeros((len(ent_names),300),"float32")
    char_vec = np.zeros((len(ent_names),len(d)),"float32")
    for i,name in tqdm(ent_names):
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word)-1):
                char_vec[i,d[word[idx:idx+2]]] += 1
        if k:
            ent_vec[i]/=k
        else:
            ent_vec[i] = np.random.random(300)-0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(d))-0.5
    
    faiss.normalize_L2(ent_vec)
    faiss.normalize_L2(char_vec)

    if mode == "word-level":
        name_feature = ent_vec
    if mode == "char-level":
        name_feature = char_vec
    if mode == "hybrid-level": 
        name_feature = np.concatenate([ent_vec,char_vec],-1)
        
    return name_feature

def sparse_sinkhorn_sims(left,right,features,top_k=500,iteration=15,mode = "test"):
    features_l = features[left]
    features_r = features[right]

    faiss.normalize_L2(features_l); faiss.normalize_L2(features_r)


    dim, measure = features_l.shape[1], faiss.METRIC_INNER_PRODUCT
    if mode == "test":
        param = 'Flat'
        index = faiss.index_factory(dim, param, measure)
    else:
        param = 'IVF256(RCQ2x5),PQ32'
        index = faiss.index_factory(dim, param, measure)
        index.nprobe = 16
    # if len(gpus):
    #
    #     res = faiss.StandardGpuResources()
    #     index = faiss.index_cpu_to_gpu(res, 0, index)
    index.train(features_r)
    index.add(features_r)
    sims, index = index.search(features_l, top_k)
    
    row_sims = K.exp(sims.flatten()/0.02)
    index = K.flatten(index.astype("int32"))

    size = len(left)
    row_index = K.transpose(([K.arange(size*top_k)//top_k,index,K.arange(size*top_k)]))
    col_index = tf.gather(row_index,tf.argsort(row_index[:,1]))
    covert_idx = tf.argsort(col_index[:,2])

    for _ in range(iteration):
        row_sims = row_sims / tf.gather(indices=row_index[:,0],params = tf.math.segment_sum(row_sims,row_index[:,0]))
        col_sims = tf.gather(row_sims,col_index[:,2])
        col_sims = col_sims / tf.gather(indices=col_index[:,1],params = tf.math.segment_sum(col_sims,col_index[:,1]))
        row_sims = tf.gather(col_sims,covert_idx)
        
    return K.reshape(row_index[:,1],(-1,top_k)), K.reshape(row_sims,(-1,top_k))

def test(test_pair,features,top_k=500,iteration=15):
    left, right = test_pair[:,0], np.unique(test_pair[:,1])
    index,sims = sparse_sinkhorn_sims(left, right,features,top_k,iteration,"test")
    ranks = tf.argsort(-sims,-1).numpy()
    index = index.numpy()
    
    wrong_list,right_list = [],[]
    h1,h10,mrr = 0, 0, 0
    pos = np.zeros(np.max(right)+1)
    pos[right] = np.arange(len(right))
    for i in range(len(test_pair)):
        rank = np.where(pos[test_pair[i,1]] == index[i,ranks[i]])[0]
        if len(rank) != 0:
            if rank[0] == 0:
                h1 += 1
                right_list.append(test_pair[i])
            else:
                wrong_list.append((test_pair[i],right[index[i,ranks[i]][0]]))
            if rank[0] < 10:
                h10 += 1
            mrr += 1/(rank[0]+1) 
    print("Hits@1: %.3f Hits@10: %.3f MRR: %.3f\n"%(h1/len(test_pair),h10/len(test_pair),mrr/len(test_pair)))
    
    return right_list, wrong_list


def plot_graph(G, v):
    import networkx as nx
    import matplotlib.pyplot as plt
    subgraph_G_v = nx.ego_graph(G, v, radius=2)

    import matplotlib.colors as mcolors  # 用于颜色处理

    # 假设 subgraph_G_v 和中心节点 v 已经从步骤一中得到

    # --- 1. 确定节点分类和颜色 ---

    # 1-hop 邻居（即 v 的直接邻居）
    one_hop_neighbors = set(G.neighbors(v))

    # 2-hop 邻居（即距离 v 恰好 2 步的节点）
    all_nodes = set(subgraph_G_v.nodes)
    # 从所有节点中排除 v 和 1-hop 邻居
    two_hop_neighbors = all_nodes - one_hop_neighbors - {v}

    # 节点颜色映射
    node_colors = []
    for node in subgraph_G_v.nodes():
        if node == v:
            node_colors.append('#E41A1C')  # 醒目的红色：中心节点
        elif node in one_hop_neighbors:
            node_colors.append('#377EB8')  # 蓝色：1 阶邻居
        else:  # 2-hop 邻居
            node_colors.append('#4DAF4A')  # 绿色：2 阶邻居

    # 节点大小映射
    node_sizes = [800 if node == v else 300 for node in subgraph_G_v.nodes()]

    # 节点标签字典（只显示中心节点和重要节点的标签）
    labels = {node: str(node) if node == v or node in one_hop_neighbors else ''
              for node in subgraph_G_v.nodes()}
    # 可以选择只显示中心节点 v 的标签
    # labels = {v: str(v)}

    # --- 2. 选择布局 ---
    # 使用 Fruchterman-Reingold 力导向布局
    pos = nx.spring_layout(subgraph_G_v, seed=42, k=0.15)

    # --- 3. 绘图 ---
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_title(f"Node {v} 的 2 阶子图 (Ego-Graph)", fontsize=16)

    # 绘制边
    nx.draw_networkx_edges(
        subgraph_G_v,
        pos,
        alpha=0.4,  # 透明度：让边不那么突出，避免杂乱
        edge_color='gray',
        width=1.5
    )

    # 绘制节点
    nx.draw_networkx_nodes(
        subgraph_G_v,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=1.5,  # 节点边框宽度
        edgecolors='black',  # 节点边框颜色
        alpha=0.8
    )

    # 绘制节点标签 (只对中心节点和1阶邻居)
    nx.draw_networkx_labels(
        subgraph_G_v,
        pos,
        labels=labels,
        font_size=12,
        font_weight='bold',
        font_color='black'
    )

    # 隐藏坐标轴
    plt.axis('off')

    # --- 4. 添加图例 (Legend) ---
    # 创建一个自定义图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'中心节点 $v={v}$',
                   markerfacecolor='#E41A1C', markersize=10, markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', label='1 阶邻居',
                   markerfacecolor='#377EB8', markersize=10, markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', label='2 阶邻居',
                   markerfacecolor='#4DAF4A', markersize=10, markeredgecolor='black')
    ]

    # 将图例放置在右上方，不遮挡图形
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)

    # 保存为高质量图片
    plt.savefig("ego_graph_visualization.png", dpi=300, bbox_inches='tight')
    # 建议保存为 .pdf 或 .svg 格式，以获得无损缩放的矢量图，更适合论文
    # plt.savefig("ego_graph_visualization.pdf", bbox_inches='tight')

    plt.show()


def cosine_similarity(X, Y=None):
    """
    计算矩阵中向量之间的余弦相似度

    参数:
    X: 输入矩阵，每行是一个向量
    Y: 可选的第二个矩阵，如果为None则计算X自身的相似度

    返回:
    相似度矩阵
    """
    X = np.array(X)

    if Y is None:
        Y = X

    Y = np.array(Y)

    # 计算范数
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)

    # 避免除以零
    X_norm = np.where(X_norm == 0, 1, X_norm)
    Y_norm = np.where(Y_norm == 0, 1, Y_norm)

    # 归一化
    X_normalized = X / X_norm
    Y_normalized = Y / Y_norm

    # 计算相似度矩阵
    similarity_matrix = np.dot(X_normalized, Y_normalized.T)

    return similarity_matrix
def test2(G,train_pair,test_pair, features1,features2, top_k=500, iteration=15):
    t_nodes1 = train_pair[:,0]
    t_nodes2 = train_pair[:,1]
    train_dict = dict(zip(t_nodes1, t_nodes2))

    left, right = test_pair[:, 0], np.unique(test_pair[:, 1])
    index1, sims1 = sparse_sinkhorn_sims(left, right, features1, top_k, iteration, "test")
    index2, sims2 = sparse_sinkhorn_sims(left, right, features2, top_k, iteration, "test")
    ranks1 = tf.argsort(-sims1, -1).numpy()
    index1 = index1.numpy()

    ranks2 = tf.argsort(-sims2, -1).numpy()
    index2 = index2.numpy()

    wrong_list, right_list = [], []
    h1, h10, mrr = 0, 0, 0
    pos = np.zeros(np.max(right) + 1)
    pos[right] = np.arange(len(right))
    for i in range(len(test_pair)):
        rank1 = np.where(pos[test_pair[i, 1]] == index1[i, ranks1[i]])[0]
        rank2 = np.where(pos[test_pair[i, 1]] == index2[i, ranks2[i]])[0]
        if len(rank1)!=0 and len(rank2)!=0 and rank2[0]==0 and rank1[0]>=1:

            focus_nodes1 = []
            focus_nodes2 = []

            for x in G.neighbors(test_pair[i, 0]):
                if x in train_dict.keys() and train_dict[x] in G.neighbors(test_pair[i, 1]):
                    if abs(G.degree[train_dict[x]]-G.degree[x]) > 3:
                        focus_nodes1.append(x)
                        focus_nodes2.append(train_dict[x])






            if len(focus_nodes1)>0:
                print(focus_nodes1,[G.degree[x] for x in focus_nodes1])
                print(focus_nodes2,[G.degree[x] for x in focus_nodes2])
                print(rank1[0],rank2[0])
                print(test_pair[i])

                sims_score1 = cosine_similarity(features1[test_pair[i, 0]].reshape((1,-1)), features1[test_pair[i, 1]].reshape((1,-1)))
                sims_score2 = cosine_similarity(features2[test_pair[i, 0]].reshape((1,-1)), features2[test_pair[i, 1]].reshape((1,-1)))
                print(sims_score1,sims_score2)
                # plot_graph(G, test_pair[i, 0])
                # plot_graph(G, test_pair[i, 1])
                tmp_dict = {"G":G,"NODE_PAIR":(test_pair[i][0],test_pair[i][1]),"train_pair":[(x, train_dict[x]) for x in focus_nodes1]}
                if test_pair[i,0] == 4270:
                    pickle.dump(tmp_dict,open(r"C:\Users\Administrator\PyCharmMiscProject\ego_graph_visualization.p","wb"))
                    raise EOFError
                print("###############")

    #     if len(rank) != 0:
    #         if rank[0] == 0:
    #             h1 += 1
    #             right_list.append(test_pair[i])
    #         else:
    #             wrong_list.append((test_pair[i], right[index[i, ranks[i]][0]]))
    #         if rank[0] < 10:
    #             h10 += 1
    #         mrr += 1 / (rank[0] + 1)
    # print("Hits@1: %.3f Hits@10: %.3f MRR: %.3f\n" % (h1 / len(test_pair), h10 / len(test_pair), mrr / len(test_pair)))
    #
    # return right_list, wrong_list