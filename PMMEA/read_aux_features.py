
from load import *



import os


def normalize(X, norm='l2', axis=1, copy=True):
    """
    对输入数据进行归一化处理

    参数:
    X: 输入数据，numpy数组或类似数组的结构
    norm: 归一化使用的范数，可选 'l1', 'l2', 'max'，默认为 'l2'
    axis: 归一化的轴，0 表示对特征归一化，1 表示对样本归一化，默认为 1
    copy: 是否创建数据的副本，默认为 True

    返回:
    归一化后的数据
    """
    if copy:
        X = X.copy()

    # 确保 X 是 numpy 数组
    X = np.asarray(X)

    # 处理不同范数类型
    if norm == 'l2':
        norms = np.sqrt(np.sum(X ** 2, axis=axis, keepdims=True))
    elif norm == 'l1':
        norms = np.sum(np.abs(X), axis=axis, keepdims=True)
    elif norm == 'max':
        norms = np.max(np.abs(X), axis=axis, keepdims=True)
    else:
        raise ValueError(f"不支持的范数类型: {norm}")

    # 避免除以零
    norms = np.where(norms == 0, 1, norms)

    # 归一化
    X_normalized = X / norms

    return X_normalized



def load_img_features(ent_num, dataname):

    if "FBDB15K"==dataname:

        img_vec_path = "mmkg\pkls\FBDB15K_id_img_feature_dict.pkl"
    else:
        img_vec_path = "mmkg\pkls\FBYG15K_id_img_feature_dict.pkl"

    img_features = load_img(ent_num, img_vec_path)
    return img_features


def load_aux_features(file_dir="data/DBP15K/zh_en", word_embedding="glove"):
    data_name = file_dir.split("/")[1]
    lang_list = [1, 2]
    ent2id_dict, ills, triples, r_hs, r_ts, ids = read_raw_data(file_dir, lang_list)
    e1 = os.path.join(file_dir, 'ent_ids_1')
    e2 = os.path.join(file_dir, 'ent_ids_2')
    left_ents = get_ids(e1)
    right_ents = get_ids(e2)
    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)
    print("total ent num: {}, rel num: {}".format(ENT_NUM, REL_NUM))
    
    img_features = load_img_features(ENT_NUM, data_name)
    img_features = normalize(img_features)
    print("image feature shape:", img_features.shape)
    
    rel_features = load_relation(ENT_NUM, triples, 1000)
    #rel_features = normalize(rel_features)
    print("relation feature shape:", rel_features.shape)
    
    
    
    # load name/char features (only for DBP15K datasets)
    data_dir, dataname = os.path.split(file_dir)
    if word_embedding == "glove":
      word2vec_path = "data/embedding/glove.6B.300d.txt"
    elif word_embedding == 'fasttext':
      pass
    else:
      raise Exception("error word embedding")


    name_features = None
    char_features = None
    a1 = os.path.join(file_dir, 'training_attrs_1')
    a2 = os.path.join(file_dir, 'training_attrs_2')
    att_features = load_attr([a1, a2], ENT_NUM, ent2id_dict, 1000)  # attr
    att_features = normalize(att_features)
 
    print("attribute feature shape:", att_features.shape)
    
    return img_features, name_features, char_features, att_features, rel_features

