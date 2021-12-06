import numpy as np


# return (1 , B)
def batched_hr(topk_idx, pos_len):
    return topk_idx.sum(axis=1).astype(np.bool).astype(np.int)  # (1 , B)


def batched_mrr(topk_idx, pos_len):
    sum_idx = np.sum(topk_idx, axis=1)
    pos_idx = topk_idx.argmax(axis=1)  # (1, B)
    result = np.full_like(pos_idx, 0, dtype=np.float)
    for i, (si, idx) in enumerate(zip(sum_idx, pos_idx)):
        result[i] = 0 if si == 0 else 1.0 / (idx + 1)
    return result  # (1 , B)


def batched_recall(topk_idx, pos_len):
    pos_len = pos_len.flatten()
    return topk_idx.sum(axis=1) / pos_len  # (1 , B)


def batched_precision(topk_idx, pos_len):
    return topk_idx.sum(axis=1) / topk_idx.shape[-1]


def batched_f1_score(topk_idx, pos_len):
    p, r = batched_precision(topk_idx, pos_len), batched_recall(topk_idx, pos_len)
    return 2 * p * r / (p + r)


def batched_map(topk_idx, pos_len):
    # (B , topk)
    # (1 , B)
    pos_len = pos_len.flatten()
    topk = topk_idx.shape[-1]  # 推荐列表长度
    topk_arr = np.full(pos_len.shape[-1], topk)  # (1 , B)
    precision_matrix = topk_idx.cumsum(axis=1) / np.arange(1, topk + 1)  # Pre_1 , Pre_2, ... Pre_3
    len = np.where(topk_arr < pos_len, topk_arr, pos_len)  # min(pos_len, topk)
    return (topk_idx * precision_matrix).sum(axis=1) / len


# (B , topk) , (B , 1)
def batched_ndcg(topk_idx, pos_len):
    pos_len = pos_len.flatten()
    topk = topk_idx.shape[-1]  # 推荐列表长度
    topk_arr = np.full(pos_len.shape[-1], topk)  # (1 , B)
    batch_size = topk_idx.shape[0]
    len = np.where(topk_arr < pos_len, topk_arr, pos_len)  # min(pos_len, topk) , (1 , B)
    tmp = np.zeros((batch_size, topk))
    tmp[:, :] = np.arange(1, topk + 1)
    tmp = 1.0 / np.log2(tmp + 1)
    # tmp 用于根据mask求idcg和dcg
    # [
    #   [1 / log2(1) , 1 / log2(2) , 1 / log2(3) , ...]
    #   ....
    # ]

    mask = np.full_like(tmp, 1, dtype=np.float)
    for row, l in enumerate(len):
        mask[row, l:] = 0

    idcg = (tmp * mask).sum(axis=1)  # 计算理想值用于归一化，理想值就是positive的样本全排在最前面
    dcg = (tmp * topk_idx).sum(axis=1)
    return dcg / idcg


metrics_dict = {
    # "hit_rate": batched_hr,
    "mrr": batched_mrr,
    # "recall": batched_recall,
    # "precision": batched_precision,
    # "map": batched_map,
    # "ndcg": batched_ndcg
}

if __name__ == '__main__':
    batched_topk_idx = np.array([
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 0]
    ])
    pos_len = np.array([[3], [4], [2]])
    # pos_len = np.array([3, 4, 2])

    for k in metrics_dict.keys():
        print("%s:" % k)
        print(metrics_dict[k](batched_topk_idx, pos_len))
