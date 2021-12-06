from myrec.config import FileConfig
from myrec.evaluator.abstract_evaluator import AbstractEvaluator
from myrec.utils.evaluator.utils import partition_arg_topK
from myrec.utils.evaluator.metrics import *
from collections import defaultdict
import pandas as pd
from pandas import DataFrame
import numpy as np


class TopKEvaluator(AbstractEvaluator):
    def __init__(self, config, train_ui_inter, test_ui_inter):
        self.config = config
        self.train_ui_inter = train_ui_inter
        self.test_ui_inter = test_ui_inter

    def evaluate(self, model):
        metrics = self.config['metrics']
        topk_iter = TopkIter(self.train_ui_inter, self.test_ui_inter, model, self.config)
        result = defaultdict(list)
        for batched_topk_idx, batched_pos_len in topk_iter:
            for metric in metrics:
                func_name = "batched_" + metric
                result[metric] += list(eval(func_name)(batched_topk_idx, batched_pos_len))
        return result


class TopkIter:
    def __init__(self, train_ui_inter, test_ui_inter, _model, config):
        self.batch_size = config['test_batch_size']
        self.user_id_field = config['user_id_field']
        self.item_id_field = config['item_id_field']
        self.label_field = config['label_field']
        self._model = _model

        self.start_idx = 0  # test_user的下标
        total_user = list(
            set(train_ui_inter[self.user_id_field].values.tolist() + test_ui_inter[self.user_id_field].values.tolist()))
        total_item = list(
            set(train_ui_inter[self.item_id_field].values.tolist() + test_ui_inter[self.item_id_field].values.tolist()))
        self.test_user = list(set(test_ui_inter[self.user_id_field].values.tolist()))

        user_num = len(total_user)
        self.uid2pos_item_len = [0] * (user_num + 1)
        self.uid2swap_col = [[]] * (user_num + 1)
        self.uid2swap_col_rev = [None] * (user_num + 1)
        self.uid2his = [[]] * (user_num + 1)

        for u in total_user:
            pos_item = test_ui_inter[(test_ui_inter[self.user_id_field] == u) & (test_ui_inter[self.label_field] > 0)][
                self.item_id_field].drop_duplicates()
            self.uid2his[u] = list(
                train_ui_inter[train_ui_inter[self.user_id_field] == u][self.item_id_field].drop_duplicates())
            self.uid2swap_col[u] = sorted(set(range(1, len(pos_item) + 1)) ^ set(pos_item))
            self.uid2swap_col_rev[u] = self.uid2swap_col[u][::-1]
            self.uid2pos_item_len[u] = len(pos_item)

        self.user_num = user_num
        self.item_num = len(total_item)
        self.topk = config['topk']

    # test_user[start:end]
    def get_score_matrix(self, start, end):
        his_row, swap_row, his_col = [], [], []
        swap_col, swap_col_rev = [], []
        batched_test_user = self.test_user[start: end]
        pos_len_matrix = []
        for idx, uid in enumerate(batched_test_user):
            his_row += [idx + 1] * len(self.uid2his[uid])
            swap_row += [idx + 1] * len(self.uid2swap_col[uid])
            his_col += self.uid2his[uid]
            swap_col += self.uid2swap_col[uid]
            swap_col_rev += self.uid2swap_col_rev[uid]
            pos_len_matrix.append(self.uid2pos_item_len[uid])

        his_row_col = (his_row, his_col)
        item_idx = [_ for _ in range(1, self.item_num + 1)]

        item_input, user_input = [], []
        for uid in batched_test_user:
            user_input += [uid] * self.item_num
            item_input += item_idx

        predictions = self._model.predict(
            [np.array(user_input), np.array(item_input)],
            batch_size=self.batch_size * self.item_num,
            verbose=0
        )
        predictions = predictions.reshape((-1, self.item_num))
        # 下标从1开始
        predictions = np.insert(predictions, 0, [0] * predictions.shape[1], axis=0)
        predictions = np.insert(predictions, 0, [0] * predictions.shape[0], axis=1)
        predictions[his_row_col] = -np.inf
        predictions[swap_row, swap_col] = predictions[swap_row, swap_col_rev]
        predictions = predictions[1:, ]
        # print(predictions[:, 1:])
        topk_idx = partition_arg_topK(matrix=predictions, K=self.topk, axis=1)
        pos_len_matrix = np.array(pos_len_matrix).reshape(-1, 1)
        return np.array(topk_idx <= pos_len_matrix).astype('int32'), pos_len_matrix

    def __iter__(self):
        return self

    def __next__(self):
        test_user_num = len(self.test_user)
        if self.start_idx == test_user_num:
            raise StopIteration
        else:
            end_idx = min(test_user_num, self.start_idx + self.batch_size)
            tmp_idx = self.start_idx
            self.start_idx = end_idx
            # [tmp_idx , end_idx)
            return self.get_score_matrix(tmp_idx, end_idx)
