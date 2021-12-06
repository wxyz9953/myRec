import scipy.sparse as sp
from myrec.config import Config, FileConfig
from myrec.dataset.abstract_dataset import AbstractDataset
from myrec.utils.dataset import get_reindex_map
import pandas as pd
from operator import itemgetter
from sklearn.preprocessing import MinMaxScaler
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split


class Movielens1mBaseDataset(AbstractDataset):
    def __init__(self, config):
        # self.rating_fields = ['uid', 'iid', 'rating' , 'ts']
        # self.user_fields = ["uid", "gender", "age", "oc", "code"]
        # self.item_fields = ["iid", "title", "genre"]
        self.rating_fields = config['rating_fields']
        self.user_fields = config['user_fields']
        self.item_fields = config['item_fields']
        self.dataset_name = config['dataset_name']
        self.dataset_path = config['dataset_path']
        self.user_id_field = config['user_id_field']
        self.item_id_field = config['item_id_field']
        self.label_field = config['label_field']
        self.__load_df()

    def __load_df(self):
        self.ratings = pd.read_table(self.dataset_path + "ratings.dat",
                                     sep="::",
                                     header=None,
                                     names=self.rating_fields,
                                     engine='python')
        self.items = pd.read_table(self.dataset_path + "movies.dat",
                                   sep="::",
                                   header=None,
                                   names=self.item_fields,
                                   engine='python')
        self.users = pd.read_table(self.dataset_path + "users.dat",
                                   sep="::",
                                   header=None,
                                   names=self.user_fields,
                                   engine="python")

    # 构建整个cate_fields的field2token2idx字典
    def build_field2token2idx(self, cate_fields):
        field2token2idx = {}
        for f in cate_fields:
            df = self.users if f in self.user_fields else self.items
            field2token2idx[f] = get_reindex_map(df.values.tolist(), lower_bound=0)
        # user , item编码从1开始
        if self.user_id_field in cate_fields:
            field2token2idx[self.user_id_field] = get_reindex_map(self.users[self.user_id_field].values.tolist(),
                                                                  lower_bound=1)
        if self.item_id_field in cate_fields:
            field2token2idx[self.item_id_field] = get_reindex_map(self.items[self.item_id_field].values.tolist(),
                                                                  lower_bound=1)
        return field2token2idx

    # 使用df构建cate_fields中field对应的field2token2idx字典
    def build_field2token2idx_df(self, df, cate_fields):
        df_field2token2idx = {}
        for field in cate_fields:
            if field not in cate_fields:
                raise TypeError("columns:%s不存在", field)
            lower_bound = 1 if field == self.user_id_field or field == self.item_id_field else 0
            df_field2token2idx[field] = get_reindex_map(df[field].values.tolist(), lower_bound=lower_bound)
        return df_field2token2idx

    def build_cate2maxLen(self, field2token2idx):
        # cate2maxLen for cate embedding
        cate2maxLen = {}
        for idx, field in enumerate(field2token2idx.keys()):
            cate2maxLen[field] = (id, max(field2token2idx[field].values()) + 1)
        return cate2maxLen

    def merge(self, users, items, ratings):
        # merge user item rating
        data = ratings.merge(items, on=self.item_id_field, how='inner')
        data = data.merge(users, on=self.user_id_field, how='inner')
        return data


# class Movielens1mNoSampleDataset(Movielens1mBaseDataset):
#     def __init__(self, config):
#         super().__init__(config)
#         # cate_feat = ['title', 'gender', 'age', 'oc', 'code']
#         self.cate_fields = config['cate_fields']
#         self.dense_fields = config['dense_fields']
#         self.multi_hot_fields = config['multi_hot_fields']
#         # [
#         #   "title" : ["Harry Potter" : 0 , "ABC" : 1],
#         #   "gender" : ["male" : 0 , "female" : 1]
#         # ]
#         self.field2token2idx = dict()
#         # [
#         #   "genre" => ["thrill" , "a" , "b" , "c"],
#         #   "t" => ["k" , "f" , "l"]
#         # ]
#         self.multiHotField2field = dict()
#         self.cate2maxLen = dict()
#         self.field2token2idx = {}
#         self.__preprocess()
#
#     def __reindex_cate_fields(self):
#         # map
#         for field in self.cate_fields:
#             self.data[field] = self.data[field].map(lambda x: self.field2token2idx[field][x])
#         self.data[self.user_id_field] = self.data[self.user_id_field].map(
#             lambda x: self.field2token2idx[self.user_id_field][x]
#         )
#         self.data[self.item_id_field] = self.data[self.item_id_field].map(
#             lambda x: self.field2token2idx[self.item_id_field][x]
#         )
#
#     def __multi_hot_encode(self):
#         # multi_hot
#         for field in self.multi_hot_fields:
#             multi_hot_df = self.data[field].str.get_dummies("|")
#             self.multiHotField2field[field] = multi_hot_df.columns
#             self.data = self.data.join(multi_hot_df)
#             assert type(self.multiHotField2field[field]) is list
#         self.data = self.data.drop(columns=[self.multi_hot_fields])
#
#     def __preprocess(self):
#         self.data = self.merge(self.users, self.items, self.items)
#         self.field2token2idx = self.build_field2token2idx(self.cate_fields)
#         self.__reindex_cate_fields()
#         self.cate2maxLen = self.build_cate2maxLen(self.field2token2idx)
#         self.__multi_hot_encode()
#         # dense
#         mms = MinMaxScaler(feature_range=(0, 1))
#         self.data[self.dense_fields] = mms.fit_transform(self.data[self.dense_fields])
#
#         # label
#         self.data[self.label_field] = self.data[self.label_field].map(lambda x: type(x > 3.0))
#
#     def __build_input_dict(self, is_user):
#         fields = self.user_fields if is_user else self.item_fields
#         cate_list = list(set(fields) & set(self.cate_fields))
#         dense_list = list(set(fields) & set(self.dense_fields))
#         multihot_fields = set(fields) & set(self.multi_hot_fields)
#         multihot_dict = {k: v for k, v in self.multiHotField2field if k in multihot_fields}
#         return cate_list, dense_list, multihot_dict
#
#     def __build_input(self, field_list):
#         return self.data.loc[:, field_list]
#
#     def build_instance(self):
#         user_cate_list, user_dense_list, user_multihot_dict = self.__build_input_dict(is_user=True)
#         item_cate_list, item_dense_list, item_multihot_dict = self.__build_input_dict(is_user=False)
#         model_inputs = [self.__build_input(user_cate_list), self.__build_input(user_dense_list)]
#         for k, v in user_multihot_dict.items():
#             model_inputs.append(self.__build_input(v))
#         model_inputs += [self.__build_input(item_cate_list), self.__build_input(item_dense_list)]
#         for k, v in item_multihot_dict.items():
#             model_inputs.append(self.__build_input(v))
#
#         return model_inputs, self.__build_input(self.label_field)


class Movielens1mSampleDataset(Movielens1mBaseDataset):
    def __init__(self, config):
        self.sample_user_num = config['sample_user_num']
        # cate_feat = ['uid' , 'iid' , 'title', 'gender', 'age', 'oc', 'code']
        self.cate_fields = config['cate_fields']
        self.multi_hot_fields = config['multi_hot_fields']
        self.remove_fields = config['remove_fields']
        self.seed = config['seed']
        super().__init__(config)
        self.sampled_users = self.__build_sampled_users()
        sampled_rating = self.ratings[self.ratings[self.user_id_field].isin(self.sampled_users)]
        self.sampled_data = self.merge(self.users, self.items, sampled_rating)
        self.field2token2idx = self.build_field2token2idx_df(self.sampled_data,
                                                             self.cate_fields)
        self.cate2maxLen = self.build_cate2maxLen(self.field2token2idx)
        self.__reindex_cate_fields(self.cate_fields)
        # 提取user-item interaction
        # self.ui_inter = self.sampled_data.loc[:, [self.user_id_field, self.item_id_field]]
        self.sampled_items = list(set(self.sampled_data.loc[:, self.item_id_field].values.tolist()))
        self.sample_item_num = len(self.sampled_items)
        self.sampled_users = list(set(self.sampled_data.loc[:, self.user_id_field].values))
        self.sample_user_num = len(self.sampled_users)

        # multi-hot
        # 构建 {"title" : matrix 1 * n, genre : matrix}
        self.field_table_dict = dict()
        self.user_cate_fields = list(set(self.cate_fields) & set(self.user_fields))
        self.item_cate_fields = list(set(self.cate_fields) & set(self.item_fields))
        for f in self.user_cate_fields:
            self.field_table_dict[f] = [-1 for _ in range(self.sample_user_num + 1)]
            for u in self.sampled_users:
                self.field_table_dict[f][u] = \
                    self.sampled_data[self.sampled_data[self.user_id_field] == u].loc[:, f].values.tolist()[0]

        for f in self.item_cate_fields:
            self.field_table_dict[f] = [-1 for _ in range(self.sample_item_num + 1)]
            for i in self.sampled_items:
                self.field_table_dict[f][i] = \
                    self.sampled_data[self.sampled_data[self.item_id_field] == i].loc[:, f].values.tolist()[0]

        field2multiHotField = {}
        for f in self.multi_hot_fields:
            df = self.sampled_data[f].str.get_dummies("|")
            field2multiHotField[f] = df.columns
            self.sampled_data = self.sampled_data.join(df)
            self.sampled_data = self.sampled_data.drop(columns=[f])

        for f in self.multi_hot_fields:
            self.field_table_dict[f] = [-1 for _ in range(self.sample_item_num + 1)]
            for i in self.sampled_items:
                self.field_table_dict[f][i] = \
                    self.sampled_data[self.sampled_data[self.item_id_field] == i].loc[:,
                    field2multiHotField[f]].values.tolist()[0]
            self.field_table_dict[f][0] = self.field_table_dict[f][1]

        self.train_sampled_data, self.test_sampled_data = train_test_split(self.sampled_data, test_size=0.2,
                                                                           random_state=self.seed)
        self.ui_matrix = None
        if config['use_ui_interaction']:
            self.ui_matrix = self.build_ui_matrix(self.train_sampled_data)

    def build_ui_matrix(self, ui_inter):
        mat = np.zeros((self.sample_user_num + 1, self.sample_item_num + 1), dtype=np.float32)
        for idx, row in ui_inter.iterrows():
            u, i = row[self.user_id_field], row[self.item_id_field]
            mat[u, i] = 1
        return mat

    def __reindex_cate_fields(self, cate_fields):
        for f in cate_fields:
            self.sampled_data[f] = self.sampled_data[f].map(lambda x: self.field2token2idx[f][x])

    def __build_sampled_users(self):
        # 先采样，再合并，再reindex
        total_users = list(set(self.ratings[self.user_id_field].values.tolist()))
        random.seed(self.seed)
        return random.sample(total_users, self.sample_user_num)

    def build_instance(self):
        train_instances = [self.train_sampled_data.loc[:, [self.user_id_field, self.item_id_field]],
                           self.train_sampled_data.loc[:, self.label_field]]
        test_instances = [self.test_sampled_data.loc[:, [self.user_id_field, self.item_id_field]],
                          self.test_sampled_data.loc[:, self.label_field]]
        return train_instances, test_instances


class Movielens1mNegSampleDataset(Movielens1mSampleDataset):
    def __init__(self, config):
        super().__init__(config)
        self.num_neg = config['num_neg']
        np.random.seed(self.seed)

    def build_instance(self):
        user_input, item_input, labels = [], [], []
        ui_matrix = self.build_ui_matrix(self.train_sampled_data) if self.ui_matrix is None else self.ui_matrix
        sparse_matrix = sp.dok_matrix(ui_matrix)
        for (u, i) in sparse_matrix.keys():
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            for _ in range(self.num_neg):
                j = np.random.randint(self.sample_item_num + 1)
                while (u, j) in sparse_matrix.keys():
                    j = np.random.randint(self.sample_item_num + 1)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)

        train_instances = [user_input, item_input], labels
        test_instances = [self.test_sampled_data.loc[:, [self.user_id_field, self.item_id_field]],
                          self.test_sampled_data.loc[:, self.label_field]]
        return train_instances, test_instances
