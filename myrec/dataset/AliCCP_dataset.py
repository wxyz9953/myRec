from myrec.dataset import AbstractDataset
import pandas as pd
from myrec.utils.dataset import get_reindex_map
from myrec.config import FileConfig


class AliCCPBuyWeightDataset(AbstractDataset):
    def __init__(self, config):
        self.user_fields = config["user_fields"]
        self.item_fields = config['item_fields']
        self.label_fields = config['label_fields']
        self.data_path = config['dataset_path']
        self.seed = config['seed']
        SAMPLE_NUM = 100000
        train_items = pd.read_csv(
            self.data_path + "sample_skeleton_train.csv",
            header=0,
            engine='python'
        )[self.item_fields]
        train_users = pd.read_csv(
            self.data_path + "common_features_train.csv",
            header=0,
            engine="python",
        )[self.user_fields]
        test_items = pd.read_csv(
            self.data_path + "sample_skeleton_test.csv",
            header=0,
            engine="python"
        )[self.item_fields]
        test_users = pd.read_csv(
            self.data_path + "common_features_test.csv",
            header=0,
            engine="python"
        )[self.user_fields]

        self.train_data = train_items.merge(train_users, on="md5", how="inner"). \
            sample(SAMPLE_NUM, random_state=self.seed)
        self.train_label = self.train_data[self.label_fields]
        self.train_data.drop(columns=["md5"] + self.label_fields, inplace=True)

        self.test_data = test_items.merge(test_users, on="md5", how="inner"). \
            sample(SAMPLE_NUM, random_state=self.seed)
        self.test_label = self.test_data[self.label_fields]
        self.test_data.drop(columns=["md5"] + self.label_fields, inplace=True)

        self.cate2maxLen = {}
        for idx, f in enumerate(self.train_data.columns):
            remap = get_reindex_map(self.train_data[f].values.tolist() + self.test_data[f].values.tolist(),
                                    lower_bound=1)
            self.train_data[f] = self.train_data[f].map(lambda x: remap[x])
            self.test_data[f] = self.test_data[f].map(lambda x: remap[x])
            self.cate2maxLen[f] = (idx, max(remap.values()) + 1)

    def build_instance(self):
        # (label , data)
        return self.train_data, self.train_label, self.test_data, self.test_label
