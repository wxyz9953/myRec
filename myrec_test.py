from myrec.config import FileConfig
import os
from myrec.utils import get_model
import pickle
import numpy as np
from myrec.evaluator import TopKEvaluator
from keras.optimizers import *
from keras.models import load_model, save_model
import heapq
from myrec.layers import *
from myrec.utils.tools import init_environment
import pandas as pd

from myrec.dataset import Movielens1mSampleDataset, Movielens1mNegSampleDataset, Movielens1mBaseDataset
import random


def get_metrics(ranklist, testlist):
    hr = 0
    mrr = 0
    recall = len(set(ranklist) & set(testlist)) / len(testlist)
    prec = len(set(ranklist) & set(testlist)) / len(ranklist)
    for idx, item in enumerate(ranklist):
        if item in testlist:
            hr += 1
            mrr = 1 / (idx + 1)
            break
    return hr, mrr, recall, prec


def neg_evaluate(train, test, _model, topk, item_id, multi_idx=-1):
    total_item = list(set(train[item_id].values.tolist() + test[item_id].values.tolist()))

    matrix = np.array([
        [0.1, 0.8, 0.2, 0.4, 0.5],
        [0.1, 0.3, 0.2, 0.5, 0.7],
        [0.6, 0.4, 0.3, 0.2, 0.1],
        [0.3, 0.1, 0.2, 0.5, 0.7],
        [0.4, 0.7, 0.6, 0.1, 0.3],
        [0.4, 0.8, 0.3, 0.6, 0.7]
    ])
    test_user = np.unique(test['uid'])
    hrs = []
    mrrs = []
    precs = []
    recalls = []
    topk_res = []
    for user in test_user:
        his_item = list(train[train['uid'] == user][item_id].drop_duplicates())
        if len(his_item) == 0:  # 没有训练数据，跳过
            continue

        pos_item = list(test[(test['uid'] == user)][item_id].drop_duplicates())  # 去重保留第一个
        if len(pos_item) == 0:
            continue

        unvisited_item = list(set(total_item) - set(his_item))

        assert len(unvisited_item) == len(set(unvisited_item))

        batch_size = len(unvisited_item)
        users = np.full(len(unvisited_item), user, dtype='int32')
        unvisited_item = np.array(unvisited_item)
        predictions = _model.predict([users, unvisited_item],
                                     batch_size=batch_size, verbose=0)

        predictions = predictions.flatten()
        # predictions = matrix[user - 1][unvisited_item - 1]

        assert len(predictions) == len(unvisited_item)

        if len(predictions) == 2:
            predictions = predictions[multi_idx]

        item2score = {}
        for i in range(len(unvisited_item)):
            item = unvisited_item[i]
            item2score[item] = predictions[i]

        # Evaluate top rank list
        # print(len(testlist))
        rank_list = heapq.nlargest(topk, item2score, key=item2score.get)
        hr, mrr, recall, prec = get_metrics(rank_list, pos_item)
        hrs.append(hr)
        mrrs.append(mrr)
        precs.append(prec)
        recalls.append(recall)
    return (hrs, mrrs, precs, recalls)


if __name__ == '__main__':
    config_file = "config/exp_setting.json"
    config = FileConfig(config_file)
    config['model'] = "FM"

    init_environment(config['seed'])
    dataset = Movielens1mNegSampleDataset(config)

    model = get_model(config, dataset)
    # model.compile(optimizer=Adam(), loss='binary_crossentropy')
    train_data, test_data = dataset.train_sampled_data, dataset.test_sampled_data

    train_instance, _ = dataset.build_instance()

    inputs, labels = train_instance
    uid, iid = inputs
    uid, iid, labels = uid[0:100], iid[0:100], labels[0:100]
    evaluator = TopKEvaluator(config, train_data, test_data)
    model.fit([uid, iid], labels, batch_size=config['batch_size'], epochs=1, verbose=0, shuffle=True)
    evaluator.evaluate(model)
    model.summary()
    model.save_weights("tmp.h5")
    model.load_weights("tmp.h5")
    save_model(model, "t.h5")
    load_model("t.h5", custom_objects={"MultiHotLinearInter": MultiHotLinearInter, "tf": tf, "SENETLayer": SENETLayer,
                                       "BilinearInteractionLayer": BilinearInteractionLayer, "MyFMLayer": MyFMLayer,
                                       "FMLayer": FMLayer})
    os.remove("tmp.h5")
    os.remove("t.h5")
    exit(0)
    for e in range(config['epochs']):
        model.fit(inputs, labels, batch_size=config['batch_size'], epochs=1, verbose=0, shuffle=True)
        # res_dict = evaluator.evaluate(model)
        # for metric in config['metrics']:
        #     print(metric, np.array(res_dict[metric]).mean())
        # print("==================")
        hrs, mrrs, precs, recalls = neg_evaluate(train_data, test_data, model,
                                                 topk=config['topk'], item_id=config['item_id_field'])

        print("=============neg_eval=================")
        print("hr", np.array(hrs).mean())
        print("mrr", np.array(mrrs).mean())
        print("recall", np.array(recalls).mean())
        print("precision", np.array(precs).mean())

    exit(0)
