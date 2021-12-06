from myrec.config import FileConfig
from myrec.dataset import AliCCPBuyWeightDataset
from myrec.utils.tools import write_to_pickle, read_from_pickle
from myrec.utils.model.utils import get_model
from keras.models import load_model, save_model
from myrec.layers import *
import tensorflow as tf
import os

if __name__ == '__main__':
    config = FileConfig("config/mtl_setting.json")
    config['model'] = "AITM"
    model_config = FileConfig("config/%s.json" % (config['model']))
    config.update(model_config)

    dataset = read_from_pickle("aliccp.pkl")
    train_data, train_label, test_data, test_label = dataset.build_instance()
    train_data, train_label, test_data, test_label = \
        train_data[0:100], train_label[0:100], test_data[0:100], test_label[0:100]
    clc, buy = config['label_fields']
    model = get_model(config, dataset)
    model.fit(train_data, [train_label[clc], train_label[buy]],
              batch_size=config['batch_size'], epochs=1,
              verbose=0, shuffle=True)
    model.save_weights(config['model'] + "_weights.h5")
    model.load_weights(config['model'] + "_weights.h5")
    # save_model(model, "tmp.h5")
    # load_model("tmp.h5", custom_objects={"DNN": DNN, "tf": tf})
    os.remove(config['model'] + "_weights.h5")
    # os.remove("tmp.h5")
    model.summary()
