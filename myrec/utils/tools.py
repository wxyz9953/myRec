import os
import numpy as np
import random
import tensorflow as tf
from keras import backend as K
import pickle


def init_environment(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf_session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(tf_session)


def write_to_pickle(path, obj):
    f = open(path, "wb")
    pickle.dump(obj, f)
    f.close()


def read_from_pickle(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj
