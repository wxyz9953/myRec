from myrec.models.abstract_model import AbstractModel
from keras.layers import *
from keras.regularizers import *
from keras.initializers import *
from keras.models import Model
from myrec.layers import *
import tensorflow as tf
from keras.optimizers import *
import numpy as np


class MyFM(AbstractModel):
    def __init__(self, config, dataset):
        super(MyFM, self).__init__(config, dataset)
        self.field_table_dict = self.dataset.field_table_dict
        self.cate2maxLen = self.dataset.cate2maxLen
        self.cate_fields = config['cate_fields']
        self.multi_hot_fields = config['multi_hot_fields']
        self.field_table_dict = self.dataset.field_table_dict
        self.user_fields = config['user_fields']
        self.item_fields = config['item_fields']

    def build_input_layer(self):
        user_id = Input(shape=(1,), dtype="int32", name="user_input")
        item_id = Input(shape=(1,), dtype="int32", name="item_input")
        return user_id, item_id

    def get_model(self):
        uid, iid = self.build_input_layer()
        emb_size, seed, emb_l2_reg = self.config['embedding_size'], self.config['seed'], self.config['emb_l2_reg']

        emb_layer = dict()
        linear_layer = dict()
        for f in self.cate_fields:
            v = self.cate2maxLen[f][-1]
            emb_layer[f] = Embedding(
                input_dim=v,
                output_dim=emb_size,
                embeddings_regularizer=l2(emb_l2_reg),
                embeddings_initializer=uniform(seed=seed),
                name="%s_embedding" % f
            )
            linear_layer[f] = Embedding(
                input_dim=v,
                output_dim=1,
                embeddings_initializer=uniform(seed=self.config["seed"]),
            )
        for f in self.multi_hot_fields:
            emb_layer[f] = Dense(
                self.config['embedding_size'],
                activation="linear",
                name=f + '_multi_embedding'
            )
            linear_layer[f] = MultiHotLinearInter(
                embeddings_initializer=uniform(seed=seed)
            )

        user_emb_list = []
        item_emb_list = []
        linear_list = []
        for f in self.multi_hot_fields + self.cate_fields:
            matrix = self.field_table_dict[f]
            idx = iid if f in self.item_fields else uid
            field_val = Lambda(lambda x: tf.gather(matrix, tf.to_int32(x)), name='get_%s_one_hot' % f)(idx)
            if len(field_val.shape) > 2:
                field_val = Flatten()(field_val)
            if f in self.multi_hot_fields:
                field_val = Lambda(lambda x: tf.cast(x, tf.float32))(field_val)

            tmp = emb_layer[f](field_val)
            emb = Lambda(lambda x: tf.expand_dims(x, axis=1))(tmp) \
                if f in self.multi_hot_fields else tmp
            if f in self.user_fields:
                user_emb_list.append(emb)
            else:
                item_emb_list.append(emb)
            tmp = linear_layer[f](field_val)
            linear = Lambda(lambda x: tf.expand_dims(x, axis=1))(tmp) \
                if f in self.multi_hot_fields else tmp
            linear_list.append(linear)

        user_emb = Concatenate(axis=1)(user_emb_list) if len(user_emb_list) > 1 else user_emb_list[0]  # (B , f , d)
        item_emb = Concatenate(axis=1)(item_emb_list) if len(item_emb_list) > 1 else item_emb_list[0]
        linear = Concatenate(axis=1)(linear_list) if len(linear_list) > 1 else linear_list[0]  # (B ,f, 1)
        fm = MyFMLayer()([user_emb, item_emb])  # (B, 1)
        linear = Lambda(lambda x: tf.reduce_sum(x, axis=1))(linear)  # (B , 1)
        output = Activation(activation="sigmoid")(Add()([fm, linear]))
        model = Model(
            [uid, iid], output
        )
        model.compile(Adam(), loss="binary_crossentropy")
        return model
