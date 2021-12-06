from myrec.models.abstract_model import AbstractModel
from myrec.layers.inter_layers import biasSVDLayer
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import *
import tensorflow as tf


class TestMF(Model):
    def call(self, inputs, mask=None):
        print(inputs)


class MF(AbstractModel):
    def __init__(self, config, dataset):
        super(MF, self).__init__(config, dataset)

    def build_input_layer(self):
        # (B, 1)
        user_id = Input(shape=(1,), dtype="int32", name="user_input")
        item_id = Input(shape=(1,), dtype="int32", name="item_input")
        return user_id, item_id

    def get_model(self):
        user_input, item_input = self.build_input_layer()

        emb_size = self.config['embedding_size']

        user_emb = Embedding(
            input_dim=self.dataset.cate2maxLen[self.config['user_id_field']][-1],
            output_dim=self.config['embedding_size'],
            embeddings_regularizer=l2(self.config["emb_l2_reg"]),
            embeddings_initializer=uniform(seed=self.config["seed"]),
            name="user_embedding"
        )(user_input)

        user_emb = Reshape((emb_size,))(user_emb)

        item_emb = Embedding(
            input_dim=self.dataset.cate2maxLen[self.config['item_id_field']][-1],
            output_dim=self.config['embedding_size'],
            embeddings_regularizer=l2(self.config["emb_l2_reg"]),
            embeddings_initializer=uniform(seed=self.config["seed"]),
            name="item_embedding"
        )(item_input)

        item_emb = Reshape((emb_size,))(item_emb)

        prediction = Activation(activation='sigmoid')(biasSVDLayer()([user_emb, item_emb]))
        model = Model(inputs=[user_input, item_input],
                      outputs=prediction)
        return model


if __name__ == '__main__':
    model = TestMF()

    model.fit(dict)
