from myrec.models.abstract_model import AbstractModel
from keras.layers import *
from keras.regularizers import *
from keras.initializers import *
from keras.models import Model
from myrec.layers import *
import tensorflow as tf
from keras.optimizers import *


class ESMM(AbstractModel):
    def __init__(self, config, dataset):
        super(ESMM, self).__init__(config, dataset)
        assert config['ctr_activations'][-1] == "sigmoid" and config['cvr_activations'][-1] == "sigmoid"
        self.ctr_dnn_layer = DNN(
            units=config['ctr_units'],
            activations=config['ctr_activations'],
            use_bias=True,
            seed=config['seed'],
            l2_reg=config['ctr_l2_reg'],
            dropout_rates=config['ctr_dropout_rates']
        )
        self.cvr_dnn_layer = DNN(
            units=config['cvr_units'],
            activations=config['cvr_activations'],
            use_bias=True,
            seed=config['seed'],
            l2_reg=config['cvr_l2_reg'],
            dropout_rates=config['cvr_dropout_rates']
        )

    def get_model(self):
        input_layer = self.build_input_layer()
        emb_layer = dict()
        cate2maxLen = self.dataset.cate2maxLen
        emb_size = self.config['embedding_size']
        for k, (_, v) in cate2maxLen.items():
            emb_layer[k] = Embedding(
                input_dim=v,
                output_dim=emb_size,
                embeddings_regularizer=l2(self.config["emb_l2_reg"]),
                embeddings_initializer=uniform(seed=self.config["seed"]),
            )
        embeddings = []
        for k, (v, _) in cate2maxLen.items():
            embed = emb_layer[k](
                Lambda(lambda x: tf.reshape(x[:, v], [-1, 1]))(input_layer)
            )
            embed = Reshape((emb_size,))(embed)
            embeddings.append(embed)
        # [(n, k), (n, k) , (n, k)]
        cate_emb = Concatenate()(embeddings)  # (n , f*k)
        assert len(cate_emb.shape) == 2
        ctr = self.ctr_dnn_layer(cate_emb)
        cvr = self.cvr_dnn_layer(cate_emb)
        ctcvr = Multiply()([ctr, cvr])
        model = Model(
            inputs=input_layer,
            outputs=[ctr, ctcvr]
        )
        model.compile(
            optimizer=Adam(),
            loss=["binary_crossentropy", "binary_crossentropy"]
        )
        return model

    def build_input_layer(self):
        feat_num = len(self.dataset.train_data.columns)
        return Input(
            shape=(feat_num,),
            dtype="int32",
            name="input_layer"
        )
