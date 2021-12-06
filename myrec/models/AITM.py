from myrec.models.abstract_model import AbstractModel
from myrec.layers import *
from keras.layers import *
from keras.initializers import *
from keras.regularizers import *
from keras.models import Model
from keras.optimizers import *
import tensorflow as tf


class AITM(AbstractModel):
    def __init__(self, config, dataset):
        super(AITM, self).__init__(config, dataset)

    def build_input_layer(self):
        feat_num = len(self.dataset.train_data.columns)
        return Input(
            shape=(feat_num,),
            dtype="int32",
            name="input_layer"
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

        tower_outputs = []
        for _ in range(self.config['num_tasks']):
            tower_output = DNN(
                units=self.config['tower_units'],
                activations=self.config['tower_activations'],
                use_bias=True,
                seed=self.config['seed'],
                l2_reg=self.config['tower_l2_reg'],
                dropout_rates=self.config['tower_dropout_rates']
            )(cate_emb)
            tower_outputs.append(tower_output)

        p = Dense(units=self.config['units'], activation="relu")(tower_outputs[0])
        q = tower_outputs[1]

        click = Dense(units=1, activation="sigmoid")(tower_outputs[0])
        purchase = AIT(
            h_initializer=glorot_uniform(self.config['seed'])
        )([p, q])
        purchase = Dense(units=1, activation="sigmoid")(purchase)

        model = Model(
            inputs=input_layer,
            outputs=[click, purchase]
        )
        model.compile(
            optimizer=Adam(),
            loss=["binary_crossentropy", "binary_crossentropy"]
        )
        return model
