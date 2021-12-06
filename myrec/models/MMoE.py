from myrec.models.abstract_model import AbstractModel
from keras.layers import *
from keras.regularizers import *
from keras.initializers import *
from keras.models import Model
from myrec.layers import *
import tensorflow as tf
from keras.optimizers import *


class MMoE(AbstractModel):
    def __init__(self, config, dataset):
        super(MMoE, self).__init__(config, dataset)

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

        mmoe_outputs = MMoELayer(
            self.config['expert_units'],
            self.config['num_experts'],
            self.config['num_tasks'],
            expert_kernel_initializer=glorot_uniform(self.config['seed']),
            gate_kernel_initializer=glorot_uniform(self.config['seed']),
        )(cate_emb)

        final_outputs = []
        for o in mmoe_outputs:
            tower_output = DNN(
                units=self.config['tower_units'],
                activations=self.config['tower_activations'],
                use_bias=True,
                seed=self.config['seed'],
                l2_reg=self.config['tower_l2_reg'],
                dropout_rates=self.config['tower_dropout_rates']
            )(o)
            final_outputs.append(tower_output)

        model = Model(
            inputs=input_layer,
            outputs=final_outputs
        )
        model.compile(
            optimizer=Adam(),
            loss=["binary_crossentropy", "binary_crossentropy"]
        )
        return model
