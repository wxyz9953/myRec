from keras.layers import *
from keras.initializers import *
from keras.regularizers import *


class DNN(Layer):
    def __init__(self, units, activations, use_bias, seed, l2_reg, dropout_rates=None, use_bn=False, **kwargs):
        if dropout_rates is None:
            dropout_rates = []
        assert len(units) == len(activations)
        self.units = units
        self.activations = activations
        self.use_bias = use_bias
        self.seed = seed
        self.l2_reg = l2_reg
        self.dropout_rates = dropout_rates
        self.use_bn = use_bn

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_layers = []
        for unit, ac in zip(self.units, self.activations):
            self.dense_layers.append(
                Dense(
                    units=unit,
                    activation=ac,
                    use_bias=self.use_bias,
                    kernel_initializer=glorot_uniform(self.seed),
                    kernel_regularizer=l2(self.l2_reg)
                )
            )

        if self.use_bn:
            self.bn_layers = [BatchNormalization() for _ in range(len(self.units))]

        self.dropout_layers = [Dropout(self.dropout_rates[i], seed=self.seed + i) for i in
                               range(len(self.units))]

        super(DNN, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        x = inputs

        for i in range(len(self.units)):
            fc = self.dense_layers[i](x)
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            if len(self.dropout_rates) > 0:
                fc = self.dropout_layers[i](fc, training=training)
            x = fc
        return x

    def compute_output_shape(self, input_shape):
        if len(self.units) > 0:
            shape = input_shape[:-1] + (self.units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activations': self.activations,
            'l2_reg': self.l2_reg,
            'use_bn': self.use_bn,
            "use_bias": self.use_bias,
            'dropout_rates': self.dropout_rates,
            'seed': self.seed
        }
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GatingNetwork(Layer):
    def __init__(self,
                 expert_num,
                 gate_kernel_initializer=None,
                 gate_kernel_regularizer=None,
                 gate_kernel_constraint=None,
                 **kwargs):
        self.expert_num = expert_num
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)
        super(GatingNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dimension = int(input_shape[-1])
        self.gate_kernel = self.add_weight(
            name='gate_kernel',
            shape=(input_dimension, self.expert_num),
            initializer=self.gate_kernel_initializer,
            regularizer=self.gate_kernel_regularizer,
            constraint=self.gate_kernel_constraint
        )

        super(GatingNetwork, self).build(input_shape)

    def call(self, inputs, **kwargs):
        gate_output = K.dot(inputs, self.gate_kernel)
        assert gate_output.shape[-1] == self.gate_kernel.shape[-1]
        gate_output = K.softmax(gate_output, axis=1)
        assert gate_output.shape[-1] == self.gate_kernel.shape[-1]
        # (n * M)
        return gate_output

    def compute_output_shape(self, input_shape):
        return (None, self.expert_num)

    def get_config(self):
        config = {
            'expert_num': self.expert_num,
            'gate_kernel_initializer': initializers.serialize(self.gate_kernel_initializer),
            'gate_kernel_regularizer': regularizers.serialize(self.gate_kernel_regularizer),
            'gate_kernel_constraint': constraints.serialize(self.gate_kernel_constraint),
        }
        base_config = super(GatingNetwork, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


# FM线性交互部分，用于处理multi-hot
# multi-hot是(B, l)
class MultiHotLinearInter(Layer):
    def __init__(self,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        self.emb_ini = initializers.get(embeddings_initializer)
        self.emb_reg = regularizers.get(embeddings_regularizer)
        self.emb_con = constraints.get(embeddings_constraint)
        super(MultiHotLinearInter, self).__init__(**kwargs)

    def build(self, input_shape):
        self.multi_hot_emb = self.add_weight(
            name="weights",
            shape=(input_shape[-1],),
            initializer=self.emb_ini,
            regularizer=self.emb_reg,
            constraint=self.emb_con
        )
        super(MultiHotLinearInter, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # (B, l)
        res = K.tf.multiply(inputs, self.multi_hot_emb)
        res = K.tf.reduce_sum(res, axis=1 , keepdims=True)
        return res

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = {
            "embeddings_initializer": initializers.serialize(self.emb_ini),
            "embeddings_regularizer": regularizers.serialize(self.emb_reg),
            "embeddings_constraint": constraints.serialize(self.emb_con)
        }
        base = super(MultiHotLinearInter, self).get_config()
        return dict(list(config.items()) + list(base.items()))
