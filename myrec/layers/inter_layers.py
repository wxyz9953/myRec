from keras import activations, initializers, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
from keras.layers import *
from keras.initializers import *
import sys


class biasSVDLayer(Layer):
    def __init__(self,
                 user_bias_ini="zeros",
                 item_bias_ini="zeros",
                 bias_init="zeros",
                 user_bias_reg="l2",
                 item_bias_reg="l2",
                 bias_reg="l2",
                 **kwargs):
        self.user_bias_ini = initializers.get(user_bias_ini)
        self.item_bias_ini = initializers.get(item_bias_ini)
        self.bias_ini = initializers.get(bias_init)
        self.user_bias_reg = regularizers.get(user_bias_reg)
        self.item_bias_reg = regularizers.get(item_bias_reg)
        self.bias_reg = regularizers.get(bias_reg)
        self.user_bias = None
        self.item_bias = None
        self.bias = None

        super(biasSVDLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)

        self.user_bias = self.add_weight(
            name="user_bias",
            shape=(1,),
            initializer=self.user_bias_ini,
            regularizer=self.user_bias_reg
        )

        self.item_bias = self.add_weight(
            name="item_bias",
            shape=(1,),
            initializer=self.item_bias_ini,
            regularizer=self.item_bias_reg
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(1,),
            initializer=self.bias_ini,
            regularizer=self.bias_reg
        )

        super(biasSVDLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        user_emb, item_emb = inputs
        res = Dot(axes=1)([user_emb, item_emb]) + self.bias + self.item_bias + self.user_bias
        assert res.shape[-1] == 1
        return res

    def compute_output_shape(self, input_shape):
        assert input_shape is not None and len(input_shape) >= 2

        input_shape = input_shape[0]

        output_shape = list(input_shape)
        output_shape[-1] = 1
        output_shape = tuple(output_shape)

        return output_shape

    def get_config(self):
        config = {
            "user_bias_ini": initializers.serialize(self.user_bias_ini),
            "item_bias_ini": initializers.serialize(self.item_bias_ini),
            "bias_ini": initializers.serialize(self.bias_ini),
            "user_bias_reg": regularizers.serialize(self.user_bias_reg),
            "item_bias_reg": regularizers.serialize(self.item_bias_reg),
            "bias_reg": regularizers.serialize(self.bias_reg)
        }

        base_config = super(biasSVDLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class FMLayer(Layer):
    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert len(inputs.shape) == 3
        # (n , k , d)
        square_of_sum = K.tf.square(K.tf.reduce_sum(inputs, axis=1, keepdims=False))  # (n , d)
        sum_of_square = K.tf.reduce_sum(inputs * inputs, axis=1, keepdims=False)  # (n , d)
        cross = square_of_sum - sum_of_square
        res = 0.5 * K.tf.reduce_sum(cross, axis=1, keepdims=True)  # (n,1)
        return res

    def compute_output_shape(self, input_shape):
        return (None, 1)


class FwFMLayer(Layer):
    def __init__(self,
                 weights_initializer=None,
                 weights_regularizer=None,
                 weights_constraint=None,
                 **kwargs):
        self.weights_ini = initializers.get(weights_initializer)
        self.weights_reg = regularizers.get(weights_regularizer)
        self.weights_con = constraints.get(weights_constraint)
        super(FwFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        emb_num = int(input_shape[1])
        self.fm_weights = self.add_weight(
            name="weight",
            shape=(emb_num, emb_num),
            initializer=self.weights_ini,
            regularizer=self.weights_reg,
            constraint=self.weights_con
        )
        super(FwFMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        emb_num = int(inputs.shape[1])
        res_list = []
        for i in range(emb_num):
            for j in range(i + 1, emb_num):
                w = self.fm_weights[i, j]
                e1 = inputs[:, i, :]  # (B, embedding_size)
                e2 = inputs[:, j, :]  # (B ,embedding_size)
                assert len(e1.shape) == 2 and e1.shape[-1] == inputs.shape[-1]

                res = K.tf.scalar_mul(w, K.batch_dot(e1, e2, axes=1))  # (B , 1)
                assert res.shape[-1] == 1
                res_list.append(res)
        return K.tf.add_n(res_list)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = {
            "weights_initializer": initializers.serialize(self.weights_ini),
            "weights_regularizer": regularizers.serialize(self.weights_reg),
            "weights_constraint": constraints.serialize(self.weights_con)
        }
        base_config = super(FwFMLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


# (B , f, d) => (B, f , d)
class SENETLayer(Layer):
    def __init__(self,
                 reduction=3,
                 weights_initializer=None,
                 weights_regularizer=None,
                 weights_constraint=None,
                 **kwargs):
        self.reduction_ratio = reduction
        self.weights_initializer = initializers.get(weights_initializer)
        self.weights_regularizer = regularizers.get(weights_regularizer)
        self.weights_constraint = constraints.get(weights_constraint)
        super(SENETLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        f = input_shape[1]
        self.W1 = self.add_weight(
            name="W1",
            shape=(f, f // self.reduction_ratio),
            initializer=self.weights_initializer,
            regularizer=self.weights_regularizer,
            constraint=self.weights_constraint
        )
        self.W2 = self.add_weight(
            name="W2",
            shape=(f // self.reduction_ratio, f),
            initializer=self.weights_initializer,
            regularizer=self.weights_regularizer,
            constraint=self.weights_constraint
        )
        super(SENETLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        mean_val = K.tf.reduce_mean(inputs, axis=2)  # (n,f)
        # 文章中默认relu
        A = K.tf.nn.relu(K.dot(mean_val, self.W1))  # (n , f / r)
        A = K.tf.nn.relu(K.dot(A, self.W2))  # (n, f)
        res = K.tf.multiply(inputs, K.tf.expand_dims(A, -1))  # (n , f , d)
        return res

    def get_config(self):
        config = {
            "reduction": self.reduction_ratio,
            "weights_initializer": initializers.serialize(self.weights_initializer),
            "weights_regularizer": regularizers.serialize(self.weights_regularizer),
            "weights_constraint": constraints.serialize(self.weights_constraint)
        }
        base_config = super(SENETLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


class BilinearInteractionLayer(Layer):
    def __init__(self,
                 type="all",
                 weights_initializer=None,
                 weights_regularizer=None,
                 weights_constraint=None,
                 **kwargs):
        self.type = type
        self.weights_initializer = initializers.get(weights_initializer)
        self.weights_regularizer = regularizers.get(weights_regularizer)
        self.weights_constraint = constraints.get(weights_constraint)
        super(BilinearInteractionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        f, e = int(input_shape[1]), int(input_shape[-1])
        if self.type == "all":
            self.w = self.add_weight(
                name="w",
                shape=(e, e),
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                constraint=self.weights_constraint
            )
        elif self.type == "each":
            self.w = [
                self.add_weight(
                    name="w_" + str(idx),
                    shape=(e, e),
                    initializer=self.weights_initializer,
                    regularizer=self.weights_regularizer,
                    constraint=self.weights_constraint
                ) for idx in range(f - 1)]
        elif self.type == "interaction":
            self.w = [[_ for _ in range(f)] for _ in range(f)]
            for i in range(f):
                for j in range(i + 1, f):
                    self.w[i][j] = self.add_weight(
                        name="w_%d%d" % (i, j),
                        shape=(e, e),
                        initializer=self.weights_initializer,
                        regularizer=self.weights_regularizer,
                        constraint=self.weights_constraint
                    )
        else:
            raise TypeError("No such type")
        super(BilinearInteractionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        f = int(inputs.shape[1])
        ps = []
        for i in range(f):
            for j in range(i + 1, f):
                v1, v2 = inputs[:, i, :], inputs[:, j, :]  # (n , d)
                assert len(v1.shape) == 2 and v1.shape[-1] == int(inputs.shape[-1])
                if self.type == "all":
                    w = self.w
                elif self.type == "each":
                    w = self.w[i]
                elif self.type == "interaction":
                    w = self.w[i][j]
                else:
                    raise TypeError("No such type")
                p = K.tf.multiply(K.dot(v1, w), v2)
                ps.append(p)
        return K.tf.stack(ps, axis=1)

    def compute_output_shape(self, input_shape):
        f, e = int(input_shape[1]), int(input_shape[-1])
        return (None, f * (f - 1) // 2, e)

    def get_config(self):
        config = {
            "type": self.type,
            "weights_initializer": initializers.serialize(self.weights_initializer),
            "weights_regularizer": regularizers.serialize(self.weights_regularizer),
            "weights_constraint": constraints.serialize(self.weights_constraint)
        }
        base_config = super(BilinearInteractionLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


class MyFMLayer(Layer):
    def __init__(self, **kwargs):
        super(MyFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.inter_w = self.add_weight(
            name='inter_w',
            shape=(1, 1),
            initializer=Ones()
        )

        self.inner_w = self.add_weight(
            name='inner_w',
            shape=(2, 1),
            initializer=Constant(value=0.5)
        )
        super(MyFMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        user, item = inputs  # (B, f, d)
        user_emb = K.tf.reduce_sum(user, axis=1, keepdims=False)  # (B , d)
        item_emb = K.tf.reduce_sum(item, axis=1, keepdims=False)  # (B , d)
        inter = self.inter_w * (user_emb * item_emb)  # (B, d)
        inter = K.tf.reduce_sum(inter, axis=1, keepdims=True)  # (B,1)

        user_square_of_sum = K.tf.square(K.tf.reduce_sum(
            user, axis=1, keepdims=False))  # (B, d)
        user_sum_of_square = K.tf.reduce_sum(
            user * user, axis=1, keepdims=False)  # (B ,d)
        user_fm = user_square_of_sum - user_sum_of_square
        user_fm = K.tf.reduce_sum(user_fm, axis=1, keepdims=True)  # (B, 1)

        item_square_of_sum = K.tf.square(K.tf.reduce_sum(
            item, axis=1, keepdims=False))
        item_sum_of_square = K.tf.reduce_sum(
            item * item, axis=1, keepdims=False)
        item_fm = item_square_of_sum - item_sum_of_square
        item_fm = K.tf.reduce_sum(item_fm, axis=1, keepdims=True)  # (B, 1)

        fm_ui = K.tf.concat([user_fm, item_fm], axis=1)  # (B, 2)

        weighted_fm = K.dot(fm_ui, self.inner_w)  # (B, 1)

        return inter + weighted_fm

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        base_config = super(MyFMLayer, self).get_config()
        return base_config
