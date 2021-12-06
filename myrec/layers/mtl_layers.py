from keras.layers import *
from myrec.layers import *
from keras.initializers import *


class AIT(Layer):
    def __init__(self,
                 h_initializer=None,
                 h_regularizer=None,
                 h_constraint=None,
                 **kwargs):
        self.h_initializer = initializers.get(h_initializer)
        self.h_regularizer = regularizers.get(h_regularizer)
        self.h_constraint = constraints.get(h_constraint)
        super(AIT, self).__init__(**kwargs)

    def build(self, input_shape):
        # [(B, K) , (B, K)]
        assert isinstance(input_shape, list)
        h_shape = int(input_shape[0][-1])
        self.h = [self.add_weight(
            name="h" + str(i),
            shape=(h_shape, h_shape),
            initializer=self.h_initializer,
            regularizer=self.h_regularizer,
            constraint=self.h_constraint
        ) for i in range(len(input_shape) + 1)]
        super(AIT, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        u = K.tf.stack(inputs, axis=1)  # (B , 2 , K)
        assert u.shape[1] == 2 and u.shape[-1] == inputs[0].shape[-1]
        h2u = K.dot(u, self.h[1])  # (B, 2, K)
        h3u = K.dot(u, self.h[2])  # (B, 2, K)
        wu = K.tf.reduce_sum(K.tf.multiply(h2u, h3u), axis=2) / \
             K.tf.sqrt(K.tf.cast(inputs[0].shape[-1], K.tf.float32))  # (B, 2)
        wu = K.tf.nn.softmax(wu, axis=1)  # (B, 2)
        assert len(wu.shape) == 2 and wu.shape[-1] == 2
        h1u = K.dot(u, self.h[0])  # (B, 2, K)
        wu = K.tf.expand_dims(wu, axis=-1)  # (B, 2, 1)
        z = K.tf.reduce_sum(K.tf.multiply(wu, h1u), axis=1)  # (B, K)
        return z


# Single-Layer
class ExpertLayer(Layer):
    def __init__(self,
                 units,
                 num_experts,
                 use_expert_bias=True,
                 expert_activation='relu',
                 expert_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 expert_bias_constraint=None,
                 expert_kernel_initializer=None,
                 expert_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 **kwargs):

        self.units = units
        self.num_experts = num_experts

        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)

        self.expert_activation = activations.get(expert_activation)

        self.use_expert_bias = use_expert_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)

        self.supports_masking = True

        super(ExpertLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        assert input_shape is not None and len(input_shape) >= 2

        input_dimension = int(input_shape[-1])
        self.expert_kernels = self.add_weight(
            name='expert_kernel',
            shape=(input_dimension, self.num_experts, self.units),
            initializer=self.expert_kernel_initializer,
            regularizer=self.expert_kernel_regularizer,
            constraint=self.expert_kernel_constraint,
        )

        if self.use_expert_bias:
            self.expert_bias = self.add_weight(
                name='expert_bias',
                shape=(self.num_experts, self.units),
                initializer=self.expert_bias_initializer,
                regularizer=self.expert_bias_regularizer,
                constraint=self.expert_bias_constraint,
            )

        super(ExpertLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        expert_outputs = K.tf.tensordot(inputs, self.expert_kernels, axes=1)  # (B, e , k)
        assert expert_outputs.shape[1] == self.num_experts
        if self.use_expert_bias:
            expert_outputs = K.bias_add(x=expert_outputs, bias=self.expert_bias)
        expert_outputs = self.expert_activation(expert_outputs)
        return expert_outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_experts, self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'num_experts': self.num_experts,
            'use_expert_bias': self.use_expert_bias,
            'expert_activation': activations.serialize(self.expert_activation),
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'expert_kernel_initializer': initializers.serialize(self.expert_kernel_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
        }
        base_config = super(ExpertLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class MMoELayer(Layer):
    def __init__(self,
                 units,
                 num_experts,
                 num_tasks,

                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',

                 expert_kernel_initializer=None,
                 expert_kernel_regularizer=None,
                 expert_kernel_constraint=None,

                 gate_kernel_initializer=None,
                 gate_kernel_regularizer=None,
                 gate_kernel_constraint=None,

                 expert_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 expert_bias_constraint=None,

                 gate_bias_initializer='zeros',
                 gate_bias_regularizer=None,
                 gate_bias_constraint=None,
                 **kwargs):
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        self.expert_activation = activations.get(expert_activation)

        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        super(MMoELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape is not None and len(input_shape) >= 2

        super(MMoELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs: (B,e*k)
        gate_outputs = [GatingNetwork(
            expert_num=self.num_experts,
            gate_kernel_initializer=self.gate_kernel_initializer,
            gate_kernel_regularizer=self.gate_kernel_regularizer,
            gate_kernel_constraint=self.gate_kernel_constraint,
            name="gate_" + str(idx)
        )(inputs) for idx in range(self.num_tasks)]
        final_outputs = []
        expert_outputs = ExpertLayer(
            units=self.units,
            num_experts=self.num_experts,
            use_expert_bias=True,
            expert_activation="relu",
            expert_kernel_initializer=self.expert_kernel_initializer,
            expert_kernel_regularizer=self.expert_kernel_regularizer,
            expert_kernel_constraint=self.expert_kernel_constraint,
            expert_bias_initializer=self.expert_bias_initializer,
            expert_bias_regularizer=self.expert_bias_regularizer,
            expert_bias_constraint=self.expert_bias_constraint,
        )(inputs)

        for gate_output in gate_outputs:
            expanded_gate_output = K.expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * K.repeat_elements(expanded_gate_output, self.units, axis=1)
            final_outputs.append(K.sum(weighted_expert_output, axis=2))

        return final_outputs

    def compute_output_shape(self, input_shape):
        assert input_shape is not None and len(input_shape) >= 2

        output_shape = (None, self.units)
        return [output_shape for _ in range(self.num_tasks)]

    def get_config(self):
        config = {
            'units': self.units,
            'num_experts': self.num_experts,
            'num_tasks': self.num_tasks,
            'use_expert_bias': self.use_expert_bias,
            'use_gate_bias': self.use_gate_bias,
            'expert_activation': activations.serialize(self.expert_activation),
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'gate_bias_initializer': initializers.serialize(self.gate_bias_initializer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'gate_bias_regularizer': regularizers.serialize(self.gate_bias_regularizer),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'gate_bias_constraint': constraints.serialize(self.gate_bias_constraint),
            'expert_kernel_initializer': initializers.serialize(self.expert_kernel_initializer),
            'gate_kernel_initializer': initializers.serialize(self.gate_kernel_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'gate_kernel_regularizer': regularizers.serialize(self.gate_kernel_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
            'gate_kernel_constraint': constraints.serialize(self.gate_kernel_constraint),
        }
        base_config = super(MMoELayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class PLELayer(Layer):
    def __init__(self,
                 expert_units,  # 所有专家unit一致,
                 shared_experts_num,
                 specific_experts_num,  # [expertA's num, expertB's num...]
                 num_tasks,
                 is_output=False,

                 expert_kernel_initializer=None,
                 expert_kernel_regularizer=None,
                 expert_kernel_constraint=None,

                 expert_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 expert_bias_constraint=None,

                 use_gate_bias=False,

                 gate_kernel_initializer=None,
                 gate_kernel_regularizer=None,
                 gate_kernel_constraint=None,

                 gate_bias_initializer='zeros',
                 gate_bias_regularizer=None,
                 gate_bias_constraint=None,
                 **kwargs):
        self.expert_units = expert_units
        self.num_tasks = num_tasks
        self.specific_experts_num = specific_experts_num
        self.shared_experts_num = shared_experts_num
        assert isinstance(self.specific_experts_num, list) and self.num_tasks == len(self.specific_experts_num)

        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)

        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)

        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        self.use_gate_bias = use_gate_bias
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        self.is_output = is_output
        super(PLELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PLELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        assert len(inputs) == self.num_tasks + 1
        # inputs = [ExpertA , ExpertB , Shared]

        specific_inputs = inputs[0:-1]
        shared_inputs = inputs[-1]

        # [(n , m_A , d) , (n , m_B , d)]
        specific_experts_output = [ExpertLayer(
            units=self.expert_units,
            num_experts=self.specific_experts_num[i],
            expert_kernel_initializer=self.expert_kernel_initializer
        )(specific_inputs[i]) for i in range(self.num_tasks)]

        # (n , m_S , d)
        shared_expert_output = ExpertLayer(
            units=self.expert_units,
            num_experts=self.shared_experts_num,
            expert_kernel_initializer=self.expert_kernel_initializer
        )(shared_inputs)

        outputs = []

        if self.is_output:
            # 是output层
            gate_outputs = [GatingNetwork(
                expert_num=self.specific_experts_num[i] + self.shared_experts_num,
                gate_kernel_initializer=self.gate_kernel_initializer,
                gate_kernel_regularizer=self.gate_kernel_regularizer,
                gate_kernel_constraint=self.gate_kernel_constraint,
            )(specific_inputs[i]) for i in range(self.num_tasks)]
            for i in range(self.num_tasks):
                experts_output = K.concatenate((specific_experts_output[i], shared_expert_output), axis=1)
                assert len(experts_output.shape) == 3
                outputs.append(K.tf.einsum('ij,ijk->ik', gate_outputs[i], experts_output))
        else:
            experts_num = []
            for i in range(self.num_tasks):
                experts_num.append(self.specific_experts_num[i] + self.shared_experts_num)
            experts_num.append(sum(self.specific_experts_num) + self.shared_experts_num)
            gate_outputs = [GatingNetwork(
                expert_num=experts_num[i],
                gate_kernel_initializer=self.gate_kernel_initializer,
                gate_kernel_regularizer=self.gate_kernel_regularizer,
                gate_kernel_constraint=self.gate_kernel_constraint,
            )(inputs[i]) for i in range(self.num_tasks + 1)]

            for i in range(self.num_tasks):
                experts_output = K.concatenate((specific_experts_output[i], shared_expert_output), axis=1)
                assert len(experts_output.shape) == 3
                outputs.append(K.tf.einsum('ij,ijk->ik', gate_outputs[i], experts_output))

            # share部分
            experts_output = concatenate(specific_experts_output + [shared_expert_output], axis=1)
            outputs.append(K.tf.einsum('ij,ijk->ik', gate_outputs[-1], experts_output))
        return outputs

    def compute_output_shape(self, input_shape):
        num = self.num_tasks if self.is_output else self.num_tasks + 1
        return [(None, self.expert_units) for _ in range(num)]

    def get_config(self):
        config = {
            'units': self.expert_units,
            'specific_num_experts': self.specific_experts_num,
            'shared_num_experts': self.shared_experts_num,
            'num_tasks': self.num_tasks,
            'use_gate_bias': self.use_gate_bias,
            "is_output": self.is_output,
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'gate_bias_initializer': initializers.serialize(self.gate_bias_initializer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'gate_bias_regularizer': regularizers.serialize(self.gate_bias_regularizer),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'gate_bias_constraint': constraints.serialize(self.gate_bias_constraint),
            'expert_kernel_initializer': initializers.serialize(self.expert_kernel_initializer),
            'gate_kernel_initializer': initializers.serialize(self.gate_kernel_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'gate_kernel_regularizer': regularizers.serialize(self.gate_kernel_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
            'gate_kernel_constraint': constraints.serialize(self.gate_kernel_constraint),
        }
        base = super(PLELayer, self).get_config()
        return dict(list(config.items()) + list(base.items()))
