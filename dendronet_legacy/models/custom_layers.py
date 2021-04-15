import tensorflow as tf
from tensorflow.keras import layers


class AdditionLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AdditionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape(input_shape)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        # Make sure to call the `build` method at the end
        super(AdditionLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.add(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(AdditionLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class StaticAdditionLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(StaticAdditionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape(input_shape)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=False)
        # Make sure to call the `build` method at the end
        super(StaticAdditionLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.add(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(StaticAdditionLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MatmulLayer(layers.Layer):
    def __init__(self, layer_shape, **kwargs):
        self.layer_shape = layer_shape
        self.output_dim = layer_shape[-1]
        super(MatmulLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape(self.layer_shape)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        # Make sure to call the `build` method at the end
        super(MatmulLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(tf.transpose(self.kernel), inputs)
        # return tf.squeeze(tf.tensordot(inputs, tf.transpose(self.kernel), axes=[1, 1]), axis=-1)

    def compute_output_shape(self, input_shape):
        # shape = tf.TensorShape(input_shape).as_list()
        # shape[-1] = self.output_dim
        shape = input_shape[-1]
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MatmulLayer, self).get_config()
        base_config['output_dim'] = self.shaoe
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class StaticMatmulLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(StaticMatmulLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((1, self.output_dim))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=False)
        # Make sure to call the `build` method at the end
        super(StaticMatmulLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(StaticMatmulLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

