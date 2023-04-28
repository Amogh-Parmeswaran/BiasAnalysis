import tensorflow as tf
from encoder import encode_text

class LinearModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(LinearModel, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation = 'leaky_relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation = 'leaky_relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation = 'softmax')
        ])
    
    def call(self, inputs):
        print('inputs')
        print(inputs)
        return self.model(inputs)
