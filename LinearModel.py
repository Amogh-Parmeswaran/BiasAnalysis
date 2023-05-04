import tensorflow as tf

class LinearModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(LinearModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation = 'relu')
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(128, activation = 'relu')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        # Classification task so want to output a probability distribution over the classes
        self.dense3 = tf.keras.layers.Dense(num_classes, activation = 'softmax')
    
    def call(self, inputs):
        # Concatenate the inputs together into one Tensor
        inputs = tf.concat(inputs, axis = 0)
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        return x
