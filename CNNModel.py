import tensorflow as tf

class CNNModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        # Reshape from (batch_size, 1024) into (batch_size, 1024, 1) since Conv1D
        # requires Tensors of shape (batch_size, height, channels)
        self.startReshape = tf.keras.layers.Reshape((-1, 1))
        self.conv1 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='leaky_relu')
        self.maxpool = tf.keras.layers.MaxPooling1D()
        self.conv2 = tf.keras.layers.Conv1D(64, kernel_size=5, strides=2, padding='same', activation='leaky_relu')
        self.conv3 = tf.keras.layers.Conv1D(128, kernel_size=3, activation='leaky_relu')
        self.flatten = tf.keras.layers.Flatten()
        # Classification task so want to output a probability distribution over the classes
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
        # Reshape the output from size (batch_size, 1024, 1) to (batch_size, 1024) so the dimensions
        # are compatible for calculating Categorical Crossentropy Loss 
        self.endReshape = tf.keras.layers.Reshape((-1, ))

    
    def call(self, inputs):
        # Concatenate the inputs together into one Tensor
        inputs = tf.concat(inputs, axis = 0)
        x = self.startReshape(inputs)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.endReshape(x)
        return x
