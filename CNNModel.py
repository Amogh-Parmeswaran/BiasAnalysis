import tensorflow as tf

class CNNModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='leaky_relu')
        self.maxpool = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='leaky_relu')
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='leaky_relu')
        self.flatten = tf.keras.layers.Flatten()
        # Classification task so want to output a probability distribution over the classes
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
        # Reshape the output from size (batch_size, 1, 1024) to (batch_size, 1024)
        self.reshape = tf.keras.layers.Reshape((-1, ))

    
    def call(self, inputs):
        # Concatenate the inputs together into one Tensor
        inputs = tf.concat(inputs, axis = 0)
        inputs = tf.expand_dims(inputs, axis = -1)
        x = self.conv1(inputs)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
