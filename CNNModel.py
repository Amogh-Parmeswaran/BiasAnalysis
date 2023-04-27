import tensorflow as tf
from encoder import encode_text

class CNNModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='leaky_relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides = (2, 2), padding = 'same', activation='leaky_relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='leaky_relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    def call(self, inputs):
        encoded_text = encode_text(inputs)
        return self.model(encoded_text)