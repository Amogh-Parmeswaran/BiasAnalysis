import tensorflow as tf

class LSTMModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(LSTMModel, self).__init__()
        # Reshape from (batch_size, 1024) into (batch_size, 1, 1024) since LSTM
        # requires Tensors of shape (batch_size, time_steps, input_dim).
        # NOTE: We choose time_steps size of 1 which is strange; however, 
        # we cannot break our encoding vector into smaller pieces since this would
        # not capture sequential dependencies since the encoding is already an average
        # pool of pooled_outputs 
        self.startReshape = tf.keras.layers.Reshape((-1, 1024))
        self.LSTM1 = tf.keras.layers.LSTM(256, return_sequences=True)
        self.LSTM2 = tf.keras.layers.LSTM(128)
        # Classification task so want to output a probability distribution over the classes
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
        # Reshape the output from size (batch_size, 1, 1024) to (batch_size, 1024) so the dimensions
        # are compatible for calculating Categorical Crossentropy Loss 
        self.endReshape = tf.keras.layers.Reshape((-1, ))
    
    def call(self, inputs):
        # Concatenate the inputs together into one Tensor
        inputs = tf.concat(inputs, axis = 0)
        x = self.startReshape(inputs)
        x = self.LSTM1(x)
        x = self.LSTM2(x)
        x = self.dense(x)
        x = self.endReshape(x)
        return x
