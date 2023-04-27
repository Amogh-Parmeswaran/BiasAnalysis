import tensorflow as tf
import numpy as np
from encoder import encode_text

class ModelTrainer():
    def __init__(self, train_data, test_data, num_classes, model):
        self.train_data = train_data
        self.test_data = test_data
        self.num_classes = num_classes
        self.model = model(num_classes)
    
    def train(self, epochs, batch_size, learning_rate):
        # Split the training data into input and target arrays
        x_train = self.train_data[:, 0]
        y_train = self.train_data[:, 1]
        
        # Compile the model with categorical cross-entropy loss and Adam optimizer
        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
        
        # Fit the model to the training data
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    
    def test(self):
        # Split the testing data into input and target arrays
        x_test = self.test_data[:, 0]
        y_test = self.test_data[:, 1]
        
        # Encode the text inputs and predict the class labels
        encoded_text = np.array([encode_text(text) for text in x_test])
        y_pred = np.argmax(self.model(encoded_text), axis=1)
        
        # Calculate the test accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"Test Accuracy: {accuracy}")

