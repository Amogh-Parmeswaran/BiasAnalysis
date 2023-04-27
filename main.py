import tensorflow as tf
import numpy as np
from encoder import encode_text
from sklearn.model_selection import train_test_split
import pandas as pd
from LinearModel import LinearModel

class ModelTrainer():
    def __init__(self, model, train_data, test_data, num_classes):
        self.train_data = train_data
        self.test_data = test_data
        self.num_classes = num_classes
        self.model = model(num_classes)
    
    def train(self, epochs, batch_size, learning_rate):
        # Split the training data into input and target arrays
        (input_train, label_train) = self.train_data
        
        # Compile the model with categorical cross-entropy loss and Adam optimizer
        self.model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate), metrics = ['accuracy'])
        
        # Fit the model to the training data
        self.model.fit(input_train, label_train, epochs = epochs, batch_size = batch_size)
    
    def test(self):
        # Split the testing data into input and target arrays
        (input_test, label_test) = self.test_data
        
        # Encode the text inputs and predict the class labels
        encoded_text_array = []
        for article_text in input_test:
            encoded_text = encode_text(article_text)
            encoded_text_array.append(encoded_text)
        encoded_text_array = np.array(encoded_text_array)

        label_predictions = np.argmax(self.model(encoded_text_array), axis = 1)
        
        # Calculate the test accuracy
        accuracy = np.mean(label_predictions == label_test)
        print(f"Test Accuracy: {accuracy}")

def process_data(filepath):
    data = pd.read_csv(filepath)
    # Get all the articles and publications from the CSV of aggregate data
    article_data = np.array(data['article'])
    publication_data = np.array(data['publication'])
    # One hot encode the article names 
    article_onehot = tf.one_hot(article_data, 10)
    # Split the publication data and the one-hot encoded article data into training and testing sets
    input_train, input_test, label_train, label_test = train_test_split(publication_data, article_onehot, train_size = 0.8, random_state = 100)
    return input_train, input_test, label_train, label_test

if __name__ == '__main__':
    input_train, input_test, label_train, label_test = process_data('csv')
    modelTrainer = ModelTrainer(LinearModel, (input_train, label_train), (input_test, label_test), 10)
    modelTrainer.train(10, 32, 0.003)
    modelTrainer.test()

