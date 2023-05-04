import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text 
from sklearn.model_selection import train_test_split
import pandas as pd
from LinearModel import LinearModel
from CNNModel import CNNModel
import pickle
from LSTMModel import LSTMModel

publishers = ["CNBC", "CNN", "Economist", "Fox News", "The New York Times", "People", "Vice News", "Politico", "Reuters", "TMZ"]

class ModelTrainer():
    def __init__(self, model, train_inputs, train_labels, test_inputs, test_labels, num_classes):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.test_inputs = test_inputs
        self.test_labels = test_labels
        self.model = model(num_classes)

    def train(self, epochs, batch_size, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        for epoch in range(epochs):
            print("Epoch: {}".format(epoch + 1))
            # Shuffle the training data for each epoch
            indices = tf.range(start=0, limit=tf.shape(self.train_inputs)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            train_inputs = tf.gather(self.train_inputs, shuffled_indices)
            train_labels = tf.gather(self.train_labels, shuffled_indices)

            for batch in range(0, len(train_inputs), batch_size):
                inputs = train_inputs[batch : batch + batch_size]
                labels = train_labels[batch : batch + batch_size]

                with tf.GradientTape() as tape:
                    predictions = self.model(inputs)
                    loss = loss_fn(labels, predictions)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            train_accuracy = self.evaluate(self.train_inputs, self.train_labels)
            train_percentage = round(train_accuracy * 100, 2)
            print(f"Training accuracy: {train_percentage}%")

    def test(self):
        test_accuracy = self.evaluate(self.test_inputs, self.test_labels)
        test_percentage = round(test_accuracy * 100, 2)
        print(f"Test accuracy: {test_percentage}%")

    def evaluate(self, inputs, labels):
        predictions = self.model(inputs)
        correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy.numpy()


def process_data(picklepath, csvpath):
    print(picklepath)
    with open(picklepath, 'rb') as f:
        encodings = pickle.load(f)
    data = pd.read_csv(csvpath)

    print('Loaded data')
    # Get all the articles and publications from the CSV of aggregate data
    article_data = list(encodings)
    publication_data = list(data['publication'])

    # Replacing publisher with classifier index
    for i in range(len(publication_data)):
        publisher_index = publishers.index(publication_data[i])
        publication_data[i] = publisher_index

    # Split article data and publication labels into training and testing sets
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(article_data, publication_data, train_size = 0.8, random_state = 100)

    # One hot encode publication labels
    train_labels, test_labels = tf.one_hot(train_labels, 10), tf.one_hot(test_labels, 10)

    return train_inputs, test_inputs, train_labels, test_labels

if __name__ == '__main__':
    # type = "sample"
    type = "stemmed"

    csv = "samples.csv"
    if type != "sample": 
        csv = "stemmed_samples.csv"

    train_inputs, test_inputs, train_labels, test_labels = process_data('Samples/' + type + '_tensors.pickle', 'Samples/' + csv)
    modelTrainer = ModelTrainer(CNNModel, train_inputs, train_labels, test_inputs, test_labels, 10)
    modelTrainer.train(20, 50, 0.0018)
    modelTrainer.test()

