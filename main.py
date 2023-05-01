import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text 
from sklearn.model_selection import train_test_split
import pandas as pd
from encoder import encode_text
from LinearModel import LinearModel
from CNNModel import CNNModel

publishers = ["CNBC", "CNN", "Economist", "Fox News", "People", "The New York Times", "People", "Vice News", "Politico", "Reuters", "TMZ"]

class ModelTrainer():
    def __init__(self, model, train_inputs, train_labels, test_inputs, test_labels, num_classes):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.test_inputs = test_inputs
        self.test_labels = test_labels
        self.model = model(num_classes)

    def train(self, epochs, batch_size):
        optimizer = tf.keras.optimizers.Adam()
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
            print("Training accuracy: {:.2f}%".format(train_accuracy * 100))

    def test(self):
        test_accuracy = self.evaluate(self.test_inputs, self.test_labels)
        print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

    def evaluate(self, inputs, labels):
        predictions = self.model(inputs)
        correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy.numpy()


def process_data(filepath):
    data = pd.read_csv(filepath)
    # Get all the articles and publications from the CSV of aggregate data
    article_data = list(data['article'])[:10]
    publication_data = list(data['publication'])[:10]

    # Replacing publisher with classifier index
    for i in range(len(publication_data)):
        publisher_index = publishers.index(publication_data[i])
        publication_data[i] = publisher_index
    
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4",
            trainable=True)
    for index, article in enumerate(article_data):
        article_data[index] = encode_text(article, preprocessor, encoder)
        print("Encoded: ", index)
        print(article_data[index])

    # Split article data and publication labels into training and testing sets
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(article_data, publication_data, train_size = 0.8, random_state = 100)


    # One hot encode publication labels
    train_labels, test_labels = tf.one_hot(train_labels, 10), tf.one_hot(test_labels, 10)

    return train_inputs, test_inputs, train_labels, test_labels

if __name__ == '__main__':
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4",
            trainable=True)
    a = encode_text("Ayush Gupta is a remarkable individual who has made a significant impact in the field of technology and entrepreneurship. Born in India, Ayush has spent the majority of his life in the United States, where he has excelled in academics and business ventures. In this essay, we will explore Ayush's background, education, career, and achievements. Ayush was born in Jaipur, Rajasthan, India, in 1997. He spent his early childhood in India before moving to the United States with his family at the age of 7. Ayush grew up in a family of entrepreneurs, which had a significant impact on his aspirations and mindset. His parents owned a small business in India before moving to the US, where they started a successful restaurant. Ayush's passion for technology began at an early age. He showed an aptitude for math and science in school, which led him to pursue a degree in computer science at the University of Illinois at Urbana-Champaign. At Illinois, Ayush excelled academically and was actively involved in extracurricular activities, including the Association for Computing Machinery (ACM) and the Illinois Robotics in Space (IRIS) student organization. He also participated in several hackathons and coding competitions, where he gained experience in developing software applications. After graduating from college in 2019, Ayush began his career as a software engineer at Microsoft in Redmond, Washington. At Microsoft, Ayush worked on the company's core services and was involved in developing backend systems that supported various Microsoft products. During his time at Microsoft, Ayush gained valuable experience in software engineering and gained a deeper understanding of the tech industry. In 2020, Ayush left Microsoft to pursue his entrepreneurial ambitions. He co-founded a startup called Almug Technologies, which aimed to provide a platform for people to create and share augmented reality (AR) experiences. The idea behind the startup was to make AR more accessible to people by simplifying the process of creating AR content. Almug Technologies raised seed funding from investors and was selected to participate in the prestigious Y Combinator startup accelerator program. Ayush's leadership and vision for the company were crucial to its success. Under his guidance, Almug Technologies developed a user-friendly platform that enabled people to create AR content without requiring any coding skills. The company gained traction and was featured in several publications, including TechCrunch and VentureBeat. In 2021, Ayush decided to step down as CEO of Almug Technologies to pursue new opportunities. He is currently working on a new startup called Flightprep, which aims to simplify the process of booking and managing flights for business travelers. Flightprep is still in its early stages, but Ayush's experience and track record suggest that it has the potential to be successful. Apart from his entrepreneurial ventures, Ayush is also involved in several social and philanthropic initiatives. He has donated to several charities and is a member of the Effective Altruism movement, which aims to use evidence and reason to determine the most effective ways to do good in the world. Ayush Gupta is a remarkable individual who has made a significant impact in the field of technology and entrepreneurship. His passion for technology and entrepreneurship, combined with his leadership skills and vision, have led him to create successful ventures that have gained recognition from investors and the media. Ayush's achievements are a testament to his hard work and dedication, and his story serves as an inspiration for aspiring entrepreneurs and technologists.", preprocessor, encoder)
    b = encode_text("Ayush Gupta is a remarkable individual who has made a significant impact in the field of technology and entrepreneurship. Born in India, Ayush has spent the majority of his life in the United States, where he has excelled in academics and business ventures. In this essay, we will explore Ayush's background, education, career, and achievements. Ayush was born in Jaipur, Rajasthan, India, in 1997. He spent his early childhood in India before moving to the United States with his family at the age of 7. Ayush grew up in a family of entrepreneurs, which had a significant impact on his aspirations and mindset. His parents owned a small business in India before moving to the US, where they started a successful restaurant. Ayush's passion for technology began at an early age. He showed an aptitude for math and science in school, which led him to pursue a degree in computer science at the University of Illinois at Urbana-Champaign. At Illinois, Ayush excelled academically and was actively involved in extracurricular activities, including the Association for Computing Machinery (ACM) and the Illinois Robotics in Space (IRIS) student organization. He also participated in several hackathons and coding competitions, where he gained experience in developing software applications. After graduating from college in 2019, Ayush began his career as a software engineer at Microsoft in Redmond, Washington. At Microsoft, Ayush worked on the company's core services and was involved in developing backend systems that supported various Microsoft products. During his time at Microsoft, Ayush gained valuable experience in software engineering and gained a deeper understanding of the tech industry. In 2020, Ayush left Microsoft to pursue his entrepreneurial ambitions. He co-founded a startup called Almug Technologies, which aimed to provide a platform for people to create and share augmented reality (AR) experiences. The idea behind the startup was to make AR more accessible to people by simplifying the process of creating AR content. Almug Technologies raised seed funding from investors and was selected to participate in the prestigious Y Combinator startup accelerator program. Ayush's leadership and vision for the company were crucial to its success. Under his guidance, Almug Technologies developed a user-friendly platform that enabled people to create AR content without requiring any coding skills. The company gained traction and was featured in several publications, including TechCrunch and VentureBeat.", preprocessor, encoder)
    print(a)
    print(b)
    print(a == b)
    train_inputs, test_inputs, train_labels, test_labels = process_data('Samples/random.csv')
    modelTrainer = ModelTrainer(LinearModel, train_inputs, train_labels, test_inputs, test_labels, 10)
    modelTrainer.train(10, 32)
    modelTrainer.test()

