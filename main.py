import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


def encode_text(article_text):
    article_text = tf.constant([article_text])
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3",
        trainable=True)
    outputs = encoder(encoder_inputs)
    
    # Create a Keras model to extract the encoded features
    model = tf.keras.Model(inputs=text_input, outputs=outputs)
    
    # Encode the text
    encoded_text = model(article_text)
    
    return encoded_text

if __name__ == '__main__':
    print(4)
    print(encode_text("Hello World I want to shoot this computer so badly holy fuck"))