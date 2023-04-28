import tensorflow as tf

def encode_text(article_text, preprocessor, encoder):
    # NOTE: The following code on creating the encoder was taken from
    # https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4
    # which details basic usage of creating a model to encode text
    # using BERT. 
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    encoder_inputs = preprocessor(text_input)
    outputs = encoder(encoder_inputs)
    # Use pooled_output since we want to represent each input sequence as a whole
    # which will be useful for classification of the article text 
    pooled_output = outputs["pooled_output"]
    # Create a Keras model to extract the encoded features
    embedding_model = tf.keras.Model(text_input, pooled_output)
    
    # Encode the text
    article_text = tf.constant([article_text])
    return embedding_model(article_text) 