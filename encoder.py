import tensorflow as tf
import math

def encode_text(article_text, preprocessor, encoder):
    # NOTE: The following code on creating the encoder was taken from
    # https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4
    # which details basic usage of creating a model to encode text
    # using BERT. 
    # Maximum input length for BERT
    max_length = 512
    
    # Create a list to store the encoded features of each segment
    encoded_segments = []
    
    # Split the input text into segments of max_length tokens
    num_segments = math.ceil(len(article_text) / max_length)
    for i in range(num_segments):
        start = i * max_length
        end = min((i + 1) * max_length, len(article_text))
        segment_text = article_text[start:end]
        
        # Encode each segment using BERT
        segment_input = tf.constant([segment_text])
        segment_output = encoder(preprocessor(segment_input))["pooled_output"]
        
        # Add the encoded features of the segment to the list
        encoded_segments.append(segment_output)
    
    # Concatenate the encoded features of all segments
    pooled_output = tf.reduce_mean(tf.stack(encoded_segments), axis=0)

        
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