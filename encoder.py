import tensorflow as tf
import math

def encode_text(article_text, preprocessor, encoder):
    # NOTE: The following code on creating the encoder was inspired by
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
        encoder_inputs = preprocessor(segment_input)
        segment_output = encoder(encoder_inputs)["pooled_output"] # [batch_size, 1024]
        
        # Add the encoded features of the segment to the list
        encoded_segments.append(segment_output)
    
    # Stack the encoded features of all segments and then compute the average 
    # pooled output of all the segments
    stacked_encodings = tf.stack(encoded_segments)
    pooled_output = tf.reduce_mean(stacked_encodings, axis=0)

    return pooled_output