import tensorflow as tf
from transformers import BertTokenizer, BertModel

def encoder(article_text):
    # Load a pretrained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Tokenize the input text
    input_ids = tokenizer.encode(article_text, add_special_tokens=True)
    # Convert the input text into a Tensor 
    input_ids = tf.constant([input_ids])

    # Forward pass through the BERT model to obtain embeddings
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state[0]

    return embeddings.numpy()



