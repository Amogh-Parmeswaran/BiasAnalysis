import pandas as pd
import os
from encoder import encode_text
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text 
import pickle

# Additional preprocessing for analysis used to stem words and remove stops

root_dir = './Samples'  # Replace with the path to your root directory
output_dir = os.path.join(os.getcwd(), 'Samples')

data_frames = []
get_header =  0
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4",
        trainable=False)

for filename in os.listdir(root_dir):
    if filename.endswith('.csv') and filename != "random.csv":
        print(filename)
        file_path = os.path.join(root_dir, filename)
        df = pd.read_csv(file_path, header=0)

        tensors = []
        thousand_counter = 1
        for index, row in df.iterrows():
            encoded_text = encode_text(row["article"], preprocessor, encoder)
            tensors.append(encoded_text)

            if (index + 1) % 100 == 0:
                print(filename + ', ' + str(index + 1) + " encoded")
            
            if (index + 1) % 1000 == 0:
                print(filename + ', ' + str(thousand_counter) + " thousand done")
                with open(output_dir + '/tensors_' + filename[:-4] + str(thousand_counter) + '.pickle', 'wb') as file:
                    pickle.dump(tensors, file)
                thousand_counter += 1
                tensors = []


        
       
   