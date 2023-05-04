import csv
import os
import pickle

# This file can be used to confirm preprocessing worked by printing the number
# of rows in the output samples or other things upon edit

root_dir = './Samples'  # Replace with the path to your root directory

csv.field_size_limit(10 * 1024 * 1024)

publishers = ["CNBC", "CNN", "Economist", "Fox News", "The New York Times", "People", "Vice News", "Politico", "Reuters", "TMZ"]
for filename in os.listdir(root_dir):
    if filename.endswith('.csv') and filename != 'news-dataset.csv':

        print(filename)
        with open(os.path.join(root_dir, filename), 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                row_count = sum(1 for row in reader)
                print(row_count)
                # for row in reader:
                #     print(row['encoded_text'])
                #     if row['publication'] not in publishers:
                #         print(row['publication'])
                # print('after')
    
    else:
        print(filename)
        # with open(os.path.join(root_dir, filename), 'rb') as file:
        #     tensors = pickle.load(file)
        # print(tensors[0][0][834])
                
