import csv
import os
import pickle

# This file can be used to confirm preprocessing worked by printing the number
# of rows in the output samples or other things upon edit

root_dir = './Samples'  #  Replace with the path to your root directory

stemmed_rows = []
sample_rows = []
check = True
for filename in os.listdir(root_dir):
    new_rows = []
    if filename.endswith('.pickle'):
        print(filename)
        with open(os.path.join(root_dir, filename), 'rb') as f:
                data = pickle.load(f)
                for row in data:
                    if check:
                        print(row)
                        check = False
                    new_rows.append(row)
        if 'stemmed' in filename: stemmed_rows += new_rows
        else: sample_rows += new_rows
    
print('done')
print(len(stemmed_rows))
print(len(sample_rows))

with open(root_dir + '/sample_tensors.pickle', 'wb') as file:
    pickle.dump(sample_rows, file)
with open(root_dir + '/stemmed_tensors.pickle', 'wb') as file:
    pickle.dump(stemmed_rows, file)