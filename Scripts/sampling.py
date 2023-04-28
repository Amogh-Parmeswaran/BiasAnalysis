import pandas as pd
import os
import random

root_dir = './Publications'  # Replace with the path to your root directory
output_dir = os.path.join(os.getcwd(), 'Samples')

data_frames = []
get_header =  0
for filename in os.listdir(root_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(root_dir, filename)
        df = pd.read_csv(file_path, header=get_header)

        # Take 5000 random rows from the data frame
        if get_header:
            random_rows = df.sample(n=5001)
            get_header = 1
        else:
            random_rows = df.sample(n=5000)

        # Append the random rows to the list of data frames
        data_frames.append(random_rows)

# Concatenate data frames
result = pd.concat(data_frames)

# Export the aggregated data to a new CSV file
result.to_csv('./Samples/samples.csv', index=False)