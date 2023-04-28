import csv
import os

# This file can be used to confirm preprocessing worked

root_dir = './Samples'  # Replace with the path to your root directory

csv.field_size_limit(10 * 1024 * 1024)

publishers = ["CNBC", "CNN", "Economist", "Fox News", "People", "The New York Times", "People", "Vice News", "Politico", "Reuters", "TMZ"]
for filename in os.listdir(root_dir):
    if filename.endswith('.csv') and filename != 'news-dataset.csv':
        print(filename)
        with open(os.path.join(root_dir, filename), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            row_count = sum(1 for row in reader)
            print(row_count)
            for row in reader:
                if row['publication'] not in publishers:
                    print(row['publication'])
            print('after')
