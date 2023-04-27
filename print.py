import csv
import os

root_dir = '.'  # Replace with the path to your root directory

csv.field_size_limit(10 * 1024 * 1024)

for filename in os.listdir(root_dir):
    if filename.endswith('.csv') and filename != 'news-dataset.csv':
        print(filename)
        with open(os.path.join(root_dir, filename), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(row['article']) <= 1000:
                    print(row['article'])