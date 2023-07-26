# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import csv
import json
from tqdm import tqdm

def convert_csv_to_json(csvfile, split):
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        lines = []
        for line in reader:
            lines.append(line)
    
    for line in tqdm(lines):
        with open(f"{split}.json", "a", encoding="utf-8") as file:
            label = int(line[0])
            review = line[1]
            json_str = json.dumps({"text": review, "label": label})
            file.write(json_str + "\n")

convert_csv_to_json("/home/v-junyan/datasets/yelp/dev.csv", "dev")
convert_csv_to_json("/home/v-junyan/datasets/yelp/train.csv", "train")
