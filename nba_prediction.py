import csv
import pandas as pd
import os

# Specify the path to the parent folder ("SCAI_comp")
parent_folder = '/Users/andersoncompalas/Documents/SCAI_comp'

# Specify the name of the dataset folder
dataset_folder = 'datasets'

# Specify the name of the CSV file
file_name = 'nba_games.csv'

# Construct the full path to the CSV file
csv_file_path = os.path.join(parent_folder, dataset_folder, file_name)

# Initialize an empty dictionary to store the data
mydict = {}

# Open the CSV file in read mode
with open(csv_file_path, 'r') as csv_file:
    # Create a CSV reader
    csv_reader = csv.reader(csv_file)
    
    # Iterate through each row in the CSV file and create the dictionary
    for row in csv_reader:
        mydict[row[0]] = row[1]

# Display the first 5 rows from the dictionary
for key, value in list(mydict.items())[:5]:
    print(f'Key: {key}, Value: {value}')






