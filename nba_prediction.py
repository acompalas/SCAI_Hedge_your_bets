import csv
import pandas as pd
import os

# Specify the path to the parent folder ("SCAI_comp")
read_file = '/Users/andersoncompalas/Documents/SCAI_comp/datasets/nba_games.csv'

# Specify the path for the new CSV file
write_file = '/Users/andersoncompalas/Documents/SCAI_comp/datasets/games.csv'

# # Construct the full path to the CSV file
# csv_file_path = os.path.join(parent_folder, dataset_folder, file_name)

# Initialize an empty list to store the data
data_list = []

# Open the CSV file in read mode
with open(read_file, 'r') as csv_file:
    # Create a CSV reader
    csv_reader = csv.reader(csv_file)
    
    # Read the first row to get the column headers
    headers = next(csv_reader)
    
    # Iterate through each row in the CSV file and create a dictionary for each row
    for row in csv_reader:
        row_dict = {header: value for header, value in zip(headers, row)}
        data_list.append(row_dict)

# Open the new CSV file in write mode
with open(write_file, 'w', newline='') as new_csv_file:
    # Define the column headers based on the keys of the dictionaries
    headers = data_list[0].keys()

    # Create a CSV writer
    csv_writer = csv.DictWriter(new_csv_file, fieldnames=headers)
    
    # Write the headers to the new CSV file
    csv_writer.writeheader()
    
    # Write the data from the list of dictionaries to the new CSV file
    for row in data_list:
        csv_writer.writerow(row)
        
# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(write_file)

# Display the first few rows of the DataFrame to view the data
print(df.head())
        








