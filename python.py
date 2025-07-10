import csv

csv_path = r'c:\Users\cervinka\cervinka\dataset_compressible_flow_60M_training_nstep180.csv'

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    print("Headings:", headers)