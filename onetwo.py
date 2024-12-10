import pandas as pd

melb_data_path = 'melb_data.csv'
melb_data = pd.read_csv(melb_data_path)

print(melb_data.describe())