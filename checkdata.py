import numpy as np
import pandas as pd

data = pd.read_csv('NYC_Jobs.csv')

print(data.head())

print(data['Job Category'].value_counts())

# count per job category, number of positions that are open (# Of Positions), export to csv
# how long are they open, >30 days
