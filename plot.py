import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


data = pd.read_csv('job_category_count.csv')
