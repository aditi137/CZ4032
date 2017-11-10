import numpy as np
import theano
import theano.tensor as T
import pandas as pd
import matplotlib.pyplot as plt


car_data = pd.read_csv('used-cars-database/cleaned_dataset.csv', sep=',', encoding="cp1252")
print(car_data.head())
print(car_data.describe())
corr_table = car_data.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
print(corr_table)
