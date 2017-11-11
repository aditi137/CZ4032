import numpy as np
import theano
import theano.tensor as T
import pandas as pd
import matplotlib.pyplot as plt


'''car_data = pd.read_csv('used-cars-database/cleaned_dataset.csv', sep=',', encoding="cp1252")
print(car_data.head())
print(car_data.describe())
corr_table = car_data.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
print(corr_table)
'''

car_data = np.loadtxt('Volkswagon/volkswagons_only.csv', delimiter=',')
sample_size = car_data.shape[0]
car_data = car_data[:sample_size]
X_data, Y_data = car_data[:,1:-1], car_data[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

print("X_data:", X_data)
print("Y_data:", Y_data)


#X_data, Y_data = shuffle_data(X_data, Y_data)

#separate train and test data
m = 3*X_data.shape[0] // 10
testX, testY = X_data[:m],Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

print("trainX:", trainX)
print("testX:", testX)

plt.figure()
plt.plot(np.arange(trainX.shape[0]), trainY)
plt.xlabel("Input Row")
plt.ylabel("Price")
plt.title("Training set Price distribution")
plt.show()