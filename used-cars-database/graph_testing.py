import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt


car_data = np.loadtxt('../dsFinal.csv', delimiter=',')
sample_size = 20000
car_data = car_data[:sample_size]
X_data, Y_data = car_data[:,:-1], car_data[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()




m = 3*X_data.shape[0] // 10
testX, testY = X_data[:m],Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

print(testX)
print(testY)

plt.figure()
plt.plot(np.arange(testX.shape[0]), testY)

plt.show()