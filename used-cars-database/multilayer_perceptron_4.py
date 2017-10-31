import time
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(10)

floatX = theano.config.floatX

# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

def init_weights(n_in = 1, n_out = 1, logistic=True):
    W_values = np.asarray(
        np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                          high=np.sqrt(6. / (n_in + n_out)),
                          size=(n_in, n_out)),
        dtype=theano.config.floatX)
    if logistic == True:
        W_values *= 4
    return(theano.shared(value=W_values, name='W', borrow=True))

def init_bias(n = 1):
    return(theano.shared(np.random.randn(n)*0.01, floatX))

#read and divide data into test and train sets
car_data = np.loadtxt('../dsFinal.csv', delimiter=',')
sample_size = 50000
car_data = car_data[:10000]
X_data, Y_data = car_data[:,1:-1], car_data[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

X_data, Y_data = shuffle_data(X_data, Y_data)

#separate train and test data
m = 3*X_data.shape[0] // 10
testX, testY = X_data[:m],Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

print("trainX:", trainX)
print("trainY:", trainY)

# scale and normalize data
#trainX_max, trainX_min =  np.max(trainX, axis=0), np.min(trainX, axis=0)
#testX_max, testX_min =  np.max(testX, axis=0), np.min(testX, axis=0)

#trainX = scale(trainX, trainX_min, trainX_max)
#testX = scale(testX, testX_min, testX_max)

epochs = 5000
batch_size = 32
#4 layer networkd
no_hidden1 = 10 #num of neurons in hidden layer 1
no_hidden2 = 5
learning_rate = 0.00001
no_features = trainX.shape[1]
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output
n = trainX.shape[0]

# initialize weights and biases for hidden layer(s) and output layer
#1st hidden layer
w_h1 = init_weights(no_features,no_hidden1)
b_h1 = init_bias(no_hidden1)
#2nd hidden layer
w_h2_o = init_weights(no_hidden2)
b_h2_o = init_bias()
w_h2 = init_weights(no_hidden1, no_hidden2)
b_h2 = init_bias(no_hidden2)

print("w_h1:", w_h1)
print("b_h1:", b_h1)
print("w_h2_o:,", w_h2_o)
print("b_h2_o:,", b_h2_o)
print("w_h2:,", w_h2)
print("b_h2:,", b_h2)

# learning rate
alpha = theano.shared(learning_rate, floatX)
print("Learning Rate =", alpha.get_value())

#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
y = T.dot(h2_out, w_h2_o) + b_h2_o

cost = T.abs_(T.mean(T.sqr(d - y)))
accuracy = T.mean(d - y)

#define gradients
dw_h, db_h = T.grad(cost, [w_h1, b_h1])
dw_h2_o, db_h2_o, dw_h2, db_h2 = T.grad(cost, [w_h2_o, b_h2_o, w_h2, b_h2])

train = theano.function(
        inputs = [x, d],
        outputs = cost,
        updates = [[w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h],
				   [w_h2, w_h2 - alpha*dw_h2],
				   [b_h2, b_h2 - alpha*db_h2],
				   [w_h2_o, w_h2_o - alpha*dw_h2_o],
				   [b_h2_o, b_h2_o - alpha*db_h2_o],
				   ],
        allow_input_downcast=True
        )

test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
    )

train_cost4l = np.zeros(epochs)
test_cost4l = np.zeros(epochs)
test_accuracy4l = np.zeros(epochs)

t = time.time()
print("training 4 Layer NN")

for iter in range(epochs):
    if iter % 100 == 0:
        print(iter)

    trainX, trainY = shuffle_data(trainX, trainY)
    for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):#divide to mini batches
        train_cost4l[iter] += train(trainX[start:end], trainY[start:end])/(n//batch_size)
    pred4l, test_cost4l[iter], test_accuracy4l[iter] = test(testX, testY)


print("testY:", testY)
print("pred4l:", pred4l)


#Plots
plt.figure()
plt.plot(range(epochs), train_cost4l,label='train error for 4 layer')
plt.plot(range(epochs), test_cost4l, label = 'test error for 4 layer')
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Training and Test Errors for 4 Layer NN')
plt.legend()
plt.savefig('4Layer-MSE.png')


plt.figure()
plt.plot(range(epochs), test_accuracy4l, label = 'test accuracy for 4 layer')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.savefig('4Layer-Accuracy.png')


plt.figure()
plt.plot(np.arange(testX.shape[0]),testY, label = 'price in dataset')
plt.plot(np.arange(testX.shape[0]),pred4l, label = 'predicted price')

plt.show()