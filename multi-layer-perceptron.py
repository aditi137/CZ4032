import time
import numpy as np
import theano
import theano.tensor as T
import ResultAnalyser

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

floatX = theano.config.floatX

np.random.seed(10)

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

def set_bias(b, n = 1):
    b.set_value(np.random.randn(n)*0.01, floatX)

def set_weights(w, n_in=1, n_out=1, logistic=True):
    W_values = np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                 high=np.sqrt(6. / (n_in + n_out)),
                                 size=(n_in, n_out))
    if logistic == True:
        W_values *= 4
    w.set_value(W_values)

#read and divide data into test and train sets
car_data = np.loadtxt('Volkswagon/volkswagons_only.csv', delimiter=',')
sample_size = car_data.shape[0]
car_data = car_data[:sample_size]
X_data, Y_data = car_data[:,1:-1], car_data[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

print("X_data:", X_data)
print("Y_data:", Y_data)


X_data, Y_data = shuffle_data(X_data, Y_data)

#separate train and test data
m = 3*X_data.shape[0] // 10
testX, testY = X_data[:m],Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

print("trainX:", trainX)
print("testX:", testX)


# scale data
#trainX_max, trainX_min =  np.max(trainX, axis=0), np.min(trainX, axis=0)
#testX_max, testX_min =  np.max(testX, axis=0), np.min(testX, axis=0)

#trainX = scale(trainX, trainX_min, trainX_max)
#testX = scale(testX, testX_min, testX_max)

#print("trainX_Scale:", trainX)
#print("testX_Scale:", testX)

epochs = 2000
batch_size = 128
no_hidden1 = 20 #num of neurons in hidden layer 1
learning_rate = 0.000001
no_features = trainX.shape[1]
n = trainX.shape[0]
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output

# initialize weights and biases for hidden layer(s) and output layer
w_h1 = init_weights(no_features,no_hidden1)#(8x30 matrix)
b_h1 = init_bias(no_hidden1)#(1 bias for each neuron)
w_o = init_weights(no_hidden1)#(30x1 matrix)
b_o = init_bias() #(1 bias)

print("w_o", w_o.get_value())
print("b_o", b_o.get_value())


# learning rate
alpha = theano.shared(learning_rate, floatX)

#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)# ?x8.8x30 = ?x30
y = T.dot(h1_out, w_o) + b_o                  # ?x30.30x1 = ?x1

cost = T.abs_(T.mean(T.sqr(d - y)))
accuracy = T.mean(d - y)

#define gradients
dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

train = theano.function(
        inputs = [x, d],
        outputs = cost,
        updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h]],
        allow_input_downcast=True
        )

test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
    )

validate = theano.function(
    inputs = [x, d],
    outputs = cost,
    allow_input_downcast=True
    )

lr = [0.00001,0.000005,0.000001]
#lr = [0.00001]
noFolds = 3
train_cost = np.zeros([len(lr),epochs])
validate_cost = np.zeros([len(lr),epochs])

best_learning_rate = 0
min_error = 1e+15
test_accuracy = np.zeros(epochs)
test_cost = np.zeros(epochs)
pred_matrix = np.zeros([epochs, testX.shape[0]])


t = time.time()
for j in range (len(lr)):
    alpha.set_value(lr[j])
    print (alpha.get_value())

    for i in range (noFolds):#divide into folds
        start, end = (i*n//noFolds), ((i+1)*n//noFolds)
        validateX, validateY = trainX[start:end], trainY[start:end]
        tX, tY = (np.append(trainX[:start], trainX[end:], axis = 0)), (np.append(trainY[:start], trainY[end:], axis = 0))

        print ("learning rate", alpha.get_value(), "Fold no.", i+1)
        print("Reseting weights and biases")
        k = tX.shape[0]
        set_weights(w_o,no_hidden1)
        set_bias(b_o)
        set_weights(w_h1,no_features,no_hidden1)
        set_bias(b_h1,no_hidden1)

        for iter in range(epochs):
            if iter % 100 == 0:
                print(iter)

            tX, tY = shuffle_data(tX, tY)
            for start, end in zip(range(0, k, batch_size), range(batch_size, k, batch_size)):#divide to mini batches
                train_cost[j][iter] += train(tX[start:end], tY[start:end])/(k//batch_size)/noFolds
            validate_cost[j][iter] += validate(validateX, validateY)/noFolds
            if (i == (noFolds-1)):
                if (validate_cost[j][iter] < min_error):
                    min_error = validate_cost[j][iter]
                    best_learning_rate = lr[j]
        print("complete fold", i+1)
    print ("Complete validation on learning rate", alpha.get_value())

#print("Optimal Learning rate =", best_learning_rate)

set_weights(w_o,no_hidden1)
set_bias(b_o)
set_weights(w_h1,no_features,no_hidden1)
set_bias(b_h1,no_hidden1)
alpha.set_value(best_learning_rate)

print("training on optimal learning rate")
for iter in range(epochs):
    if iter % 100 == 0:
        print(iter)

    trainX, trainY = shuffle_data(trainX, trainY)
    for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):#divide to mini batches
        train(trainX[start:end], trainY[start:end])
    pred, test_cost[iter], test_accuracy[iter] = test(testX, testY)
    pred_trans = np.transpose(pred)
    pred_matrix[iter] = pred_trans

min_index = np.argmin(abs(test_accuracy))
print("min_index:", min_index)
print("test_accuracy:", test_accuracy[min_index])
pred = (pred_matrix[min_index])
pred = np.reshape(pred, (pred.size,1))


print("pred:" , pred)
print("testY:", testY)

price_diff = pred-testY

combined = np.concatenate((pred,testY, price_diff),axis=1)
np.savetxt('Results/pred_testY.csv', combined, delimiter=',')
#Plots
plt.figure()
i=0
for x,y in zip(train_cost, validate_cost):
    plt.plot(np.arange(epochs), x, label="train_cost" + lr[i])
    plt.plot(np.arange(epochs), y, label="validate_cost"+ lr[i])
    plt.xlabel("epochs")
    plt.ylabel("train_cost")
    plt.legend()
    plt.savefig("train_cost_validate_cost.png")
    i+=1

'''
plt.figure()
plt.plot(np.arange(epochs),validate_cost[0],label='Validation alpha = 0.001')
plt.plot(np.arange(epochs),validate_cost[1],label='Validation alpha=0.005')
plt.plot(np.arange(epochs),validate_cost[2],label='Validation alpha=0.0001')
plt.plot(np.arange(epochs),validate_cost[3],label='Validation alpha=0.0005')
plt.plot(np.arange(epochs),validate_cost[4],label='Validation alpha=0.00001')

#plt.plot(np.arange(epochs),train_cost[0],'--',label='Training alpha=0.001')

plt.plot(np.arange(epochs),train_cost[1],'--',label='Training alpha=0.005')
plt.plot(np.arange(epochs),train_cost[2],'--',label='Training alpha=0.0001')
plt.plot(np.arange(epochs),train_cost[3],'--',label='Training alpha=0.0005')
plt.plot(np.arange(epochs),train_cost[4],'--',label='Training alpha=0.00001')
plt.xlabel('Time (s)')
plt.ylabel('Mean Square Error')
plt.title('Training and validating for different learning rates')
plt.legend()
plt.savefig('training_error_validation_error.png')
'''

plt.figure()
plt.plot(np.arange(epochs),test_accuracy,label=('test accuracy'))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()
plt.savefig('Images/test_accuracy.png')

plt.figure()
plt.plot(np.arange(epochs),test_cost,label=('test_cost'))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Test Cost')
plt.legend()
plt.savefig('Images/test_cost.png')

plt.figure()
plt.plot(np.arange(testX.shape[0]),testY, label=('price in dataset'))
plt.plot(np.arange(testX.shape[0]),pred, label=('predicted price'))
plt.legend()
plt.xlabel('Input row')
plt.ylabel('Price')
plt.savefig('Images/Price_prediction.png')

'''
plt.figure()
plt.plot(np.arange(testX.shape[0]),pred, label=('predicted price'))
plt.xlabel('Input row')
plt.ylabel('Price')
#plt.savefig('Price_prediction.png')'''

plt.figure()
plt.plot(np.arange(testX.shape[0]), price_diff, label=("diff between actual price and predicted price"))
plt.legend()
plt.xlabel('Input row')
plt.ylabel('Price')
plt.savefig('Images/actual_vs_predict.png')
plt.show()

resultView = ResultAnalyser('Results/pred_testY.csv', 1500)
resultView.analyse()