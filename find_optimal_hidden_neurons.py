import time
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from six.moves import cPickle

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

epochs = 1000
batch_size= 32
no_hidden_list = [20,25,30,35,40] #num of neurons in hidden layer 1
learning_rate = 0.00001
no_features = trainX.shape[1]
n = trainX.shape[0]
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output

# initialize weights and biases for hidden layer(s) and output layer
w_h1 = init_weights(no_features,no_hidden_list[0])#(8x30 matrix)
b_h1 = init_bias(no_hidden_list[0])#(1 bias for each neuron)
w_o = init_weights(no_hidden_list[0])#(30x1 matrix)
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

lr = [0.00001]
#lr = [0.00001]
noFolds = 5
train_cost = np.zeros([len(no_hidden_list), epochs])
validate_cost = np.zeros([len(no_hidden_list),epochs])

best_hidden_neurons = 0
min_error = 1e+15
test_accuracy = np.zeros(epochs)
test_cost = np.zeros(epochs)
pred_matrix = np.zeros([epochs, testX.shape[0]])


t = time.time()
for j in range (len(no_hidden_list)):
    #alpha.set_value(lr[j])
    print (alpha.get_value())

    for i in range (noFolds):#divide into folds
        start, end = (i*n//noFolds), ((i+1)*n//noFolds)
        #rows used to validate, still within training set
        validateX, validateY = trainX[start:end], trainY[start:end] 
        #rows used for training
        tX, tY = (np.append(trainX[:start], trainX[end:], axis = 0)), (np.append(trainY[:start], trainY[end:], axis = 0)) 
        k = tX.shape[0]

        set_weights(w_o,no_hidden_list[j])
        set_bias(b_o)
        set_weights(w_h1,no_features,no_hidden_list[j])
        set_bias(b_h1,no_hidden_list[j])

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
                    best_hidden_neurons = no_hidden_list[j]
        print("complete fold", i+1)
    print ("Complete validation on neuron layer size", no_hidden_list[j])

filename = 'result_components/find_optimal_hidden_neurons/validate_cost.save'
f = open(filename, 'wb')
cPickle.dump(validate_cost, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

filename = 'result_components/find_optimal_hidden_neurons/train_cost.save'
f = open(filename, 'wb')
cPickle.dump(train_cost,f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

filename = 'result_components/find_optimal_hidden_neurons/best_hidden_neurons.save'
f = open(filename, 'wb')
cPickle.dump(best_hidden_neurons, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()


print("Optimal number of hidden neurons size =", best_hidden_neurons)

set_weights(w_o,best_hidden_neurons)
set_bias(b_o)
set_weights(w_h1,no_features,best_hidden_neurons)
set_bias(b_h1,best_hidden_neurons)
#alpha.set_value(best_learning_rate)


full_train_cost = np.zeros([epochs])
for iter in range(epochs):
    if iter % 100 == 0:
        print(iter)

    trainX, trainY = shuffle_data(trainX, trainY)
    for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):#divide to mini batches
        full_train_cost[iter] += train(trainX[start:end], trainY[start:end]) / (n//batch_size)
    pred, test_cost[iter], test_accuracy[iter] = test(testX, testY)
    #pred_trans = np.transpose(pred)
    #pred_matrix[iter] = pred_trans

#min_index = np.argmin(abs(test_accuracy))
#print("min_index:", min_index)
#print("test_accuracy:", test_accuracy[min_index])
#pred = (pred_matrix[min_index])
#pred = np.reshape(pred, (pred.size,1))


print("pred:" , pred)
print("testY:", testY)

price_diff = pred-testY

combined = np.concatenate((pred,testY, price_diff),axis=1)
np.savetxt('result_components/find_optimal_hidden_neurons/pred_testY.csv', combined, delimiter=',')
#Plots
plt.figure()
i=0
for x,y in zip(train_cost, validate_cost):
    plt.plot(np.arange(epochs), x, label=("train_cost",no_hidden_list[i]))
    plt.plot(np.arange(epochs), y, label=("validate_cost",no_hidden_list[i]))
    plt.xlabel("epochs")
    plt.ylabel("train_cost")
    plt.legend()
    plt.savefig("result_components/find_optimal_hidden_neurons/train_cost_validate_cost.png")
    i+=1

plt.figure()
plt.plot(np.arange(epochs), full_train_cost, label=("training cost"))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Training Cost')
plt.legend()
plt.savefig('result_components/find_optimal_hidden_neurons/training_cost.png')


plt.figure()
plt.plot(np.arange(epochs),test_accuracy,label=('test accuracy'))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()
plt.savefig('result_components/find_optimal_hidden_neurons/test_accuracy.png')

plt.figure()
plt.plot(np.arange(epochs),test_cost,label=('test_cost'))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Test Cost')
plt.legend()
plt.savefig('result_components/find_optimal_hidden_neurons/test_cost.png')

plt.figure()
plt.plot(np.arange(testX.shape[0]),testY, label=('price in dataset'))
plt.plot(np.arange(testX.shape[0]),pred, label=('predicted price'))
plt.legend()
plt.xlabel('Input row')
plt.ylabel('Price')
plt.savefig('result_components/find_optimal_hidden_neurons/Price_prediction.png')

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
plt.savefig('result_components/find_optimal_hidden_neurons/actual_vs_predict.png')
plt.show()
