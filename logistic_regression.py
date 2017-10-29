import numpy as np
import theano
import theano.tensor as T
import matplotlib as plt

floatX = theano.config.floatX
# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)

def normalize(X, X_mean, X_std):
    return (X - X_mean)/X_std

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
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


usedCarDs = np.loadtxt('dsFinal.csv', delimiter=',')
X_data, Y_data = usedCarDs[:,:-1], usedCarDs[:,-1]
print(X_data)
print(Y_data)

sample_size = 20

X_data, Y_data = X_data[:sample_size], Y_data[:sample_size]
Y_dataMin, Y_dataMax = np.min(Y_data, axis=0), np.max(Y_data, axis=0)
Y_data = scale(Y_data, Y_dataMin, Y_dataMax)
Y_data = (np.asmatrix(Y_data)).transpose()

#scale Y_data


train_ratio = 7
test_ratio = 3
m = X_data.shape[0] * train_ratio //10
train_X_data, train_Y_data = X_data[:m], Y_data[:m]
test_X_data, test_Y_data = X_data[m:], Y_data[m:]

print("train_X_data: " , train_X_data)
print("train_Y_data: ", train_Y_data)

n = train_X_data.shape[0]

#linear regression neuron.
no_features = X_data.shape[1]
w1 = init_weights(no_features,1,False)
b1 = init_bias(1)

print("w1: ", w1.get_value())
print("b1: ", b1.get_value())

x = T.matrix('x') #input matrix
d = T.matrix('d') #desired output matrix

y = T.dot(x,w1) + b1
learning_rate = 0.001
alpha = theano.shared(learning_rate, floatX)

cost = T.abs_(T.mean(T.sqr(d - y)))
accuracy = T.mean(d - y)

#define gradients
dw1, db1 = T.grad(cost, [w1,b1])
train = theano.function(
        inputs = [x, d],
        outputs = cost,
        updates = [[w1, w1 - alpha*dw1],
                   [b1, b1 - alpha*db1]],
        allow_input_downcast=True
        )
test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
    )
epochs = 10
batch_size = 32

train_cost = np.zeros(epochs)
test_cost = np.zeros(epochs)
test_accuracy = np.zeros(epochs)

min_error = 1e+15
best_iter = 0
best_w1 = np.zeros(no_features)
best_b1 = 0

##Okay. linear regression doesnt work. Need multi-layer perceptron, with a linear regression output neuron. Okay. Fair enough.
for iter in range(epochs):
    if iter % 100 == 0:
        print(iter)

    train_X_data, train_Y_data = shuffle_data(train_X_data, train_Y_data)
    print("train_X_data: " , train_X_data)
    print("train_Y_data: ", train_Y_data)
    for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):#divide to mini batches
        #print("start: ", start, "end: ", end)
        train_cost[iter] += train(train_X_data[start:end], train_Y_data[start:end])/(n//batch_size)
    pred, test_cost[iter], test_accuracy[iter] = test(test_X_data, test_Y_data)
    print("pred: " ,pred)
    print("test_cost[iter]: " , test_cost[iter])
    print("test_accuracy[iter]:", test_accuracy[iter])
    if test_cost[iter] < min_error:
        best_iter = iter
        min_error = test_cost[iter]
        best_w1 = w1.get_value()
        best_b1 = b1.get_value()
        print("best_w1: ",best_w1)
        print("best_b1: ", best_b1)

w1.set_value(best_w1)
b1.set_value(best_b1)
best_pred, best_cost, best_accuracy = test(test_X_data, test_Y_data)


print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))

#Plots
plt.figure()
plt.plot(range(epochs), train_cost, label='train error')
plt.plot(range(epochs), test_cost, label = 'test error')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.title('Training and Test Errors at Alpha = %.3f'%learning_rate)
plt.legend()
plt.savefig('P_1B-(1)MSE.png')
plt.show()

plt.figure()
plt.plot(range(epochs), test_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.savefig('P_1B-(1)Accuracy.png')
plt.show()


#print(train_X_data)
#print(train_Y_data)