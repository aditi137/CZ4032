import numpy as np 
from six.moves import cPickle

filename = "train_cost.save"
f = open(filename, 'rb')
train_cost = cPickle.load(f)
f.close()

filename = "validate_cost.save"
f = open(filename, 'rb')
validate_cost = cPickle.load(f)
f.close()

filename = "best_learning_rate.save"
f = open(filename, 'rb')
best_learning_rate = cPickle.load(f)
f.close()

print(train_cost)
print(validate_cost)
print(best_learning_rate)

print(train_cost[:,-1])
print(validate_cost[:,-1])