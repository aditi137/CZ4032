import pandas as pd
import numpy as np
'''results = np.loadtxt('Results/pred_testY.csv', delimiter=',')
print(results)
list_diff = results[:,2]

print("number of rows: " , results.shape[0])


acceptable_price_range = 1500

list_index = [index if abs(list_diff[index])<acceptable_price_range else -1 for index in np.arange(list_diff.shape[0])]
#remove all the -1 from the list
list_index = list(filter(lambda a: a != -1, list_index))

accepted_results = results[list_index]
print("number of rows accepted: " , accepted_results.shape[0])
portion_accepted = accepted_results.shape[0]/results.shape[0] * 100
print("percentage of price within", acceptable_price_range ,"range: ", portion_accepted)
mean_absolute_error = np.mean(results[:,2]/results[:,1])
print("mean_absolute_error: " , mean_absolute_error)'''


##rewrite as class

class ResultAnalyser:
    _results = ""
    _acceptable_price_range = 0
    _mean_relative_absolute_error =0
    _mean_absolute_error=0

    def __init__(self, file, range):
        self._results = np.loadtxt(file, delimiter=',')
        self._acceptable_price_range = range

    def setRange(self, range):
        _acceptable_price_range_range = range

    def analyse(self):
        list_diff = self._results[:,2]
        print("number of rows: " , self._results.shape[0])
        list_index = [index if abs(list_diff[index])<self._acceptable_price_range else -1 for index in np.arange(list_diff.shape[0])]
        #remove all the -1 from the list
        list_index = list(filter(lambda a: a != -1, list_index))

        accepted_results = self._results[list_index]
        print("number of rows accepted: " , accepted_results.shape[0])
        portion_accepted = accepted_results.shape[0]/self._results.shape[0] * 100
        print("percentage of price within", self._acceptable_price_range ,"range: ", portion_accepted)
        self._mean_relative_absolute_error = np.mean(abs(self._results[:,2])/self._results[:,1])
        print("mean_relative_absolute_error: " , self._mean_relative_absolute_error)
        self._mean_absolute_error = np.mean(abs(self._results[:,2]))
        print("mean_absolute_error: " , self._mean_absolute_error)



resultView = ResultAnalyser('Results/pred_testY.csv', 1500)
resultView.analyse()

