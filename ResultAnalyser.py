import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt



class ResultAnalyser:
    _results = ""
    _acceptable_price_range = 0

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

        '''
                          
        Mean absolute error   - okay                
        Root mean squared error - okay              
                 
        Total Number of Instances             
        '''

        pred = self._results[:,0]
        actual = self._results[:,1]
        diff = self._results[:,2]

        accepted_results = self._results[list_index]
        #print(accepted_results)
        
        portion_accepted = accepted_results.shape[0]/self._results.shape[0] * 100

        mean_abs_error = np.mean(abs(diff))

        #root mean squared error
        rmse = sqrt(mean_squared_error(actual,pred))

        #mean absolute percentage error
        mape = np.mean(abs(diff)/actual) *100
        


        print("number of rows accepted: " , accepted_results.shape[0])
        print("percentage of price within", self._acceptable_price_range ,"range: ", portion_accepted)
        print("mean absolute error: ", mean_abs_error)
        print("rmse: " , rmse)
        print("mape: ", mape)



filename = input('Enter filepath')
resultView = ResultAnalyser(filename, 1500)
resultView.analyse()

