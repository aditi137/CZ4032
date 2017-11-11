import pandas as pd
from sklearn import preprocessing as prep

#get a dataset containing attributes: km, powerPS, yearRegistered, vehicleType, brand='volkswagen'

df = pd.read_csv('datasets/cleaned_dataset.csv', sep=',', header=0, encoding='utf8')
#
df_vw = pd.DataFrame(df.loc[df['brand']=='volkswagen'])
#print(df_vw.head())
df_vw.drop(['name', 'monthOfRegistration'], axis='columns', inplace=True)
print(df_vw.head())

dsContValues = pd.concat([df_vw['powerPS'], df_vw['kilometer'], df_vw['yearOfRegistration'], df_vw['price']], axis=1)
#First, scale continuous values except price.
scaler = prep.MinMaxScaler()
#dsScaledValues is a numpy array
dsScaledValues = scaler.fit_transform(dsContValues[dsContValues.columns[:-1]])
#create dataframe from scaled values
dsScaledValues = pd.DataFrame(data=dsScaledValues[:,:], columns=[dsContValues.columns[:-1]], index=dsContValues.index.values)


print(df_vw.head())


df_vw_gearbox = pd.get_dummies(df_vw['gearbox'])
df_vw_notRepairedDamage = pd.get_dummies(df_vw['notRepairedDamage'])
df_vw_model = pd.get_dummies(df_vw['model'])
df_vw_fuelType = pd.get_dummies(df_vw['fuelType'])
df_vw_vehicleType = pd.get_dummies(df_vw['vehicleType'])



'''
df_vw = pd.concat([df_vw.drop(['gearbox'], axis=1), df_vw_gearbox], axis=1)
df_vw = pd.concat([df_vw.drop(['notRepairedDamage'], axis=1), df_vw_notRepairedDamage], axis=1)
df_vw = pd.concat([df_vw.drop(['model'], axis=1), df_vw_model], axis=1)
df_vw = pd.concat([df_vw.drop(['fuelType'], axis=1), df_vw_fuelType], axis=1)
'''

df_vw = pd.concat([dsScaledValues,df_vw_gearbox, df_vw_notRepairedDamage ,df_vw_model,df_vw_fuelType,df_vw_vehicleType, dsContValues['price']], axis=1)

print(df_vw.head())


m = 3*df_vw.shape[0] // 10
testSet = df_vw[:m]
trainingSet = df_vw[m:]
df_vw.to_csv('Volkswagon/volkswagons_only.csv', sep=',', header=0, index=0)
df_vw.to_csv('Volkswagon/volkswagons_only_w_header.csv', sep=',', header=1, index=0)
trainingSet.to_csv('Volkswagon/trainingSet_vw.csv',sep=',', header=0, index=False)
trainingSet.to_csv('Volkswagon/trainingSet_vw_w_header.csv',sep=',', header=1, index=False)
testSet.to_csv('Volkswagon/testSet_vw.csv',sep=',',header=0, index = False)
testSet.to_csv('Volkswagon/testSet_vw_w_header.csv',sep=',',header=1, index = False)