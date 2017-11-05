import pandas as pd
from sklearn import preprocessing as prep
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.style.use='default'
'''
Predict price of old cars? Hmmm......

Attributes:

1) date crawled
2) name
3) seller
4) Offer type
5) Price
6) abtest - wtf is this
7) vehicle type
8) yearOfRegistration
9) gearbox
10) powerPS
11) model
12) kilometer
13) monthOfRegistration
14) fuel type
15) Brand
16) Not Repaired Damage
17) Date created
18) Number of pictures
19) Postal Code
20) last seen


What kind of model? Neural Network? Regression model? How to quantify names? Dummy variables! Hmm...
Steps:
1) Clean data.

- Which attributes to keep, which to throw? First find what values the attributes hold. you can do that in python code.
'''

# 1st stage of data-cleaning procedure =======================================================================================================================
df = pd.read_csv('used-cars-database//autos.csv', sep =',', header = 0, encoding='cp1252')

sample_A = df.sample(100) #test code to get first 100 samples
sample_A_description = df.describe() #.describe() gets count mean std min max + other properties of each column




df.drop(['seller', 'offerType', 'abtest', 'dateCrawled', 'nrOfPictures', 'lastSeen', 'postalCode', 'dateCreated'], axis='columns', inplace=True)
#remove columns named above. dateCrawled, dateCreated open to discussion. Not intending to use when predicting price for the moment.


#### Removing the duplicates
dsWD = df.drop_duplicates(['name','price','vehicleType','yearOfRegistration'
                         ,'gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType'
                         ,'notRepairedDamage'])


# remove outliers. range values can be changed
dsWD = dsWD[
        (dsWD.yearOfRegistration <= 2016)
      & (dsWD.yearOfRegistration >= 1950)
      & (dsWD.price >= 100)
      & (dsWD.price <= 150000)
      & (dsWD.powerPS >= 10)
      & (dsWD.powerPS <= 500)]





print("-----------------\nData kept for analisys: %d percent of the entire set\n-----------------" % (100 * dsWD['name'].count() / df['name'].count()))

#dsWD.to_csv('data_without_duplicates.csv',sep=',')

#print(dsWD.isnull().sum())
#get count of columns that have null values. 5 columns containing values = nan
'''
name                       0
price                      0
vehicleType            10818
yearOfRegistration         0
gearbox                 5260
powerPS                    0
model                  11347
kilometer                  0
monthOfRegistration        0
fuelType               15400
brand                      0
notRepairedDamage      42124
'''

#for each category replace nan conditions with 'not declared'
dsWD['notRepairedDamage'].fillna(value='not-declared_nrp', inplace=True)
dsWD['fuelType'].fillna(value='not-declared_ft', inplace=True)
dsWD['gearbox'].fillna(value='not-declared_gb', inplace=True)
dsWD['vehicleType'].fillna(value='not-declared_vt', inplace=True)
dsWD['model'].fillna(value='not-declared_mo', inplace=True)

#check that there no more columns with null values
#print(dsWD.isnull().sum())


categories = ['gearbox', 'model', 'brand', 'vehicleType', 'fuelType', 'notRepairedDamage']




gearBoxVal = df['gearbox'].unique()
gearBoxRep = ['automatic','manual','not-declared_gb']

vehicleTypeVal = df['vehicleType'].unique()
vehicleTypeRep = ['not-declared_vt','sport sedan', 'suv', 'small car', 'limousine', 'convertible', 'bus', 'combination', 'other_vt']

fuelTypeVal = df['fuelType'].unique()
fuelTypeRep = ['benzene','diesel','not-declared_ft','lpg','other_ft','hybrid', 'cng', 'electric']

notRepairedDamageVal = df['notRepairedDamage'].unique()
notRepairedDamageRep = ['not-declared_nrp','yes','no']



#translate german words to english words. Only these 4 columns contain german words
dsWD['gearbox'].replace(gearBoxVal,gearBoxRep, inplace=True)
dsWD['vehicleType'].replace(vehicleTypeVal, vehicleTypeRep, inplace=True)
dsWD['fuelType'].replace(fuelTypeVal, fuelTypeRep, inplace=True)
dsWD['notRepairedDamage'].replace(notRepairedDamageVal, notRepairedDamageRep, inplace=True)


#rearrange column names, price is in last column.
cols = ['name', 'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType', 'brand', 'notRepairedDamage', 'price']

dsWD = dsWD[cols]
dsWD.to_csv('datasets/cleaned_dataset.csv', sep=',',index=0)
#end of 1st stage of data cleaning ==================================================================================================================================================

'''
Attributes to convert into dummy variables:
1) gearbox
2) notRepairedDamage
3) brand
4) model (combine model and brand into 1 column? Hmmmm)
5) vehicleType
6) fuelType

'''


#code for visualisation
att_values_to_plot = ['kilometer','vehicleType','brand','model','yearOfRegistration','fuelType','notRepairedDamage']
'''for attribute in att_values_to_plot:
    plt.figure()
    group = dsWD.groupby(attribute).size()
    dfTemp = pd.DataFrame(data = group)
    dfTemp.plot(kind='bar', figsize=(40,20))
    plt.savefig('/Images/' + attribute+'_distribution.png')
'''







#Convert categorical variables into dummy variables. When you do this, you get, Curse of dimentionality! Hmmmmm...
#For columns like models/brands that have many categories, you end up having a lot of 0s and one 1 only. Hmmmm....
dsWD_gearbox = pd.get_dummies(dsWD['gearbox'])
dsWD_notRepairedDamage = pd.get_dummies(dsWD['notRepairedDamage'])
dsWD_brand = pd.get_dummies(dsWD['brand'])
dsWD_model = pd.get_dummies(dsWD['model'])
dsWD_vehicleType = pd.get_dummies(dsWD['vehicleType'])
dsWD_fuelType = pd.get_dummies(dsWD['fuelType'])


dsDummyVar = pd.concat([dsWD_gearbox,dsWD_notRepairedDamage,dsWD_brand,dsWD_model,dsWD_vehicleType,dsWD_fuelType], axis=1)
dsDummyVar.to_csv('datasets/data_dummy_var.csv', sep=',')

print(dsWD_brand.columns)

#scale attributes that are continuous values? Between 0 and 1? Hmmmm

list_of_col_names = dsWD.columns #get list of column names.
for col_name in list_of_col_names:
    print(col_name, ", ", dsWD[col_name].unique())

print(dsWD.describe())


#Scale continuous values except price ============================================================================================================================
dsContValues = pd.concat([dsWD['powerPS'], dsWD['kilometer'], dsWD['monthOfRegistration'], dsWD['yearOfRegistration'], dsWD['price']], axis=1)
#dsContValues.to_csv('data_continuousAttributes.csv', sep=',')


#First, scale continuous values except price.
scaler = prep.MinMaxScaler()
#dsScaledValues is a numpy array
dsScaledValues = scaler.fit_transform(dsContValues[dsContValues.columns[:-1]])

#create dataframe from scaled values
dsScaledValues = pd.DataFrame(data=dsScaledValues[:,:], columns=[dsContValues.columns[:-1]], index=dsContValues.index.values)
print(dsContValues.head())
print(dsScaledValues.head())

#concat scaled values with price, output csv
dsFinal = pd.concat([dsScaledValues, dsContValues['price']], axis=1)
'''
Sample Output
    powerPS  kilometer  monthOfRegistration  yearOfRegistration
1  0.367347   0.827586             0.416667            0.924242
2  0.312245   0.827586             0.666667            0.818182
3  0.132653   1.000000             0.500000            0.772727
4  0.120408   0.586207             0.583333            0.878788
5  0.187755   1.000000             0.833333            0.681818
    powerPS  kilometer  monthOfRegistration  yearOfRegistration  price
1  0.367347   0.827586             0.416667            0.924242  18300
2  0.312245   0.827586             0.666667            0.818182   9800
3  0.132653   1.000000             0.500000            0.772727   1500
4  0.120408   0.586207             0.583333            0.878788   3600
5  0.187755   1.000000             0.833333            0.681818    650

'''

print(dsFinal.head())
dsFinal.to_csv('datasets/dsFinal.csv', sep=',', header = 0)
dsFinal.to_csv('datasets/dsFinalWithHeaders.csv', sep=',')

m = 3*dsFinal.shape[0] // 10
testSet = dsFinal[:m]
trainingSet = dsFinal[m:]

trainingSet.to_csv('datasets/trainingSet.csv',sep=',', header=0, index=False)
testSet.to_csv('datasets/testSet.csv',sep=',',header=0, index = False)


'''
Predict :
1. selling price of the car
(Not by order of weightage...let regression decide that)

- # km
- year registration
- brand
- model
- not repaired damage
- gearbox
- powerPS


2. predict if repairing the damages of the car will help to increase the car value for sale
- for the same brand and model, compare the # km, year registration and compare the prices before or after repair >>> 1 example

3. if the car will be scraped base on the certain set of attributes
4. based on when the car ad is placed and the value



Basic model:
Predict price. Attributes to use? Hmmm......

Need dummy variables for these columns:



'''





'''

for i, c in enumerate(categories):
    print("c: " , c)
    v = dsWithoutDuplicates[c].unique() #get unique values in for each attribute
    print("v: ", v)
    g = dsWithoutDuplicates.groupby(by=c)[c].count().sort_values(ascending=False)
    print(g)

    r = range(min(len(v), 5))
    #print(g.head())
    plt.figure(figsize=(5,3))
    plt.bar(r, g.head())
    #plt.xticks(r, v)
    plt.xticks(r, g.index)
    plt.show()


#Dataset is cleaned. What should we use to predict price? Model name? Brand? Months registered?
'''



