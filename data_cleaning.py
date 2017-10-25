import pandas as pd
import matplotlib.pyplot as plt

'''
Predict price of old cars? based on data? Hmmm......

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


What kind of model? Neural Network? Regression model? How to quantify names? Hmm...
Steps:
1) Clean data.

- Which attributes to keep, which to throw? First find what values the attributes hold. you can do that in python code.
'''


df = pd.read_csv('used-cars-database//autos.csv', sep =',', header = 0, encoding='cp1252')

sample_A = df.sample(100)
sample_A_description = df.describe()
list_of_col_names = df.columns

print(df.seller.unique())
print(df.offerType.unique())
print(df.abtest.unique())
print(df.nrOfPictures.unique())

df.drop(['seller', 'offerType', 'abtest', 'dateCrawled', 'nrOfPictures', 'lastSeen', 'postalCode', 'dateCreated'], axis='columns', inplace=True)



'''
df.loc[boolean] looks at table and picks out rows that satisfy specified condition. count() sums the number of rows, .count()
returns a series. get specific number by specifying ['column name']

'''
#print(df.loc[df.yearOfRegistration < 1950]['name'])
#this gives all the rows where yearOfRegistration fits condition specified

print("Too old: %d" % df.loc[df.yearOfRegistration < 1950]['name'].count())
print("Too cheap: %d" % df.loc[df.price < 100]['name'].count())
print("Too expensive: " , df.loc[df.price > 150000]['name'].count())
print("Too few km: " , df.loc[df.kilometer < 5000]['name'].count())
print("Too many km: " , df.loc[df.kilometer > 200000]['name'].count())
print("Too few PS: " , df.loc[df.powerPS < 10]['name'].count())
print("Too many PS: " , df.loc[df.powerPS > 500]['name'].count())


fuelTypes = df['fuelType'].unique()
#['benzin' 'diesel' nan 'lpg' 'andere' 'hybrid' 'cng' 'elektro']
# benzene, diesel, nan, lpg, other, hybrid, cng, electric

print("Fuel types: " ,fuelTypes)
print("Damages: " , df['notRepairedDamage'].unique()) #ja : yes, nein: no
print("Vehicle types: " , df['vehicleType'].unique())
# [nan 'coupe' 'suv' 'kleinwagen' 'limousine' 'cabrio' 'bus' 'kombi' 'andere']
# nan, sport sedan, suv, small car, limousine, convertible, bus, combination, other
print("Brands: " , df['brand'].unique())

#### Removing the duplicates
dsWithoutDuplicates = df.drop_duplicates(['name','price','vehicleType','yearOfRegistration'
                         ,'gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType'
                         ,'notRepairedDamage'])
dsWithoutDuplicates = dsWithoutDuplicates[
        (dsWithoutDuplicates.yearOfRegistration <= 2016)
      & (dsWithoutDuplicates.yearOfRegistration >= 1950)
      & (dsWithoutDuplicates.price >= 100)
      & (dsWithoutDuplicates.price <= 150000)
      & (dsWithoutDuplicates.powerPS >= 10)
      & (dsWithoutDuplicates.powerPS <= 500)]


print("-----------------\nData kept for analisys: %d percent of the entire set\n-----------------" % (100 * dsWithoutDuplicates['name'].count() / df['name'].count()))


print(dsWithoutDuplicates.isnull().sum())

categories = ['gearbox', 'model', 'brand', 'vehicleType', 'fuelType', 'notRepairedDamage']

dsWithoutDuplicates['notRepairedDamage'].fillna(value='not-declared', inplace=True)
dsWithoutDuplicates['fuelType'].fillna(value='not-declared', inplace=True)
dsWithoutDuplicates['gearbox'].fillna(value='not-declared', inplace=True)
dsWithoutDuplicates['vehicleType'].fillna(value='not-declared', inplace=True)
dsWithoutDuplicates['model'].fillna(value='not-declared', inplace=True)


gearBoxVal = df['gearbox'].unique()
gearBoxRep = ['automatic','manual','not-declared']

vehicleTypeVal = df['vehicleType'].unique()
vehicleTypeRep = ['not-declared','sport sedan', 'suv', 'small car', 'limousine', 'convertible', 'bus', 'combination', 'other']

fuelTypeVal = df['fuelType'].unique()
fuelTypeRep = ['benzene','diesel','not-declared','lpg','other','hybrid', 'cng', 'electric']

notRepairedDamageVal = df['notRepairedDamage'].unique()
notRepairedDamageRep = ['not-declared','yes','no']




dsWithoutDuplicates['gearbox'].replace(gearBoxVal,gearBoxRep, inplace=True)
dsWithoutDuplicates['vehicleType'].replace(vehicleTypeVal, vehicleTypeRep, inplace=True)
dsWithoutDuplicates['fuelType'].replace(fuelTypeVal, fuelTypeRep, inplace=True)
dsWithoutDuplicates['notRepairedDamage'].replace(notRepairedDamageVal, notRepairedDamageRep, inplace=True)

cols = list(dsWithoutDuplicates.columns.values)
print(cols)
cols = ['name', 'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType', 'brand', 'notRepairedDamage', 'price']
dsWithoutDuplicates = dsWithoutDuplicates[cols]
print(dsWithoutDuplicates)
dsWithoutDuplicates.to_csv('cleaned_dataset.csv', sep=',')


'''print(dsWithoutDuplicates.isnull().sum())

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



