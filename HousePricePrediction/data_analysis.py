# coding: utf-8 -*-

import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10,6)

train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')

print("Train boyutu : ", train.shape)
print("Test boyutu : ", test.shape)
print(train.head())

# Data analizi islemleri START
print("SalePrice Istatistikleri\n", train.SalePrice.describe())

# Skewness : Hedef degerin carpikligini/daginikligini aldigimiz deger
# Veri carpik i se, log-transformation yapmak onemlidir.

# Log-Transformation : Regresyon yaparken, skewness varsa giris-donusumu yapilir.
print("Skew : ", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()

# Yukaridaki data, pozitif yonlu dagilim var.
# Datayi normal dagilima cevirirsek isimiz kolaylasir. (0'a yaklasacak datalar ve grafik dogru olacak)

# Log-transformation yapiliyor.
target = numpy.log(train.SalePrice)
print("Skew : ", target.skew())
plt.hist(target, color='blue')
plt.show()

numeric_features = train.select_dtypes(include=[numpy.number])
corr = numeric_features.corr()

print("\nKorelasyon Siralamasi")
print(corr['SalePrice'].sort_values(ascending=False), '\n')

# OverallQual niteligini inceleyelim
overall_vals = train.OverallQual.unique()
quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=numpy.median)

# Burada, overallqual degerinin degisimine oranla SalePrice nasil degisiyor inceleniyor.
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

# Burada Living area buyuklugunu gormek icin scatter grafigi ciziyoruz
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Living area square feet')
plt.show()
# Living area degeri arttikca, fiyatin arttigini goruyoruz. Ayni seyi 3. correlation icin yapalim

plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('SalePrice')
plt.xlabel('GarageArea')
plt.show()

# GarageArea ya baktigimizda, 1200'den buyukler bizim icin outlier dedigimiz bir class
# onlari temizleyelim
train_not_lier = train[train['GarageArea'] < 1200]
plt.scatter(x=train_not_lier['GarageArea'], y=numpy.log(train_not_lier.SalePrice))
plt.xlim(-200, 1600) # X limiti -200 ile 1600 arasinda olsun
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()

#Null degerleri yakalayacagiz
nulls = pandas.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# Sira geldi, numeric olmayan verileri incelemeye
print("\nSozel Degerler")
kategoriler = train.select_dtypes(exclude=[numpy.number])
print(kategoriler.describe())


# Street degerleri icinde, pave ve grvl ne kadar geciyor
print("Original \n")
print(train.Street.value_contents(), "\n")


train['enc_street'] = pandas.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pandas.get_dummies(train.Street, drop_first=True)

train['enc_street'] = pandas.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pandas.get_dummies(train.Street, drop_first=True)

data = train.select_dtypes(include=[np.number]).interpolate().dropna()

y = numpy.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()