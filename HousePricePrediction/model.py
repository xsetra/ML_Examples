# -*- coding: utf-8 -*-

import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10,6)

train_file_name = 'train.csv'
test_file_name = 'test.csv'


class LearningModel(object):

    def __init__(self):
        self.train = pandas.read_csv(train_file_name)
        self.test = pandas.read_csv(test_file_name)

        self.features = None
        self.model = None
        self.attribute_selection = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def __repr__(self):
        return "<LearningModel>"

    def feature_selection(self, wanted_features=None):
        if wanted_features is None:
            wanted_features = ['OverallQual',
                               'TotalBsmtSF',
                               'GrLivArea',
                               'BsmtFullBath',
                               'FullBath',
                               'Fireplaces',
                               'GarageCars',
                               'WoodDeckSF',
                               '3SsnPorch',
                               'enc_street']
        for feature in self.features.columns:
            if feature not in wanted_features:
                self.features = self.features.drop(feature, axis=1)
        return self.features

    def prepare_features(self):
        self.features = self.train.select_dtypes(include=[numpy.number]).interpolate().dropna()
        if self.attribute_selection is True:
            self.feature_selection()
        else:
            self.features.drop(['SalePrice', 'Id'], axis=1)

    def make_predict(self, data):
        return self.model.predict([data])

    @staticmethod
    def deger_gosterimi(data):
        return "<<<<< Normal Degeri >>>>>\n{}\n<<<<<<<< Degeri >>>>>>>>\n{}".format(data, numpy.exp(data))

    def show_predict(self, predict_data, real_data):
        prediction = self.make_predict(predict_data)
        print("*************\nTahmin Edilen Ev Fiyat Degerleri\n", self.deger_gosterimi(prediction[0]))
        print("\nEvin Gercek Degerleri\n", self.deger_gosterimi(real_data), "\n*************")


class LinearRegressionModel(LearningModel):
    def __init__(self, attribute_selection=False):
        super(LinearRegressionModel, self).__init__()
        self.attribute_selection = attribute_selection

    def __repr__(self):
        return "<LinearRegressionModel>"

    def one_hot_encoding(self):
        self.train['enc_street'] = pandas.get_dummies(self.train.Street, drop_first=True)
        self.test['enc_street'] = pandas.get_dummies(self.test.Street, drop_first=True)

    def build_model(self):
        self.one_hot_encoding()
        self.prepare_features()

        x = self.features
        y = numpy.log(self.train.SalePrice)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=.33, random_state=42)
        linear_regression = linear_model.LinearRegression()
        self.model = linear_regression.fit(self.x_train, self.y_train)

    def evaluate_performance(self):
        return "R^2 : {}".format(self.model.score(self.x_test, self.y_test))

    def predict_all_test_datas(self):
        predictions = self.model.predict(self.x_test)
        plt.scatter(predictions, self.y_test, color='red')
        plt.xlabel('Tahmin Edilen Fiyat')
        plt.ylabel('Gercek Degeri')
        plt.title('Linear Regression Model')
        plt.show()

    @staticmethod
    def find_best_predictions(test_verileri, gercek_degerler):
        minus_predict = []
        plus_predict = []
        i = 0
        for test in test_verileri:
            prediction = lrm.make_predict(test)
            val = prediction - gercek_degerler[i]
            if val < 0:
                minus_predict.append([val, i])
            else:
                plus_predict.append([val, i])
            i += 1

        minus_predict.sort(reverse=True)
        plus_predict.sort()
        print(plus_predict)
        print(minus_predict)


class RidgeModel(LearningModel):

    def __init__(self):
        super(RidgeModel, self).__init__()
        self.alpha = None

    def build_model(self):
        self.prepare_features()

        x = self.features
        y = numpy.log(self.train.SalePrice)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=.33, random_state=42)
        self.alpha = 1
        ridge_model = linear_model.Ridge(alpha=self.alpha)
        self.model = ridge_model.fit(self.x_train, self.y_train)

    def predict_all_test_datas(self):
        predictions = self.model.predict(self.x_test)
        plt.scatter(predictions, self.y_test, color='red')
        plt.xlabel('Tahmin Edilen Fiyat')
        plt.ylabel('Gercek Deger')
        plt.title('Ridge Regularization Alpha:{}'.format(self.alpha))

        overlay = 'R^2 : {}\nRMSE : {}'.format(self.model.score(self.x_test, self.y_test),
                                               mean_squared_error(self.y_test, predictions))
        plt.annotate(s=overlay, xy=(12.1, 10.6), size='x-large')
        plt.show()


if __name__ == '__main__':
    lrm = LinearRegressionModel(attribute_selection=False)
    lrm.build_model()
    print(lrm.evaluate_performance())

    test_verileri = lrm.x_test.as_matrix()
    gercek_degerler = lrm.y_test.as_matrix()

    lrm.find_best_predictions(test_verileri, gercek_degerler)

    lrm.show_predict(test_verileri[416], gercek_degerler[416])
    lrm.show_predict(test_verileri[296], gercek_degerler[296])


    lrm.predict_all_test_datas()

    train_datas = lrm.train.select_dtypes(include=[numpy.number])
    train_datas = train_datas.as_matrix()
    print(train_datas[0], type(train_datas[0]))
    print(train_datas[0].tolist())
    predict = lrm.make_predict(train_datas[0])
    print(numpy.exp(predict), " ? ", lrm.train.SalePrice[0])

    js_array = train_datas[0].tolist()
    new_train = numpy.asarray(js_array, dtype=numpy.float64)
    predict = lrm.make_predict(new_train)

    rm = RidgeModel()
    rm.build_model()
    rm.predict_all_test_datas()
