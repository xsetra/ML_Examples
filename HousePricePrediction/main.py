# -*- coding: utf-8 -*-

from flask import Flask, request, Response
import json
from house_predictions.model import *
import numpy

app = Flask(__name__)
linear_model = LinearRegressionModel()
ridge_model = RidgeModel()

linear_model.build_model()
ridge_model.build_model()


def add_resp_headers(resp):
    resp.headers.add("Content-Type", "application/json")
    resp.headers.add("Access-Control-Allow-Origin", "*")


@app.route('/')
def index():
    resp = Response(json.dumps('{"status" : "PONG"}'))
    resp.headers.add("Content-Type", "application/json")
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp


@app.route('/house/linear', methods=["GET", "POST"])
def linear_regression():
    req = request.form.to_dict()
    new_req = None
    for key in req.keys():
        new_req = json.loads(key)
        break
    new_train = numpy.asarray(new_req['data'], dtype=numpy.float64)
    predict = linear_model.make_predict(new_train)
    predict = numpy.exp(predict)
    resp = Response()
    mesaj = '{"predicted":' +str(predict)+ '}'
    resp.data = json.dumps(mesaj)
    resp.headers.add("Content-Type", "application/json")
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp


@app.route('/house/ridge', methods=["POST", "GET"])
def ridge_regularization():
    req = request.form.to_dict()
    new_req = None
    for key in req.keys():
        new_req = json.loads(key)
        break
    new_train = numpy.asarray(new_req['data'], dtype=numpy.float64)
    resp = Response(json.dumps('{"predicted": '+str(ridge_model.model.predict(new_train))+'}'))
    resp.headers.add("Content-Type", "application/json")
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp


if __name__ == '__main__':
    app.run('localhost', 5000, debug=1)
