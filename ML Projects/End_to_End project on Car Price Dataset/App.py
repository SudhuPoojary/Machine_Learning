
import numpy as np
import pandas as pd
import pickle
import sklearn
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST','GET'])
def results():
    wheelbase = float(request.form['wheelbase'])
    carlength = float(request.form['carlength'])
    carwidth = float(request.form['carwidth'])
    carheight = float(request.form['carheight'])
    curbweight = float(request.form['curbweight'])
    enginesize = float(request.form['enginesize'])
    boreratio = float(request.form['boreratio'])
    stroke = float(request.form['stroke'])
    compressionratio = float(request.form['compressionratio'])
    horsepower = float(request.form['horsepower'])
    peakrpm = float(request.form['peakrpm'])
    citympg = float(request.form['citympg'])
    highwaympg = float(request.form['highwaympg'])


    X = np.array([[wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio,horsepower, peakrpm, citympg, highwaympg]])
    model = pickle.load(open('model.pkl', 'rb'))
    Y_predict = model.predict(X)
    return jsonify({'prediction' : float(Y_predict)})

if __name__ == '__main__':
    app.run(debug = True, port = 1010)
