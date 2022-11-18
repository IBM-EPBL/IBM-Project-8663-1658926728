import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import time
import pandas
import os
from Flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open(r'C:\Users\91882\IBM-Project-8663-1658926728\Project Development Phase\Sprint 1\RandomForestClassifier.pkl', 'rb'))
scale = pickle.load(open(r'C:\Users\91882\IBM-Project-8663-1658926728\Project Development Phase\Sprint 1\scale.pkl' , 'rb'))
                         
@app.route('/')# route to display the home page
def home():
    return render_template(r'C:\Users\91882\IBM-Project-8663-1658926728\Project Development Phase\Sprint 2\index.html') #rendering the home page
                         
@app.route('/predict', methods=["POST","GET1"])# route to show the predictions in a web UI
def predict():
# reading the inputs given by the user 
    input_feature=[x for x in request.form.values() ] 
    features_values=[np.array(input_feature)]
    names = [['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
        'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday',
        'WindGustDir', 'WindDir9am', 'WindDir3pm', 'year', 'month', 'day']]
    data = pandas.DataFrame(features_values,columns=names)
    data = scale.fit_transform(data)
    data = pandas.DataFrame(data,columns = names) 
    # predictions using the loaded model file 
    prediction=model.predict(data)
    pred_prob = model. predict_proba(data) 
    print(prediction)
    if prediction == "Yes":
        return render_template(r"C:\Users\91882\IBM-Project-8663-1658926728\Project Development Phase\Sprint 2\chance.html")
    else:
        return render_template(r"C:\Users\91882\IBM-Project-8663-1658926728\Project Development Phase\Sprint 2\nochance.html") # showing the prediction results in a UI
if __name__ == "__main__":
    app.run(debug = True,host='0.0.0.0',port=80)
