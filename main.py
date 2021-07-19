from logging import debug
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# load a model
model = joblib.load('hiring_model.pkl')

@app.route('/')
def welcome():
    return render_template('base.html')

@app.route('/predict', methods = ['POST'])
def predict():

    exp = request.form.get('experience')
    score = request.form.get('test_score')
    interview_score = request.form.get('interview_score')

    prediction = model.predict([[int(exp), int(score), int(interview_score)]])

    output = round(prediction[0], 2)

    return render_template('base.html', prediction_text = f"Employee salary will be $ {output}")


if __name__== '__main__':
    app.run(debug=True)