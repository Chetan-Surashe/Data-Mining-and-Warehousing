from flask import Flask, render_template, request, redirect, session
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Load your trained model from the file
model = joblib.load("DMW.pkl")

@app.route('/')
def index():
   
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Make a prediction using the loaded model
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        prediction = model.predict(input_data)

        result = "Heart Disease Detected" if prediction[0] == 1 else "Heart Disease Not Detected"

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('error.html', error=e)

if __name__ == '__main__':
    app.run(debug=True)
