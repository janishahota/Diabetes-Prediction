from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Load the trained model and scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    data = {
        'gender': [int(request.form['gender'])],
        'age': [float(request.form['age'])],
        'hypertension': [int(request.form['hyper'])],
        'heart_disease': [int(request.form['hd'])],
        'smoking_history': [int(request.form['sm'])],
        'bmi': [float(request.form['bmi'])],
        'HbA1c_level': [float(request.form['hb'])],
        'blood_glucose_level': [int(request.form['bgl'])]
    }
    dataframe = pd.DataFrame(data)

    # Scale the features
    scaled_data = scaler.transform(dataframe)

    # Make a prediction
    prediction = model.predict(scaled_data)[0]
    message = 'has diabetes / मधुमेह है' if prediction == 1 else 'does not have diabetes / मधुमेह नहीं है'

    return render_template('index.html', prediction=prediction, message=f'Your patient {request.form["name"]} {message}')

if __name__ == '__main__':
    app.run(debug=True)