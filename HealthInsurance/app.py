from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

# Load pre-trained model (or train directly in the script as shown)
df = pd.read_csv('../datasets/Health_insurance.csv')
df['sex'] = df['sex'].apply(lambda A: 1 if A == 'female' else 0)
df['smoker'] = df['smoker'].apply(lambda A: 1 if A == 'yes' else 0)
X = df[['age', 'sex', 'bmi', 'children', 'smoker']]
y = df['charges']
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    age = int(data['age'])
    sex = int(data['sex'])
    bmi = float(data['bmi'])
    children = int(data['children'])
    smoker = int(data['smoker'])
    prediction = model.predict([[age, sex, bmi, children, smoker]])
    return jsonify({'prediction': f'{prediction[0]:.2f}'})

if __name__ == '__main__':
    app.run(debug=True)
