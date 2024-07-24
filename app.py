from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the model, scaler, and label encoder
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        data['Data.Kilocalories'],  
        data['Data.Fat.Total Lipid'],
        data['Data.Protein'],
        data['Data.Carbohydrate'],
        data['Data.Sugar Total']
    ]])
    
    # Scale the features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    # Convert prediction to food name
    predicted_category = label_encoder.inverse_transform(prediction)
    
    return jsonify({'suitable_food': predicted_category[0]})

if __name__ == '__main__':
    app.run(debug=True)
