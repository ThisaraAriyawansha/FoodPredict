# Food Prediction Web Application

This project is a web application that predicts suitable food categories based on nutritional information using a trained machine learning model. The model is built with a RandomForestClassifier and deployed using Flask. The front end of the application is designed with HTML, CSS, and JavaScript.

## Features

- Predicts suitable food categories based on input nutritional information
- User-friendly interface for inputting nutritional data
- Real-time predictions displayed to the user
- Flask-based backend with a machine learning model
- Model trained using `scikit-learn`
- Frontend styled with modern design and animations

## Dataset

The dataset used for training the model should be a CSV file named `food.csv`. It should contain the following columns:

- `Data.Kilocalories`
- `Data.Fat.Total Lipid`
- `Data.Protein`
- `Data.Carbohydrate`
- `Data.Sugar Total`
- `Category` (the target variable)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/food-prediction.git
    cd food-prediction
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have `food.csv` in the root directory of the project.

4. Train the model and save the scaler and model:
    ```python
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pickle

    # Load the dataset
    df = pd.read_csv('food.csv')

    # Select relevant features and target variable
    features = df[['Data.Kilocalories', 'Data.Fat.Total Lipid', 'Data.Protein', 'Data.Carbohydrate', 'Data.Sugar Total']]
    target = df['Category']  # Assuming 'Category' is the target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the model and scaler
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print('Model and scaler saved successfully.')
    ```

## Usage

1. Start the Flask application:
    ```bash
    python app.py
    ```

2. Open your browser and go to `http://127.0.0.1:5000/`.

3. Enter the nutritional information in the form and click on "Predict" to get the suitable food category.

## Files and Directories

- **static/background.jpg**: Contains the background image used in the HTML template.
- **templates/index.html**: Contains the HTML template for the web interface.
- **app.py**: The Flask application that serves the web interface and handles predictions.
- **food.csv**: The dataset used to train the machine learning model.
- **model.pkl**: The trained RandomForest model saved as a pickle file.
- **scaler.pkl**: The scaler used to normalize the input features saved as a pickle file.
- **requirements.txt**: A file listing the required Python packages.
- **README.md**: This README file.

## Requirements

- Python 3.x
- `flask`
- `numpy`
- `pandas`
- `scikit-learn`
- `pickle-mixin`

You can install the required Python packages using the following command:
```bash
pip install -r requirements.txt
