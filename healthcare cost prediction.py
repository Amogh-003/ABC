import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge
from sklearn.compose import ColumnTransformer

# Sample dataset creation for demonstration purposes
data = {
    'Age': np.random.randint(18, 80, 1000),
    'Hospital': np.random.choice(['Hospital A', 'Hospital B', 'Hospital C'], 1000),
    'Treatment': np.random.choice(['Treatment X', 'Treatment Y', 'Treatment Z'], 1000),
    'Days': np.random.randint(1, 15, 1000),
    'Cost': np.random.randint(500, 10000, 1000)  # Randomized for example
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Feature and target split
X = df[['Age', 'Hospital', 'Treatment', 'Days']]
y = df['Cost']

# Preprocessing pipeline for categorical and numerical features
numeric_features = ['Age', 'Days']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['Hospital', 'Treatment']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', BayesianRidge())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# User input simulation
def predict_cost():
    print("Welcome to MY Healthcare Cost Prediction System")
    name = input("Enter your name: ")
    age = int(input("Enter your age: "))
    hospital = input("Choose a hospital (Hospital A, Hospital B, Hospital C): ")
    treatment = input("Choose a treatment (Treatment X, Treatment Y, Treatment Z): ")
    days = int(input("Enter the number of days of hospital stay: "))

    # Prepare the input data
    user_data = pd.DataFrame({
        'Age': [age],
        'Hospital': [hospital],
        'Treatment': [treatment],
        'Days': [days]
    })

    # Predict the cost
    predicted_cost = model.predict(user_data)[0]

    print(f"\n{name}, based on the provided information, the predicted healthcare cost is: ${predicted_cost:.2f}")

# Run the prediction function
if __name__ == "__main__":
    predict_cost()
