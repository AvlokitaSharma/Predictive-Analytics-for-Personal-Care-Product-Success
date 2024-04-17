import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_data(filepath):
    return pd.read_csv(filepath)

# Data Cleaning
def clean_data(data):
    data = data.dropna()
    data = data[data['Sales'] > 0]  # Removing non-sensible sales records
    return data

# Feature Engineering
def engineer_features(data):
    data['Price_Ratio'] = data['Competitor_Price'] / data['Price']
    data['Season'] = data['Month'].apply(lambda x: 'High' if x in [11, 12, 1, 2] else 'Low')
    return data

# Data Visualization
def plot_data(data):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Product_Type', y='Sales', data=data)
    plt.title('Sales by Product Type')
    plt.show()

# Model Training
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Predictions
def make_predictions(model, X):
    return model.predict(X)

# Main execution function
def main():
    data = load_data('path_to_your_data.csv')
    data = clean_data(data)
    data = engineer_features(data)
    
    plot_data(data)

    # Prepare data for modeling
    feature_cols = ['Price', 'Price_Ratio', 'Marketing_Spend', 'Is_Holiday']
    X = data[feature_cols]
    y = data['Sales']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Make predictions
    predictions = make_predictions(model, X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    main()
