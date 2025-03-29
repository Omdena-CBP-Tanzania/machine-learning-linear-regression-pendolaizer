import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Loads preprocessed data.
    """
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')
    y_test = pd.read_csv('data/y_test.csv')
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """
    Trains a linear regression model and evaluates it.
    """
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Optionally, save the model if you want to use it later
    # import joblib
    # joblib.dump(model, 'model/linear_regression_model.pkl')

    return model, mse, r2

if __name__ == "__main__":
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_data()

    # Train the model
    model, mse, r2 = train_model(X_train, X_test, y_train, y_test)

    # Optionally, you can save the model using joblib if needed
    # import joblib
    # joblib.dump(model, 'model/linear_regression_model.pkl')
