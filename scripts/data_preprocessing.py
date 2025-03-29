import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Loads the dataset from the given file path.
    """
    return pd.read_csv('BostonHousing.csv')

def remove_outliers_iqr(data, columns, threshold=1.5):
    """
    Removes outliers from numerical columns using the Interquartile Range (IQR) method.
    """
    df_clean = data.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def encode_categorical_features(df, categorical_columns):
    """
    Encodes categorical columns using OneHotEncoder.
    """
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    
    for col in categorical_columns:
        # Apply OneHotEncoder to the categorical column
        encoded_cols = encoder.fit_transform(df[[col]])
        
        # Create DataFrame for encoded columns
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out([col]))
        
        # Drop the original column and add encoded columns to the DataFrame
        df = df.drop(columns=[col]).join(encoded_df)
        
    return df

def preprocess_data(df):
    """
    Preprocesses the data by removing outliers, standardizing features, 
    and encoding categorical columns (if any).
    """
    # Identify numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    numerical_columns = numerical_columns.drop("medv")  # Exclude target variable
    
    # Remove outliers
    df_clean = remove_outliers_iqr(df, numerical_columns)
    
    # Standardize numerical features
    scaler = StandardScaler()
    df_clean[numerical_columns] = scaler.fit_transform(df_clean[numerical_columns])
    
    return df_clean

def split_data(df, target_variable="medv"):
    """
    Splits the data into training and testing sets.
    """
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Load dataset
    file_path = "BostonHousing.csv"  # Replace with your actual dataset path
    df = load_data(file_path)

    # Optional: Encode categorical columns if they exist
    # Example: df = encode_categorical_features(df, categorical_columns=['chas'])
    
    # Preprocess data
    df_clean = preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df_clean)

    # Save processed data to CSV files
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("Preprocessing completed and data saved!")
