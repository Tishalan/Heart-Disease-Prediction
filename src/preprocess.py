import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """Handle missing values (if any)."""
    return df.dropna()

def handle_outliers(df):
    """Replace 0 cholesterol values with the median of non-zero values."""
    if 'Cholesterol' in df.columns:
        median_cholesterol = df[df['Cholesterol'] > 0]['Cholesterol'].median()
        df['Cholesterol'] = df['Cholesterol'].replace(0, median_cholesterol)
    return df

def encode_categorical(df, columns):
    """One-hot encode categorical columns."""
    return pd.get_dummies(df, columns=columns, drop_first=True)

def split_features_target(df, target_col):
    """Separate features and target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def scale_features(X_train, X_test, numeric_columns):
    """Scale numeric features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    return X_train_scaled, X_test_scaled, scaler
