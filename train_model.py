import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle
import os

def load_and_prepare_data():
    if os.path.exists('data/ai4i2020.csv'):
        print("Dataset already exists, loading from disk...")
        df = pd.read_csv('data/ai4i2020.csv')
    else:
        print("Fetching dataset for first time...")
        ai4i = fetch_ucirepo(id=601)
        X = ai4i.data.features
        y = ai4i.data.targets
        df = pd.concat([X, y], axis=1)
        df.to_csv('data/ai4i2020.csv', index=False)
        print(f"Dataset saved. Shape: {df.shape}")
    return df

def train_model(df):
    df = df.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')
    
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])
    
    X = df.drop(columns=['Machine failure'])
    y = df['Machine failure']
    
    print(f"Features: {list(X.columns)}")
    print(f"Failure rate: {y.mean():.2%}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=10,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    with open('data/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('data/encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print("Model saved to data/model.pkl")
    return model, le

if __name__ == "__main__":
    df = load_and_prepare_data()
    model, le = train_model(df)