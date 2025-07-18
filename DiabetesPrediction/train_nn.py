# train_nn.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import NeuralNetwork

def load_and_preprocess_data():
    dataset_path = ('data/diabetes.csv')
    df = pd.read_csv(dataset_path)

    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    nn = NeuralNetwork(input_layer=8, hidden_layer=8, output_layer=1)
    nn.train(X_train, y_train, learning_rate=0.03, epochs=5000)

    with open('nn_model.pkl', 'wb') as f:
        pickle.dump(nn, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Model and scaler saved!")
