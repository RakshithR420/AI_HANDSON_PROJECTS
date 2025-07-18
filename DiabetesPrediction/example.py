import pickle
import numpy as np
from model import NeuralNetwork
print("Script started")

with open('nn_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded")

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Scaler loaded")

sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
sample_scaled = scaler.transform(sample)
print("Sample scaled")

pred, prob = model.predict(sample_scaled)
print("Prediction:", pred)
print("Probability:", prob)
