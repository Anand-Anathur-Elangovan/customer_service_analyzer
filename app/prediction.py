import pickle
import numpy as np

# Load pretrained models
with open('ml_models/satisfaction_model.pkl', 'rb') as f:
    satisfaction_model = pickle.load(f)
with open('ml_models/behavior_model.pkl', 'rb') as f:
    behavior_model = pickle.load(f)

def predict_satisfaction(features):
    return satisfaction_model.predict([features])

def predict_behavior(text):
    return behavior_model.predict([text])
