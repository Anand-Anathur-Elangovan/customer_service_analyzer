import pickle

# Load the behavior prediction model and vectorizer
with open("ml_models/behavior_model.pkl", "rb") as model_file:
    behavior_model = pickle.load(model_file)

with open("ml_models/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_behavior(conversation_text):
    # Preprocess the conversation using the same vectorizer
    conversation_vector = vectorizer.transform([conversation_text])

    # Predict the behavior based on the model
    predicted_behavior = behavior_model.predict(conversation_vector)
    
    return predicted_behavior[0]
