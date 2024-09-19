from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Sample training data
conversation_data = [
    "Customer was polite and asked about a product",  # Behavior: calm
    "Customer was frustrated and angry about a service issue",  # Behavior: aggressive
    "Customer calmly requested a refund",  # Behavior: calm
    "Customer was irate about billing problems",  # Behavior: aggressive
]

behavior_labels = ["calm", "aggressive", "calm", "aggressive"]

# Text preprocessing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(conversation_data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, behavior_labels, test_size=0.2, random_state=42)

# Train a simple model (RandomForest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open("ml_models/behavior_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the vectorizer (to use later when processing text inputs)
with open("ml_models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
