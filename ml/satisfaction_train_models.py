# Import necessary libraries
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Adjust the parameters, increasing `n_informative`
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple RandomForestClassifier (you can replace it with any other model)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model as a .pkl file
with open('ml_models/satisfaction_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model saved to 'ml_models/satisfaction_model.pkl'")
