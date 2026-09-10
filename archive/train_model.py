import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# Sample training data for IT support tickets
data = {
    "ticket": [
        "WiFi is not working", "Internet connection is very slow",
        "VPN is not connecting", "Unable to access company VPN",

        "I forgot my password", "Please reset my password",
        "Install Microsoft Office", "Application installation required",
        "Laptop keyboard is not working", "Monitor display is not working",
        "Blue screen error on system startup", "Operating system crashes frequently"
    ],
    "category": [
        "Network", "Network", "VPN", "VPN", 
        "Password", "Password", "Software", "Software", 
        "Hardware", "Hardware", "System", "System"
    ]
}

df = pd.DataFrame(data)
X = df["ticket"]
y = df["category"]

# Vectorization & Model Training
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)

# Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/ticket_classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model trained and saved successfully in 'models/' folder!")