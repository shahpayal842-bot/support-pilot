from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load trained model and vectorizer
model_path = "models/ticket_classifier.pkl"
vectorizer_path = "models/vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_severity(text):
    text = text.lower()
    if any(w in text for w in ["server down", "production down", "security breach"]):
        return "Critical"
    elif any(w in text for w in ["urgent", "cannot work", "client meeting", "vpn not working"]):
        return "High"
    elif any(w in text for w in ["slow", "error", "issue"]):
        return "Medium"
    return "Low"

def calculate_priority(severity):
    if severity == "Critical":
        return "P1"
    elif severity == "High":
        return "P2"
    elif severity == "Medium":
        return "P3"
    return "P4"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/submit-ticket", methods=["POST"])
def submit_ticket():
    data = request.form.to_dict()
    description = data.get("description", "")
    
    # AI Classification & Severity Logic
    ticket_vector = vectorizer.transform([description])
    category = model.predict(ticket_vector)[0]
    severity = predict_severity(description)
    priority = calculate_priority(severity)
    
    result = {
        "category": category,
        "severity": severity,
        "priority": priority,
        "status": "Open"
    }
    
    return render_template("index.html", result=result, submitted_data=data)

if __name__ == "__main__":
    app.run(debug=True)