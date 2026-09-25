import csv
import json
import os
import sqlite3
import uuid
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required

from agents import SupportPilot
from database import list_processed_tickets, save_processed_ticket
from email_service import EmailService
from jira_service import JiraService

load_dotenv(override=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "supportpilot-development-secret-key-change-me")
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
support_pilot = SupportPilot()
jira_service = JiraService()
email_service = EmailService()
USERS_DB = os.path.join(BASE_DIR, "users.db")
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_customer_support_tickets.csv")


def init_users_db():
    with sqlite3.connect(USERS_DB) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL)")
        connection.commit()


def load_dashboard_metrics():
    try:
        with open(DATA_PATH, newline="", encoding="utf-8-sig") as data_file:
            rows = list(csv.DictReader(data_file))
    except (OSError, csv.Error):
        rows = []
    total = len(rows)
    resolved = sum(str(row.get("ticket_status", "")).lower() in {"closed", "resolved"} for row in rows)
    response_seconds = []
    covered_tickets = 0
    for row in rows:
        ticket_text = f"{row.get('ticket_subject', '')} {row.get('ticket_description', '')}".strip()
        if ticket_text and support_pilot.retrieval_agent.search(ticket_text, top_k=1)[1] >= 0.20:
            covered_tickets += 1
        first_response = str(row.get("first_response_time", "")).strip()
        resolution_time = str(row.get("time_to_resolution", "")).strip()
        if first_response and resolution_time:
            try:
                elapsed = (datetime.fromisoformat(resolution_time) - datetime.fromisoformat(first_response)).total_seconds()
                if elapsed >= 0:
                    response_seconds.append(elapsed)
            except ValueError:
                pass
    average_response = round(sum(response_seconds) / len(response_seconds), 1) if response_seconds else 0.0
    category_distribution = {}
    priority_distribution = {"P1": 0, "P2": 0, "P3": 0, "P4": 0}
    for row in rows:
        text = f"{row.get('ticket_subject', '')} {row.get('ticket_description', '')}".strip()
        if text:
            category = str(support_pilot.diagnosis_agent.model.predict(
                support_pilot.diagnosis_agent.vectorizer.transform([text])
            )[0])
            category_distribution[category] = category_distribution.get(category, 0) + 1
        priority = str(row.get("ticket_priority", "")).strip().upper()
        priority_code = priority.split(" ", 1)[0]
        if priority_code in priority_distribution:
            priority_distribution[priority_code] += 1
    return {
        "total_tickets": total,
        "resolved_tickets": resolved,
        "resolution_rate": round(resolved / total * 100, 1) if total else 0.0,
        "retrieval_accuracy": round(covered_tickets / total * 100, 1) if total else 0.0,
        "avg_response_time": f"{average_response}s",
        "category_distribution": category_distribution,
        "priority_distribution": priority_distribution,
    }


init_users_db()


@app.route("/")
def dashboard():
    return render_template("dashboard.html", metrics=load_dashboard_metrics())


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/health")
def health():
    return jsonify({"status": "running", "service": "SupportPilot AI"})


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    if not username or not password:
        return jsonify({"msg": "Username and password are required."}), 400
    try:
        with sqlite3.connect(USERS_DB) as connection:
            password_hash = bcrypt.generate_password_hash(password).decode("utf-8")
            connection.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password_hash))
            connection.commit()
    except sqlite3.IntegrityError:
        return jsonify({"msg": "Username already exists."}), 409
    return jsonify({"msg": "User registered successfully", "username": username}), 201


@app.route("/login-jwt", methods=["POST"])
def login_jwt():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    with sqlite3.connect(USERS_DB) as connection:
        user = connection.execute("SELECT username, password FROM users WHERE username = ?", (username,)).fetchone()
    if user and bcrypt.check_password_hash(user[1], password):
        return jsonify({"access_token": create_access_token(identity=username), "username": username})
    return jsonify({"msg": "Invalid username or password."}), 401


@app.route("/api/profile")
@jwt_required()
def profile():
    return jsonify({"username": get_jwt_identity()})


@app.route("/api/tickets")
@jwt_required()
def ticket_history():
    return jsonify({"tickets": list_processed_tickets()})


@app.route("/api/ticket", methods=["POST"])
@jwt_required()
def process_ticket():
    data = request.get_json(silent=True) or {}
    ticket = str(data.get("ticket", "")).strip()
    customer_name = str(data.get("customer_name", "")).strip()
    department = str(data.get("department", "")).strip()
    email = str(data.get("email", "")).strip()
    if not ticket:
        return jsonify({"error": "ticket is required"}), 400
    result = support_pilot.process_ticket(ticket)
    status = result["validation"]["status"]
    confidence = result["validation"]["confidence"]
    ticket_id = f"SP-{uuid.uuid4().hex[:8].upper()}"
    escalation_body = json.dumps({"diagnosis": result["diagnosis"], "attempted_resolution": result["resolution"]["steps"], "confidence": confidence}, indent=2)
    if status == "AUTO_RESOLVE":
        email_subject = "SupportPilot resolution"
        email_body = json.dumps(result["resolution"], indent=2)
        transaction = {"email": email_service.send_email(email, email_subject, email_body) if email else {"success": False, "message": "No recipient email provided"}}
    else:
        transaction = {
            "jira": jira_service.create_ticket(
                f"SupportPilot escalation: {result['diagnosis']['category']}",
                escalation_body,
            ),
        }
        if email:
            transaction["email"] = email_service.send_email(
                email,
                "SupportPilot escalation received",
                "Your support request was escalated to a human agent.\n\n" + escalation_body,
            )
        else:
            transaction["email"] = {"success": False, "message": "No recipient email provided"}
    save_processed_ticket(ticket_id, customer_name or None, department or None, ticket, email or None, result["diagnosis"]["category"], result["diagnosis"]["priority"], confidence, status)
    return jsonify({"status": status, "confidence": confidence, "result": result, **transaction})


@app.route("/api/chat", methods=["POST"])
@jwt_required()
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    result = support_pilot.process_ticket(message)
    response = result["resolution"]
    return jsonify({
        "message": message,
        "answer": response["intro"],
        "steps": response["steps"],
        "category": result["diagnosis"]["category"],
        "confidence": result["validation"]["confidence"],
        "status": result["validation"]["status"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
