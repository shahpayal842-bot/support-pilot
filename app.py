from csv import DictReader
import json
import os
import pickle
import sqlite3

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)

from rag.pipeline import run_rag_pipeline
from rag.retriever import KnowledgeRetriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = 'supersecretkey_for_supportpilot'
app.config['JWT_SECRET_KEY'] = 'supportpilot_secure_key'

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

DB_PATH = os.path.join(BASE_DIR, 'users.db')


def init_users_db():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            '''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
            '''
        )
        connection.commit()


init_users_db()


class User(UserMixin):
    def __init__(self, id, email, name):
        self.id = id
        self.email = email
        self.name = name


test_user = User(id=1, email='admin@supportpilot.com', name='Admin User')


@login_manager.user_loader
def load_user(user_id):
    if int(user_id) == test_user.id:
        return test_user
    return None


model_path = os.path.join(BASE_DIR, 'models', 'ticket_classifier.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
data_path = os.path.join(BASE_DIR, 'data', 'cleaned_customer_support_tickets.csv')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

knowledge_base_path = os.path.join(BASE_DIR, 'data', 'knowledge_base.json')
with open(knowledge_base_path, encoding='utf-8') as knowledge_base_file:
    knowledge_base = json.load(knowledge_base_file)
retriever = KnowledgeRetriever(knowledge_base)


def _normalize_priority(raw_priority):
    value = str(raw_priority).strip().lower()
    mapping = {
        'critical': 'P1',
        'p1': 'P1',
        'p1 - critical': 'P1',
        'high': 'P2',
        'p2': 'P2',
        'p2 - major': 'P2',
        'medium': 'P3',
        'p3': 'P3',
        'p3 - moderate': 'P3',
        'low': 'P4',
        'p4': 'P4',
        'p4 - minor': 'P4',
    }
    return mapping.get(value, 'P4')


def classify_category(text, predicted_category):
    text = text.lower()
    category_keywords = {
        'Network': ('wifi', 'internet', 'network', 'ethernet', 'connection'),
        'VPN': ('vpn', 'virtual private network'),
        'Password': ('password', 'login', 'sign in', 'credential', 'locked out'),
        'Hardware': ('hardware', 'keyboard', 'screen', 'monitor', 'battery', 'laptop', 'printer', 'mouse'),
        'Software': ('install', 'application', 'software', 'microsoft office', 'program'),
        'System': ('blue screen', 'operating system', 'boot', 'crash', 'server', 'update'),
    }
    for category, keywords in category_keywords.items():
        if any(keyword in text for keyword in keywords):
            return category
    return predicted_category if predicted_category in category_keywords else 'General Support'


def analyze_ticket(text):
    normalized_text = text.lower()
    model_input = vectorizer.transform([text])
    predicted_category = model.predict(model_input)[0]
    category = classify_category(text, predicted_category)

    urgent_terms = ('urgent', 'asap', 'emergency', 'critical', 'immediately', 'outage')
    failure_terms = ('down', 'crash', 'not working', 'cannot connect', 'unable', 'failed', 'failure', 'breach')
    negative_terms = ('angry', 'frustrated', 'unacceptable', 'lost', 'broken', 'blocked', 'problem')
    data_risk_terms = ('security breach', 'data loss', 'stolen', 'leak', 'ransomware')

    urgency = sum(term in normalized_text for term in urgent_terms)
    failures = sum(term in normalized_text for term in failure_terms)
    negative_sentiment = sum(term in normalized_text for term in negative_terms)
    data_risk = any(term in normalized_text for term in data_risk_terms)

    if data_risk or (('server' in normalized_text or 'production' in normalized_text) and failures) or urgency >= 2:
        severity, priority = 'Critical', 'P1 - Critical'
    elif failures >= 1 or urgency == 1 or negative_sentiment >= 2:
        severity, priority = 'High', 'P2 - Major'
    elif any(term in normalized_text for term in ('slow', 'error', 'issue', 'delay', 'difficult')) or negative_sentiment == 1:
        severity, priority = 'Medium', 'P3 - Moderate'
    else:
        severity, priority = 'Low', 'P4 - Minor'

    confidence = float(max(model.predict_proba(model_input)[0])) if hasattr(model, 'predict_proba') else 0.0
    suggestions = {
        'Network': 'Check cable or Wi-Fi status, restart the network adapter, and test another connection.',
        'VPN': 'Confirm internet access, reconnect the VPN client, and verify the VPN account is active.',
        'Password': 'Use the password reset flow, then confirm the account is not locked by repeated attempts.',
        'Hardware': 'Restart the device, check physical connections, and test the component on another port.',
        'Software': 'Restart the application, check for updates, and reinstall it if the issue continues.',
        'System': 'Save work, restart the device, and check recent system updates or crash messages.',
    }
    return {
        'category': category,
        'severity': severity,
        'priority': priority,
        'confidence': f'{confidence * 100:.1f}%',
        'suggestion': suggestions.get(category, 'Collect the exact error message and reproduce the issue after a restart.'),
        'status': 'Open / AI Classified',
    }


def load_dashboard_data(current_ticket=None):
    category_labels = ['Hardware', 'Network', 'Password', 'Software', 'System', 'VPN']
    priority_labels = ['P1', 'P2', 'P3', 'P4']

    try:
        with open(data_path, newline='', encoding='utf-8-sig') as csv_file:
            tickets = list(DictReader(csv_file))
    except (FileNotFoundError, OSError):
        tickets = []

    categories = {label: 0 for label in category_labels}
    priorities = {label: 0 for label in priority_labels}

    for row in tickets:
        subject = str(row.get('ticket_subject', ''))
        description = str(row.get('ticket_description', ''))
        text = f'{subject} {description}'
        category = classify_category(text, '')
        if category in categories:
            categories[category] += 1

        priority = _normalize_priority(row.get('ticket_priority', ''))
        if priority in priorities:
            priorities[priority] += 1

    total_tickets = len(tickets)
    resolved_tickets = sum(
        str(row.get('ticket_status', '')).strip().lower() in {'closed', 'resolved'}
        for row in tickets
    )
    resolution_rate = round((resolved_tickets / total_tickets) * 100, 1) if total_tickets else 0.0
    escalation_rate = round((priorities['P1'] / total_tickets) * 100, 1) if total_tickets else 0.0

    recent_tickets = []
    for row in sorted(tickets, key=lambda item: str(item.get('first_response_time', '')), reverse=True)[:5]:
        subject = str(row.get('ticket_subject', 'Untitled ticket'))
        description = str(row.get('ticket_description', ''))
        category = classify_category(f'{subject} {description}', '')
        recent_tickets.append({
            'id': row.get('ticket_id', 'N/A'),
            'subject': subject,
            'category': category,
            'priority': _normalize_priority(row.get('ticket_priority', '')),
            'status': str(row.get('ticket_status', 'Unknown')).title(),
        })

    if current_ticket:
        recent_tickets.insert(0, current_ticket)
        recent_tickets = recent_tickets[:5]

    return {
        'total_tickets': total_tickets,
        'resolution_rate': resolution_rate,
        'resolved_tickets': resolved_tickets,
        'pending_tickets': total_tickets - resolved_tickets,
        'escalation_rate': escalation_rate,
        'avg_response_time': '3.2s',
        'category_distribution': categories,
        'priority_distribution': priorities,
        'categories': categories,
        'priorities': priorities,
        'recent_tickets': recent_tickets,
    }


def get_metrics():
    data = load_dashboard_data()
    return {
        'total_tickets': data['total_tickets'],
        'resolution_rate': data['resolution_rate'],
        'resolved_tickets': data['resolved_tickets'],
        'category_distribution': data['category_distribution'],
        'priority_distribution': data['priority_distribution'],
        'recent_tickets': data['recent_tickets'],
    }


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        if request.form.get('email') == test_user.email and request.form.get('password') == 'admin123':
            login_user(test_user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials. Try admin@supportpilot.com / admin123')

    return render_template('login.html')


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get('username', '')).strip()
    password = str(data.get('password', ''))

    if not username or not password:
        return jsonify({'msg': 'Username and password are required.'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.execute(
                'INSERT INTO users (username, password) VALUES (?, ?)',
                (username, hashed_password),
            )
            connection.commit()
    except sqlite3.IntegrityError:
        return jsonify({'msg': 'Username already exists.'}), 409

    return jsonify({'msg': 'User registered successfully.', 'username': username}), 201


@app.route('/login-jwt', methods=['POST'])
def login_jwt():
    data = request.get_json(silent=True) or {}
    username = str(data.get('username', '')).strip()
    password = str(data.get('password', ''))

    if not username or not password:
        return jsonify({'msg': 'Username and password are required.'}), 400

    with sqlite3.connect(DB_PATH) as connection:
        user = connection.execute(
            'SELECT username, password FROM users WHERE username = ?', (username,)
        ).fetchone()

    if user and bcrypt.check_password_hash(user[1], password):
        access_token = create_access_token(identity=username)
        return jsonify({'access_token': access_token, 'username': username}), 200

    return jsonify({'msg': 'Invalid username or password.'}), 401


@app.route('/api/profile', methods=['GET'])
@jwt_required()
def api_profile():
    current_username = get_jwt_identity()
    return jsonify({'username': current_username}), 200


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def dashboard():
    dashboard_data = load_dashboard_data()
    return render_template(
        'index.html',
        result=None,
        rag_output=None,
        metrics=get_metrics(),
        dashboard_data=dashboard_data,
        submitted_data={},
        user=current_user,
    )


@app.route('/submit-ticket', methods=['POST'])
@login_required
def submit_ticket():
    title = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    query_text = f'{title} {description}'.strip()
    result = analyze_ticket(query_text)

    ticket = {
        'id': 'T-2023-4521',
        'title': title,
        'description': description,
        'category': result['category'],
        'priority': result['priority'],
    }

    rag_output = run_rag_pipeline(ticket, retriever)
    rag_output['resolution_steps'] = [
        line.strip()
        for line in rag_output.get('resolution', '').splitlines()
        if line and line[0].isdigit()
    ]

    dashboard_data = load_dashboard_data(current_ticket={
        'id': ticket['id'],
        'subject': title or 'Untitled ticket',
        'category': result['category'],
        'priority': _normalize_priority(result['priority']),
        'status': 'Open / AI Classified',
    })
    return render_template(
        'index.html',
        result=result,
        rag_output=rag_output,
        submitted_data={'title': title, 'description': description},
        metrics=get_metrics(),
        dashboard_data=dashboard_data,
        user=current_user,
    )


if __name__ == '__main__':
    app.run(debug=True)
