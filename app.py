from csv import DictReader
import json
import os
import pickle
from datetime import datetime

from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)
from rag.pipeline import run_rag_pipeline
from rag.retriever import KnowledgeRetriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = 'supersecretkey_for_supportpilot'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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


def get_dashboard_data():
    categories = {name: 0 for name in ('Hardware', 'Network', 'Password', 'Software', 'System', 'VPN')}
    priorities = {name: 0 for name in ('P1', 'P2', 'P3', 'P4')}
    try:
        with open(data_path, newline='', encoding='utf-8-sig') as data_file:
            tickets = list(DictReader(data_file))
    except (FileNotFoundError, OSError):
        tickets = []

    total_tickets = len(tickets)
    resolved_tickets = sum(row.get('ticket_status', '').strip().lower() in {'closed', 'resolved'} for row in tickets)
    pending_tickets = total_tickets - resolved_tickets
    for row in tickets:
        text = f"{row.get('ticket_subject', '')} {row.get('ticket_description', '')}"
        category = classify_category(text, '')
        if category in categories:
            categories[category] += 1

        priority = row.get('ticket_priority', '').strip().lower()
        priority_key = {
            'critical': 'P1', 'p1': 'P1', 'p1 - critical': 'P1',
            'high': 'P2', 'p2': 'P2', 'p2 - major': 'P2',
            'medium': 'P3', 'p3': 'P3', 'p3 - moderate': 'P3',
            'low': 'P4', 'p4': 'P4', 'p4 - minor': 'P4',
        }.get(priority)
        if priority_key:
            priorities[priority_key] += 1

    recent_tickets = sorted(
        tickets,
        key=lambda row: _ticket_date(row),
        reverse=True,
    )[:10]
    recent_tickets = [
        {
            'id': row.get('ticket_id', 'N/A'),
            'subject': row.get('ticket_subject', 'Untitled ticket'),
            'category': classify_category(
                f"{row.get('ticket_subject', '')} {row.get('ticket_description', '')}", ''
            ),
            'priority': _priority_code(row.get('ticket_priority', '')),
            'status': row.get('ticket_status', 'Unknown'),
        }
        for row in recent_tickets
    ]
    resolution_rate = round((resolved_tickets / total_tickets) * 100, 1) if total_tickets else 0.0
    escalation_rate = round((priorities['P1'] / total_tickets) * 100, 1) if total_tickets else 0.0
    return {
        'total_tickets': total_tickets,
        'resolution_rate': resolution_rate,
        'resolved_tickets': resolved_tickets,
        'pending_tickets': pending_tickets,
        'escalation_rate': escalation_rate,
        'avg_response_time': '3.2s',
        'categories': categories,
        'priorities': priorities,
        'recent_tickets': recent_tickets,
    }


def _ticket_date(row):
    value = row.get('first_response_time', '')
    try:
        return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError):
        return datetime.min


def _priority_code(priority):
    return {
        'critical': 'P1', 'high': 'P2', 'medium': 'P3', 'low': 'P4',
    }.get(priority.strip().lower(), priority or 'P4')


def get_metrics():
    data = get_dashboard_data()
    return {
        'total_tickets': data['total_tickets'],
        'resolution_rate': data['resolution_rate'],
        'closed_tickets': data['resolved_tickets'],
    }


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


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def dashboard():
    dashboard_data = get_dashboard_data()
    return render_template(
        'index.html',
        result=None,
        rag_output=None,
        metrics=dashboard_data,
        dashboard_data=dashboard_data,
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
        line for line in rag_output.get('resolution', '').splitlines()
        if line and line[0].isdigit()
    ]
    dashboard_data = get_dashboard_data()
    return render_template(
        'index.html',
        result=result,
        rag_output=rag_output,
        submitted_data={'title': title, 'description': description},
        metrics=dashboard_data,
        dashboard_data=dashboard_data,
        user=current_user,
    )


if __name__ == '__main__':
    app.run(debug=True)
