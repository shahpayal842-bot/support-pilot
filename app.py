from csv import DictReader
import os
import pickle

from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)

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
data_path = os.path.join(BASE_DIR, 'cleaned_customer_support_tickets.csv')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


def get_metrics():
    try:
        with open(data_path, newline='', encoding='utf-8-sig') as data_file:
            tickets = list(DictReader(data_file))
    except (FileNotFoundError, OSError):
        return {'total_tickets': 0, 'resolution_rate': 0.0, 'closed_tickets': 0}

    total_tickets = len(tickets)
    closed_tickets = sum(
        row.get('ticket_status', '').strip().lower() == 'closed'
        for row in tickets
    )
    resolution_rate = round((closed_tickets / total_tickets) * 100, 1) if total_tickets else 0.0
    return {
        'total_tickets': total_tickets,
        'resolution_rate': resolution_rate,
        'closed_tickets': closed_tickets,
    }


def classify_category(text, predicted_category):
    text = text.lower()
    category_keywords = {
        'Network': ('wifi', 'internet', 'network', 'ethernet', 'connection'),
        'VPN': ('vpn', 'virtual private network'),
        'Password': ('password', 'login', 'sign in', 'credential', 'locked out'),
        'Hardware': ('keyboard', 'screen', 'monitor', 'battery', 'laptop', 'printer', 'mouse'),
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
    return render_template('index.html', result=None, metrics=get_metrics(), user=current_user)


@app.route('/submit-ticket', methods=['POST'])
@login_required
def submit_ticket():
    title = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    query_text = f'{title} {description}'.strip()
    result = analyze_ticket(query_text)
    return render_template('index.html', result=result, metrics=get_metrics(), user=current_user)


if __name__ == '__main__':
    app.run(debug=True)
