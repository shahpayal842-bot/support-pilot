import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict


class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_email = os.getenv("SMTP_EMAIL", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")

    def send_email(self, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        placeholder_values = {
            "support@example.com",
            "your-app-password",
            "your-email@example.com",
        }
        if not self.smtp_email or not self.smtp_password or self.smtp_email in placeholder_values or self.smtp_password in placeholder_values:
            return {
                "success": False,
                "message": "SMTP is not configured. Set SMTP_EMAIL and SMTP_PASSWORD in .env using a Gmail app password.",
            }
        message = MIMEMultipart()
        message["From"] = self.smtp_email
        message["To"] = recipient
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=15) as server:
                server.starttls()
                server.login(self.smtp_email, self.smtp_password)
                server.sendmail(self.smtp_email, recipient, message.as_string())
            return {"success": True, "message": "Email sent successfully"}
        except (OSError, smtplib.SMTPException) as error:
            return {"success": False, "message": f"Email could not be sent: {error}"}

