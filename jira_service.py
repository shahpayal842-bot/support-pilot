import os
from typing import Any, Dict

import requests
from requests.auth import HTTPBasicAuth


class JiraService:
    def __init__(self):
        self.jira_url = os.getenv("JIRA_URL", "").rstrip("/")
        self.jira_email = os.getenv("JIRA_EMAIL", "")
        self.jira_api_token = os.getenv("JIRA_API_TOKEN", "")
        self.jira_project_key = os.getenv("JIRA_PROJECT_KEY", "IT")

    def create_ticket(self, summary: str, description: str, priority: str = "High") -> Dict[str, Any]:
        if not self.jira_url or not self.jira_email or not self.jira_api_token:
            return {"success": False, "message": "Jira configuration is incomplete"}

        payload = {
            "fields": {
                "project": {"key": self.jira_project_key},
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description}],
                        }
                    ],
                },
                "issuetype": {"name": "Task"},
                "priority": {"name": priority},
            }
        }
        try:
            response = requests.post(
                f"{self.jira_url}/rest/api/3/issue",
                auth=HTTPBasicAuth(self.jira_email, self.jira_api_token),
                json=payload,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                timeout=15,
            )
            if response.status_code in (200, 201):
                return {
                    "success": True,
                    "ticket_id": response.json().get("key"),
                    "message": "Jira ticket created successfully",
                }
            return {"success": False, "message": f"Jira request failed with status {response.status_code}"}
        except requests.RequestException as error:
            return {"success": False, "message": f"Jira request failed: {error}"}

