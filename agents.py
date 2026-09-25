import json
import os
import re
from typing import Any, Dict, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class DiagnosisAgent:
    """Uses the Milestone 1 classifier and severity rules to diagnose a ticket."""

    def __init__(self):
        self.model = joblib.load(os.path.join(BASE_DIR, "models", "ticket_classifier.pkl"))
        self.vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

    @staticmethod
    def _severity(text: str) -> str:
        normalized = text.lower()
        urgent_terms = ("urgent", "asap", "emergency", "critical", "immediately", "outage")
        failure_terms = ("down", "crash", "not working", "cannot connect", "unable", "failed", "failure", "breach")
        negative_terms = ("angry", "frustrated", "unacceptable", "lost", "broken", "blocked", "problem")
        data_risk_terms = ("security breach", "data loss", "stolen", "leak", "ransomware")
        urgency = sum(term in normalized for term in urgent_terms)
        failures = sum(term in normalized for term in failure_terms)
        negative_sentiment = sum(term in normalized for term in negative_terms)
        data_risk = any(term in normalized for term in data_risk_terms)
        if data_risk or (("server" in normalized or "production" in normalized) and failures) or urgency >= 2:
            return "Critical"
        if failures >= 1 or urgency == 1 or negative_sentiment >= 2:
            return "High"
        if any(term in normalized for term in ("slow", "error", "issue", "delay", "difficult")) or negative_sentiment == 1:
            return "Medium"
        return "Low"

    @staticmethod
    def _priority(severity: str) -> str:
        return {"Critical": "P1", "High": "P2", "Medium": "P3", "Low": "P4"}[severity]

    def analyze(self, ticket: str) -> Dict[str, Any]:
        text = str(ticket or "").strip()
        model_input = self.vectorizer.transform([text])
        category = str(self.model.predict(model_input)[0])
        confidence = float(max(self.model.predict_proba(model_input)[0])) if hasattr(self.model, "predict_proba") else 0.0
        severity = self._severity(text)
        return {
            "category": category,
            "severity": severity,
            "priority": self._priority(severity),
            "summary": f"Milestone 1 ML classification identified this as a {category} ticket.",
            "confidence": confidence,
        }


class RetrievalAgent:
    """Implements the Milestone 2 TF-IDF knowledge-base retriever."""

    def __init__(self, knowledge_base_path: str = None):
        path = knowledge_base_path or os.path.join(BASE_DIR, "data", "knowledge_base.json")
        with open(path, encoding="utf-8") as knowledge_file:
            self.documents: List[Dict[str, Any]] = json.load(knowledge_file)
        self.texts = [f"{doc['title']} {doc['content']}" for doc in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.document_vectors = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 2) -> Tuple[Dict[str, Any], float]:
        query_vector = self.vectorizer.transform([str(query or "")])
        scores = cosine_similarity(query_vector, self.document_vectors)[0]
        ranked_indices = scores.argsort()[::-1]
        results = []
        for index in ranked_indices[:top_k]:
            document = self.documents[index]
            results.append({
                "id": document["id"],
                "title": document["title"],
                "category": document.get("category", "IT Support"),
                "content": document["content"],
                "score": float(scores[index]),
            })
        best = results[0]
        return {"best_match": best, "results": results}, best["score"]


class ResolutionAgent:
    """Uses the Milestone 2 numbered RAG content to build resolution steps."""

    def generate(self, diagnosis: Dict[str, Any], retrieval: Dict[str, Any]) -> Dict[str, Any]:
        best_match = retrieval["best_match"]
        if best_match["score"] < 0.20:
            steps = ["No sufficiently relevant knowledge-base article was found. Escalate for human review."]
        else:
            steps = []
            for line in best_match["content"].splitlines():
                match = re.match(r"^\s*\d+\.\s*(.+)$", line)
                if match:
                    steps.append(f"{match.group(1).strip()} (Source: {best_match['id']} - {best_match['title']})")
            if not steps:
                steps = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", best_match["content"]) if sentence.strip()]
            steps.append("If the issue persists, provide the exact error and escalate with the attempted steps.")
        return {
            "intro": f"Recommended resolution for your {diagnosis['category']} ticket:",
            "steps": steps,
            "context": retrieval["results"],
        }


class ValidationAgent:
    def validate(self, diagnosis_confidence: float, retrieval_similarity: float, number_of_steps: int) -> Dict[str, Any]:
        confidence = round((diagnosis_confidence * 0.40 + retrieval_similarity * 0.40 + min(number_of_steps / 6, 1) * 0.20) * 100, 2)
        return {"confidence": confidence, "status": "AUTO_RESOLVE" if confidence >= 70 else "ESCALATE"}


class EscalationAgent:
    def should_escalate(self, validation: Dict[str, Any]) -> bool:
        return validation.get("status") == "ESCALATE"


class SupportPilot:
    def __init__(self):
        self.diagnosis_agent = DiagnosisAgent()
        self.retrieval_agent = RetrievalAgent()
        self.resolution_agent = ResolutionAgent()
        self.validation_agent = ValidationAgent()
        self.escalation_agent = EscalationAgent()

    def process_ticket(self, ticket: str) -> Dict[str, Any]:
        diagnosis = self.diagnosis_agent.analyze(ticket)
        retrieval, similarity = self.retrieval_agent.search(ticket)
        resolution = self.resolution_agent.generate(diagnosis, retrieval)
        validation = self.validation_agent.validate(diagnosis["confidence"], similarity, len(resolution["steps"]))
        return {
            "ticket": ticket,
            "diagnosis": diagnosis,
            "retrieval": {**retrieval, "similarity": similarity},
            "resolution": resolution,
            "validation": validation,
            "escalation": self.escalation_agent.should_escalate(validation),
        }
