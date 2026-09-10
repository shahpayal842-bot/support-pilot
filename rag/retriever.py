from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc["title"] + " " + doc["content"] for doc in documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.document_vectors = self.vectorizer.fit_transform(self.texts)

    def search(self, query, top_k=2):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.document_vectors)[0]
        ranked_indices = scores.argsort()[::-1]
        results = []
        for index in ranked_indices[:top_k]:
            results.append({
                "id": self.documents[index]["id"],
                "title": self.documents[index]["title"],
                "category": self.documents[index]["category"],
                "content": self.documents[index]["content"],
                "score": float(scores[index])
            })
        return results
