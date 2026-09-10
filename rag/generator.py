def build_context(results):
    context = ""
    for doc in results:
        context += f"SOURCE: {doc['id']}\nTITLE: {doc['title']}\nRELEVANCE: {doc['score']:.2f}\n{doc['content']}\n=============================\n"
    return context


def generate_resolution(ticket, retrieved_docs):
    MIN_RELEVANCE = 0.20
    valid_docs = [doc for doc in retrieved_docs if doc["score"] >= MIN_RELEVANCE]
    if not valid_docs:
        return "No sufficiently relevant knowledge-base articles were found.\nStatus: Ticket escalated to human support team."

    resolution_steps = ["Recommended Resolution:"]
    step_number = 1
    for doc in valid_docs:
        lines = doc["content"].split("\n")
        for line in lines:
            line = line.strip()
            if any(line.startswith(f"{i}.") for i in range(1, 7)):
                cleaned = line.split(".", 1)[1].strip()
                resolution_steps.append(f"{step_number}. {cleaned} (Source: {doc['id']} – {doc['title']})")
                step_number += 1

    resolution_steps.append(f"{step_number}. If the problem persists, test from another network to isolate whether the corporate network is causing the failure.")
    return "\n".join(resolution_steps)
