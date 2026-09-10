from rag.analyzer import analyze_ticket
from rag.generator import build_context, generate_resolution


def run_rag_pipeline(ticket, retriever):
    analysis = analyze_ticket(ticket)
    results = retriever.search(analysis["query"], top_k=2)
    context = build_context(results)
    resolution = generate_resolution(ticket, results)

    return {
        "ticket": ticket,
        "analysis": analysis,
        "retrieved_documents": results,
        "context": context,
        "resolution": resolution
    }
