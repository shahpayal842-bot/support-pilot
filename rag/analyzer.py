def analyze_ticket(ticket):
    text = (ticket.get("title", "") + " " + ticket.get("description", "")).lower()
    possible_keywords = ["vpn", "network", "firewall", "authentication", "timeout", "connection", "dns", "password"]
    keywords = [kw for kw in possible_keywords if kw in text]
    return {
        "ticket_id": ticket.get("id", "T-2023-4521"),
        "category": ticket.get("category", "Network Connectivity"),
        "priority": ticket.get("priority", "High"),
        "keywords": keywords,
        "query": text
    }
