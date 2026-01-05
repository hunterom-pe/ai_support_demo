import streamlit as st
import json
import math
import random

# ------------------------
# Simulated customer tickets database (stateful)
# ------------------------
if "tickets" not in st.session_state:
    st.session_state.tickets = [
        {"id": "TCK-1001", "customer": "Alice Johnson", "issue": "The iOS app crashes immediately when I try to log in.", "status": "Open"},
        {"id": "TCK-1002", "customer": "Bob Smith", "issue": "I've emailed twice already and no one has responded.", "status": "Pending"},
        {"id": "TCK-1003", "customer": "Charlie Davis", "issue": "I was charged twice for my March invoice and no one has fixed it.", "status": "Open"},
        {"id": "TCK-1004", "customer": "Dana Lee", "issue": "The website is loading very slowly for me.", "status": "Open"},
        {"id": "TCK-1005", "customer": "Evan Kim", "issue": "I received the wrong item in my order and need a replacement.", "status": "Pending"}
    ]

# ------------------------
# Fake embeddings
# ------------------------
def embed(text):
    return [0.1] * 768

# ------------------------
# Cosine similarity
# ------------------------
def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(x*x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0
    return dot / (mag_a * mag_b)

# ------------------------
# Retrieve relevant tickets
# ------------------------
def retrieve_relevant_tickets(query, tickets, top_k=3):
    query_embedding = embed(query)
    scored = [(cosine_similarity(query_embedding, embed(t["issue"])), t) for t in tickets]
    scored.sort(reverse=True)
    return [t for _, t in scored[:top_k]]

# ------------------------
# Fake AI response
# ------------------------
def call_llm(prompt):
    return json.dumps({
        "issue_summary": [
            "The customer reports app crashes, lack of support response, and billing issues."
        ],
        "customer_sentiment": random.choice(["Frustrated", "Upset", "Concerned"]),
        "draft_reply": (
            "Dear Customer, we are investigating your issue. "
            "Our support team is prioritizing the app crash, email response delays, and billing concerns. "
            "We will follow up shortly with a resolution."
        ),
        "recommended_actions": [
            "Investigate app crash",
            "Respond to pending tickets",
            "Resolve billing issue",
            "Follow up with customers"
        ]
    })

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="AI Customer Support Demo", page_icon="🤖", layout="wide")

st.title("🤖 AI Customer Support Assistant (Interactive Demo)")
st.markdown("""
Simulated integration with a customer database (like Salesforce or Zendesk).  
AI automatically retrieves relevant tickets and drafts structured replies. **All outputs are demo-only.**
""")

# --- Top metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Tickets in Database", len(st.session_state.tickets))
col2.metric("Avg Response Time", "3s", delta="Demo")
col3.metric("Customer Satisfaction", "95%", delta="Simulated")
st.markdown("---")

# --- Show tickets table ---
st.subheader("📂 Customer Tickets")
ticket_table = []
for t in st.session_state.tickets:
    ticket_table.append({
        "ID": t["id"], "Customer": t["customer"], "Issue": t["issue"], "Status": t["status"]
    })
st.table(ticket_table)
st.markdown("---")

# --- Input query ---
st.subheader("📝 Enter Customer Issue")
sample_issues = [
    "Customer is upset about billing and lack of response",
    "App crashes on login",
    "Received wrong item in order",
    "Website loading slowly",
    "Customer wants refund for double charge"
]
selected_issue = st.selectbox("Or select a sample issue:", sample_issues)
user_issue = st.text_input("Type a customer issue here (or edit selected):", selected_issue)

# --- Run AI ---
if st.button("Run AI"):
    st.info("Retrieving relevant tickets and generating AI response...")

    relevant_tickets = retrieve_relevant_tickets(user_issue, st.session_state.tickets)

    col1, col2 = st.columns(2)

    # --- Left column: relevant tickets with interactive status ---
    with col1:
        st.subheader("🔍 Relevant Tickets")
        for t in relevant_tickets:
            st.markdown(
                f"<div style='background-color:#FFF3E0; padding:10px; border-radius:5px; margin-bottom:5px'>"
                f"<strong>{t['id']} - {t['customer']}</strong><br>{t['issue']}<br>Status: {t['status']}</div>",
                unsafe_allow_html=True
            )
            new_status = st.selectbox(
                f"Update status for {t['id']}",
                options=["Open", "Pending", "Resolved", "Followed Up"],
                index=["Open", "Pending", "Resolved", "Followed Up"].index(t["status"]),
                key=f"status_{t['id']}"
            )
            if new_status != t["status"]:
                t["status"] = new_status
                st.success(f"Ticket {t['id']} updated to {new_status}")

    # --- Right column: AI outputs ---
    with col2:
        prompt = "\n".join([t["issue"] for t in relevant_tickets])
        response = call_llm(prompt)

        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            response_json = {
                "issue_summary": ["Failed to parse JSON."],
                "customer_sentiment": "Unknown",
                "draft_reply": response,
                "recommended_actions": []
            }

        st.subheader("💬 AI Summary / Draft Reply")
        st.markdown(
            f"<div style='background-color:#E8F5E9; padding:10px; border-radius:5px'>"
            f"<strong>Issue Summary:</strong> {' '.join(response_json.get('issue_summary', []))}<br>"
            f"<strong>Customer Sentiment:</strong> {response_json.get('customer_sentiment', 'Unknown')}<br>"
            f"<strong>Draft Reply:</strong> {response_json.get('draft_reply', 'No reply generated.')}<br>"
            f"<strong>Recommended Actions:</strong><br>- {'<br>- '.join(response_json.get('recommended_actions', []))}"
            f"</div>",
            unsafe_allow_html=True
        )

        # --- Structured JSON ---
        st.subheader("💡 Structured JSON Output")
        st.text_area("Copy JSON", value=json.dumps(response_json, indent=2), height=200, key="copy_output")
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(response_json, indent=2),
            file_name="ai_output.json",
            mime="application/json"
        )

    st.markdown("---")
    st.success("Demo complete! Update tickets above or enter another issue to try again.")
