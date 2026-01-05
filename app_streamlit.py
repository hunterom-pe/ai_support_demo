import streamlit as st
import json
import math
import random
import pandas as pd
import plotly.express as px

# ------------------------
# Stateful ticket DB
# ------------------------
if "tickets" not in st.session_state:
    st.session_state.tickets = [
        {"id": "TCK-1001", "customer": "Alice Johnson", "issue": "The iOS app crashes immediately when I try to log in.", "status": "Open"},
        {"id": "TCK-1002", "customer": "Bob Smith", "issue": "I've emailed twice already and no one has responded.", "status": "Pending"},
        {"id": "TCK-1003", "customer": "Charlie Davis", "issue": "I was charged twice for my March invoice and no one has fixed it.", "status": "Open"},
        {"id": "TCK-1004", "customer": "Dana Lee", "issue": "The website is loading very slowly for me.", "status": "Open"},
        {"id": "TCK-1005", "customer": "Evan Kim", "issue": "I received the wrong item in my order and need a replacement.", "status": "Pending"}
    ]

if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []

# ------------------------
# Fake embeddings / AI
# ------------------------
def embed(text):
    # Fake embedding for demo
    return [0.1] * 768

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(x*x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0
    return dot / (mag_a * mag_b)

def retrieve_relevant_tickets(query, tickets, top_k=3):
    query_embedding = embed(query)
    scored = [(cosine_similarity(query_embedding, embed(t["issue"])), t) for t in tickets]
    scored.sort(reverse=True)
    return [t for _, t in scored[:top_k]]

def call_llm(prompt):
    # Fake AI response for demo
    sentiment = random.choice(["Frustrated", "Upset", "Concerned"])
    st.session_state.sentiment_history.append(sentiment)
    return json.dumps({
        "issue_summary": [
            "The customer reports app crashes, lack of support response, and billing issues."
        ],
        "customer_sentiment": sentiment,
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
st.set_page_config(page_title="AI Support Dashboard", page_icon="🤖", layout="wide")
st.title("🤖 AI Customer Support Dashboard")

st.markdown("""
Simulated SaaS integration demo with **interactive tickets** and AI-assisted summaries.  
All outputs are **demo-only** and fully compatible with Streamlit Cloud.
""")

# --- Metrics ---
open_count = sum(1 for t in st.session_state.tickets if t["status"] == "Open")
pending_count = sum(1 for t in st.session_state.tickets if t["status"] == "Pending")
resolved_count = sum(1 for t in st.session_state.tickets if t["status"] in ["Resolved", "Followed Up"])

col1, col2, col3 = st.columns(3)
col1.metric("Open Tickets", open_count, delta=f"{open_count - 2}")
col2.metric("Pending Tickets", pending_count, delta=f"{pending_count - 1}")
col3.metric("Resolved / Followed Up", resolved_count, delta=f"{resolved_count - 0}")

st.markdown("---")

# --- Ticket Status Pie Chart ---
st.subheader("📊 Ticket Status Distribution")
status_counts = pd.DataFrame({
    "Status": ["Open", "Pending", "Resolved/Followed Up"],
    "Count": [open_count, pending_count, resolved_count]
})
fig_status = px.pie(status_counts, names="Status", values="Count", color="Status",
                    color_discrete_map={"Open":"#FFCDD2","Pending":"#FFF9C4","Resolved/Followed Up":"#C8E6C9"})
st.plotly_chart(fig_status, use_container_width=True)

st.markdown("---")

# --- Tickets as cards ---
st.subheader("📂 Customer Tickets")
status_colors = {"Open": "#FFCDD2", "Pending": "#FFF9C4", "Resolved": "#C8E6C9", "Followed Up": "#B3E5FC"}

for t in st.session_state.tickets:
    color = status_colors.get(t["status"], "#E0E0E0")
    st.markdown(
        f"<div style='background-color:{color}; padding:15px; border-radius:8px; margin-bottom:10px'>"
        f"<strong>{t['id']} - {t['customer']}</strong><br>{t['issue']}<br>Status: <strong>{t['status']}</strong></div>",
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

st.markdown("---")

# --- Input / AI ---
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

if st.button("Run AI"):
    st.info("Analyzing tickets and generating AI output...")

    relevant_tickets = retrieve_relevant_tickets(user_issue, st.session_state.tickets)

    col1, col2 = st.columns(2)

    # --- Left: Relevant tickets ---
    with col1:
        st.subheader("🔍 Relevant Tickets")
        for t in relevant_tickets:
            color = status_colors.get(t["status"], "#E0E0E0")
            st.markdown(
                f"<div style='background-color:{color}; padding:12px; border-radius:6px; margin-bottom:8px'>"
                f"<strong>{t['id']} - {t['customer']}</strong><br>{t['issue']}<br>Status: {t['status']}</div>",
                unsafe_allow_html=True
            )

    # --- Right: AI output ---
    with col2:
        prompt = "\n".join([t["issue"] for t in relevant_tickets])
        response = call_llm(prompt)
        response_json = json.loads(response)

        st.subheader("💬 AI Summary / Draft Reply")
        st.markdown(
            f"<div style='background-color:#E3F2FD; padding:12px; border-radius:6px'>"
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

    # --- AI Sentiment Chart ---
    st.subheader("📊 AI Customer Sentiment (Simulation)")
    if st.session_state.sentiment_history:
        sentiment_counts = pd.Series(st.session_state.sentiment_history).value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig_sentiment = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                               color_discrete_map={"Frustrated":"#EF5350","Upset":"#FFA726","Concerned":"#29B6F6"})
        st.plotly_chart(fig_sentiment, use_container_width=True)

    st.success("Demo complete! Update tickets above or enter another issue to try again.")
