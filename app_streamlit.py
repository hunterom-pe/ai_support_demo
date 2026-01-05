import streamlit as st
import json
import math

# ------------------------
# Load demo tickets
# ------------------------
def load_documents():
    # Pre-filled sample tickets
    return [
        "Ticket: The iOS app crashes immediately when I try to log in.",
        "Ticket: I've emailed twice already and no one has responded.",
        "Ticket: I was charged twice for my March invoice and no one has fixed it.",
        "Ticket: The website is loading very slowly for me.",
        "Ticket: I received the wrong item in my order and need a replacement."
    ]

# ------------------------
# Fake embeddings for demo
# ------------------------
def embed(text):
    # Fixed vector for cosine similarity demo
    return [0.1] * 768

# ------------------------
# Cosine similarity (safe)
# ------------------------
def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(x*x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0
    return dot / (mag_a * mag_b)

# ------------------------
# Semantic search
# ------------------------
def retrieve_relevant_docs(query, documents, top_k=3):
    query_embedding = embed(query)
    scored = []
    for doc in documents:
        score = cosine_similarity(query_embedding, embed(doc))
        scored.append((score, doc))
    scored.sort(reverse=True)
    return [doc for _, doc in scored[:top_k]]

# ------------------------
# Fake AI response for demo
# ------------------------
def call_llm(prompt):
    return json.dumps({
        "issue_summary": [
            "The customer reports login crashes and billing issues."
        ],
        "customer_sentiment": "Frustrated",
        "draft_reply": (
            "Dear Customer, we are investigating your issue. "
            "Regarding the app crash, our engineers are working on a fix. "
            "For billing concerns, our support team will resolve the double charge promptly."
        ),
        "recommended_actions": [
            "Fix the app crash issue",
            "Resolve the double charge",
            "Follow up with the customer to ensure satisfaction"
        ]
    })

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="AI Customer Support Demo", page_icon="🤖", layout="wide")

# --- Title & intro ---
st.title("🤖 AI-Powered Customer Support Assistant (Demo)")
st.markdown("""
This demo shows how AI can automatically retrieve relevant customer data and draft structured responses.
All data and AI outputs are **simulated** for demonstration purposes.
""")

# --- Demo metrics ---
st.subheader("⏱ Demo Stats")
st.metric(label="Tickets Processed", value="100", delta="Processed instantly")
st.metric(label="AI Accuracy (simulated)", value="95%", delta="High accuracy for demo")
st.markdown("---")

# --- Why this matters ---
st.markdown("""
### 📌 Why this matters
Customer support teams spend hours reading tickets and notes.  
This AI assistant:
- Automatically finds the most relevant context
- Drafts structured responses
- Saves time and reduces errors

This demo shows the **potential of AI integration** in real-world customer support workflows.
""")
st.markdown("---")

# --- Show demo tickets ---
st.subheader("📄 Customer Tickets / Notes")
documents = load_documents()
for doc in documents:
    st.markdown(f"- {doc}")
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
    st.info("Retrieving relevant context and generating AI response...")

    # Retrieve relevant context
    relevant_docs = retrieve_relevant_docs(user_issue, documents)

    # --- Side-by-side layout ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Relevant Context Found")
        st.markdown(
            "<div style='background-color:#E0F7FA; padding:10px; border-radius:5px'>"
            + "\n".join(f"- {doc}" for doc in relevant_docs)
            + "</div>",
            unsafe_allow_html=True
        )

    with col2:
        # Build a fake prompt (just for demo)
        context = "\n".join(relevant_docs)
        prompt = f"Simulated prompt based on context:\n{context}"
        response = call_llm(prompt)

        # --- Parse JSON safely ---
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            response_json = {
                "issue_summary": ["Failed to parse JSON."],
                "customer_sentiment": "Unknown",
                "draft_reply": response,
                "recommended_actions": []
            }

        # --- Readable AI output ---
        st.subheader("💬 AI Summary / Draft Reply")
        st.markdown(f"""
**Issue Summary:**  
{ ' '.join(response_json.get('issue_summary', [])) }

**Customer Sentiment:**  
{response_json.get('customer_sentiment', 'Unknown')}

**Draft Reply:**  
{response_json.get('draft_reply', 'No reply generated.')}

**Recommended Actions:**  
- { '\n- '.join(response_json.get('recommended_actions', [])) }
""")

        # --- Structured JSON + copy/download ---
        st.subheader("💡 AI Output (Structured JSON)")
        st.text_area("Copy AI Output", value=json.dumps(response_json, indent=2), height=200, key="copy_output")

        st.download_button(
            label="📥 Download AI Output as JSON",
            data=json.dumps(response_json, indent=2),
            file_name="ai_output.json",
            mime="application/json"
        )

    st.markdown("---")
    st.success("Demo complete! You can enter another issue above to try again.")
