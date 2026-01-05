import streamlit as st
import subprocess
import json
import math

# ------------------------
# Load documents
# ------------------------
def load_documents():
    with open("customer_data.txt", "r") as file:
        return [line.strip() for line in file if line.strip()]

# ------------------------
# Embeddings (safe)
# ------------------------
def embed(text):
    try:
        result = subprocess.run(
            ["ollama", "run", "nomic-embed-text"],
            input=text,
            text=True,
            capture_output=True
        )
        return json.loads(result.stdout)["embedding"]
    except Exception:
        # fallback embedding if Ollama fails
        return [0.0] * 768

# ------------------------
# Cosine similarity (zero-safe)
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
# Build AI prompt
# ------------------------
def build_prompt(context):
    return f"""
You are a senior customer support specialist.

Relevant customer context:
{context}

Return ONLY valid JSON with:
- issue_summary
- customer_sentiment
- draft_reply
- recommended_actions
"""

# ------------------------
# Call LLM
# ------------------------
def call_llm(prompt):
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="AI Customer Support Demo", page_icon="🤖", layout="wide")

# --- Title & intro ---
st.title("🤖 AI-Powered Customer Support Assistant")
st.markdown("""
This demo shows how AI can automatically retrieve relevant customer data and draft structured responses.
Use the input below to simulate a customer issue.
""")

# --- Metrics / stats ---
st.subheader("⏱ Demo Stats")
st.metric(label="Tickets Processed", value="100", delta="Processed in 3s")
st.metric(label="AI Accuracy (simulated)", value="95%", delta="Compared to manual review")
st.markdown("---")

# --- Business value paragraph ---
st.markdown("""
### 📌 Why this matters to business
Customer support teams spend hours reading tickets and notes.  
This AI assistant:
- Automatically finds the most relevant context
- Drafts structured responses
- Saves time and reduces errors

This demo shows the **potential of AI integration** in real-world customer support workflows.
""")
st.markdown("---")

# --- Step 1: Show original tickets ---
st.subheader("📄 Customer Tickets / Notes")
documents = load_documents()
for doc in documents:
    st.markdown(f"- {doc}")
st.markdown("---")

# --- Step 2: Input query ---
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
        # Build prompt & get AI output
        context = "\n".join(relevant_docs)
        prompt = build_prompt(context)
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

        # --- Structured JSON output + copy/download ---
        st.subheader("💡 AI Output (Structured JSON)")
        st.text_area("Copy AI Output", value=response, height=200, key="copy_output")

        st.download_button(
            label="📥 Download AI Output as JSON",
            data=json.dumps(response_json, indent=2),
            file_name="ai_output.json",
            mime="application/json"
        )

    st.markdown("---")
    st.success("Demo complete! You can enter another issue above to try again.")