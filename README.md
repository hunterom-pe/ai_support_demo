# AI-Powered Customer Support Intelligence (RAG)

## Overview
This project demonstrates a real-world AI integration using Retrieval Augmented Generation (RAG).
The system retrieves the most relevant customer support data using semantic search and passes it
to a language model to generate structured support insights.

## Problem
Customer support agents waste time reading long ticket histories and internal notes to understand
a customer's issue before responding. This leads to slower responses, inconsistent tone, and errors.

## Solution
The system:

1. Splits customer history into individual documents (tickets and notes)
2. Converts each document into embeddings (semantic meaning)
3. Uses cosine similarity to retrieve only the most relevant context
4. Injects that context into an LLM prompt
5. Returns structured JSON output for easy automation

## Architecture
Customer Data → Embeddings → Semantic Search → Context Assembly → LLM → Structured Output

## Example Output
```json
{
  "issue_summary": "Customer was charged twice and is frustrated with lack of response.",
  "customer_sentiment": "frustrated",
  "draft_reply": "Thank you for your patience...",
  "recommended_actions": [
    "Refund duplicate charge",
    "Escalate mobile app crash issue"
  ]
}