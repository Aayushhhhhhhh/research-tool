# 📈 Research Tool: AI-Powered News Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://research-tool-cefmtbpsflvtrebncoucog.streamlit.app/)

**Research Tool** is a financial intelligence application designed to automate the analysis of equity research articles. By leveraging a **Retrieval-Augmented Generation (RAG)** pipeline, it transforms unstructured news content into actionable investment insights.

Built with **Python**, **LangChain**, and **LLAMA-3**, this tool addresses the "information overload" problem for analysts by providing instant executive summaries and an interactive Q&A interface for deep dives.

## 🚀 Live Demo
**Try the App:** [Click Here to Launch Research Tool](https://research-tool-cefmtbpsflvtrebncoucog.streamlit.app/)

## 🛠️ Key Features
* **Automated Executive Summaries:** Instantly generates a bulleted financial summary (Risks, Outlook, Key Metrics) upon processing a URL.
* **RAG Pipeline:** Uses **FAISS** vector storage and **Hugging Face Embeddings** to retrieve exact context from long-form articles, minimizing hallucinations.
* **Interactive Q&A:** Chat with the article to extract specific data points (e.g., "What is the target price?", "What are the Q3 margins?").
* **Performance Optimized:** Implements **Streamlit Caching** (`@st.cache_resource`) and **Session State** management to eliminate redundant computations and reduce latency by 90% for repeated queries.
* **Cost-Efficient Inference:** Integrated with **LLAMA-3.3-70B** (via OpenRouter) using token-limited prompts for high-speed, low-cost responses.

## 🏗️ Technical Architecture
* **Frontend:** Streamlit (Python)
* **LLM Orchestration:** LangChain
* **Language Model:** LLAMA-3.3-70B-Instruct (via OpenRouter API)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embedding Model:** HuggingFace (`all-MiniLM-L6-v2`)
* **Data Processing:** UnstructuredURLLoader, RecursiveCharacterTextSplitter

Created by Aayush Sonawane

## 📂 Project Structure
```bash
research-tool/
├── main.py              # Main application logic (UI + RAG Pipeline)
├── requirements.txt     # Python dependencies (pinned for stability)
├── packages.txt         # System-level dependencies for Linux deployment
├── .gitignore           # Security exclusions
└── README.md            # Project documentation
