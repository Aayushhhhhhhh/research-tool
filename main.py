import os
import streamlit as st
import time
import langchain
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Page Config
st.set_page_config(page_title="RockyBot: Equity Research Tool", layout="wide")
st.title("RockyBot: Equity Research Tool 📈")

# 2. Sidebar
st.sidebar.title("News Article URL")
url = st.sidebar.text_input("Paste URL here")
process_url_clicked = st.sidebar.button("Analyze Article")

# 3. API Key Check
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except:
    st.error("API Key not found. Please set OPENROUTER_API_KEY in Streamlit secrets.")
    st.stop()

# --- OPTIMIZATION 1: CACHING THE HEAVY COMPUTATION ---
# @st.cache_resource tells Streamlit: "If the input (url) hasn't changed, 
# don't run this function again. Just return the saved result from RAM."
@st.cache_resource(show_spinner=False)
def process_data_to_vectorstore(url_input):
    # Load Data
    loader = UnstructuredURLLoader(urls=[url_input])
    data = loader.load()
    
    # Split Data (Token Limiting Strategy 1: Smart Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    
    # Create Embeddings
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# --- OPTIMIZATION 2: TOKEN LIMITING IN MODEL ---
def get_llm():
    return ChatOpenAI(
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model_name="meta-llama/llama-3.3-70b-instruct:free",
        temperature=0.0,
        max_tokens=700  # <--- LIMITS OUTPUT SIZE (Reduces Latency)
    )

# 4. Session State Management
if "summary" not in st.session_state:
    st.session_state.summary = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 5. Processing Logic
if process_url_clicked:
    if not url:
        st.error("Please enter a URL.")
    else:
        st.session_state.summary = None
        main_placeholder = st.empty()
        main_placeholder.text("Processing... ⏳")
        
        try:
            # LATENCY REDUCTION: usage of cache instead of re-computing
            vectorstore = process_data_to_vectorstore(url)
            
            # LATENCY REDUCTION: Store in Session State (RAM) instead of Disk (Pickle)
            st.session_state.vectorstore = vectorstore
            
            main_placeholder.text("Analysis Ready! ✅")

            # Generate Summary
            llm = get_llm()
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            
            # Prompt Engineering for Brevity
            summary_query = "Summarize the key financial points, risks, and future outlook. Be concise and use bullet points."
            result = chain.invoke({"question": summary_query}, return_only_outputs=True)
            
            st.session_state.summary = result["answer"]
            
        except Exception as e:
            st.error(f"Error processing URL: {e}")

# 6. Display Summary
if st.session_state.summary:
    st.markdown("### 📝 Executive Summary")
    st.info(st.session_state.summary)

# 7. Follow-up Question Logic
st.markdown("---")
st.subheader("Ask Follow-up Questions")
query = st.text_input("Example: What is the target price?")

if query:
    if st.session_state.vectorstore:
        llm = get_llm()
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever()
        )
        
        result = chain.invoke({"question": query}, return_only_outputs=True)
        st.write(result["answer"])
        
        if result.get("sources"):
            st.caption(f"Sources: {result['sources']}")
    else:
        st.warning("Please analyze an article first.")
