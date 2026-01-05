import os
import streamlit as st
import pickle
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

# 2. Sidebar for User Inputs (Changed to Single URL)
st.sidebar.title("News Article URL")
url = st.sidebar.text_input("Paste URL here")
process_url_clicked = st.sidebar.button("Analyze Article")

# 3. Load API Key
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except:
    st.error("API Key not found. Please set OPENROUTER_API_KEY in Streamlit secrets.")
    st.stop()

# 4. Processing Logic
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

if process_url_clicked:
    if not url:
        st.error("Please enter a URL.")
    else:
        # Load Data
        main_placeholder.text("Data Loading...Started...✅")
        try:
            # We wrap the single URL in a list because the loader expects a list
            loader = UnstructuredURLLoader(urls=[url])
            data = loader.load()
            
            # Split Data
            main_placeholder.text("Text Splitting...Started...✅")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)
            
            # Create Embeddings
            main_placeholder.text("Embedding Vector Building...Started...✅")
            embeddings = HuggingFaceEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            
            # Save Vector Store
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)
                
            main_placeholder.text("Analysis Ready!")

            # --- NEW FEATURE: AUTO-SUMMARY ---
            st.markdown("### 📝 Executive Summary")
            
            # Initialize LLM & Chain immediately for the summary
            llm = ChatOpenAI(
                openai_api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model_name="meta-llama/llama-3.3-70b-instruct:free",
                temperature=0.0
            )
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=vectorstore_openai.as_retriever()
            )
            
            # We ask the bot to summarize programmatically
            summary_query = "Summarize the key financial points, risks, and future outlook from this article in bullet points."
            result = chain.invoke({"question": summary_query}, return_only_outputs=True)
            
            # Display the summary inside a nice container
            st.info(result["answer"])
            # ---------------------------------

        except Exception as e:
            st.error(f"Error processing URL: {e}")

# 5. Follow-up Question Logic
st.markdown("---")
st.subheader("Ask Follow-up Questions")
query = st.text_input("Example: What is the target price?")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            
            llm = ChatOpenAI(
                openai_api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model_name="meta-llama/llama-3.3-70b-instruct:free",
                temperature=0.0
            )
            
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            
            result = chain.invoke({"question": query}, return_only_outputs=True)
            
            st.write(result["answer"])
            
            sources = result.get("sources", "")
            if sources:
                st.caption(f"Sources: {sources}")
    else:
        st.warning("Please analyze an article first.")
