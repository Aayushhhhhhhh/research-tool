import os
import streamlit as st
import pickle
import time
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Page Config
st.set_page_config(page_title="RockyBot: Equity Research Tool", layout="wide")
st.title("RockyBot: Equity Research Tool 📈")

# 2. Sidebar for User Inputs
st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# 3. Load API Key securely from Streamlit Secrets
# We will set this up in Step 4
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except:
    st.error("API Key not found. Please set OPENROUTER_API_KEY in Streamlit secrets.")
    st.stop()

# 4. Processing Logic
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

if process_url_clicked:
    if not urls:
        st.error("Please enter at least one URL.")
    else:
        # Load Data
        main_placeholder.text("Data Loading...Started...✅")
        try:
            loader = UnstructuredURLLoader(urls=urls)
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
            
            # Save Vector Store (Pickling)
            # Note: In production, use a persistent DB like Pinecone, but pickle works for simple demos
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)
                
            main_placeholder.success("Analysis Ready! Ask your question below.")
        except Exception as e:
            st.error(f"Error processing URLs: {e}")

# 5. Question & Answer Logic
query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            
            # Initialize LLM (LLAMA-3 via OpenRouter)
            llm = ChatOpenAI(
                openai_api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model_name="meta-llama/llama-3.3-70b-instruct:free", # Using a free sturdy model
                temperature=0.0
            )
            
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            
            result = chain.invoke({"question": query}, return_only_outputs=True)
            
            # Display Answer
            st.header("Answer")
            st.write(result["answer"])
            
            # Display Sources
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  
                for source in sources_list:
                    st.write(source)
    else:
        st.warning("Please process URLs first.")
