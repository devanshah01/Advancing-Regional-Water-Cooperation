import os
import glob
import streamlit as st
import ollama
import chromadb
from pypdf import PdfReader

# --- CONFIGURATION ---
MODEL_NAME = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_db"))
DOCS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))

st.set_page_config(page_title="Local RAG Chatbot", page_icon="🤖")
st.title("Local Document Chatbot")
st.write("Chat with your local `.txt` and `.pdf` files using Ollama & Llama 3.2.")

@st.cache_resource
def initialize_vector_db():
    """Loads docs and builds ChromaDB vector database once inside a spinner."""
    with st.spinner("Initializing database and scanning documents..."):
        client = chromadb.PersistentClient(path=DB_PATH)
        
        # Check if collection already exists, if not create it
        try:
            collection = client.get_collection("local_docs")
            if collection.count() > 0:
                return collection, "loaded_existing"
        except Exception:
            pass
            
        collection = client.get_or_create_collection("local_docs")
        
        # Read files
        documents = []
        for filepath in glob.glob(os.path.join(DOCS_FOLDER, "*.txt")):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                documents.append({"source": os.path.basename(filepath), "text": f.read()})

        for filepath in glob.glob(os.path.join(DOCS_FOLDER, "*.pdf")):
            try:
                reader = PdfReader(filepath)
                text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
                if text.strip():
                    documents.append({"source": os.path.basename(filepath), "text": text})
            except Exception as e:
                print(f"Could not read {filepath}: {e}")

        if not documents:
            return None, "no_documents"

        # Chunk and embed
        doc_id = 0
        for doc in documents:
            text = doc["text"]
            chunk_size, overlap = 500, 50
            for start in range(0, len(text), chunk_size - overlap):
                chunk = text[start:start + chunk_size]
                response = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
                collection.add(
                    ids=[str(doc_id)],
                    embeddings=[response["embedding"]],
                    documents=[chunk],
                    metadatas=[{"source": doc["source"]}]
                )
                doc_id += 1
                
        return collection, "created_new"

# Call the function directly without wrapping it externally in a layout block/spinner
collection, status = initialize_vector_db()

# Handle status/UI messages outside the cached function
if status == "loaded_existing":
    st.toast("Loaded existing database index!")

if collection is None or status == "no_documents":
    st.warning("⚠️ No `.txt` or `.pdf` files found in this folder! Add some files and restart the app.")
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt input box at the bottom
if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & thinking..."):
            # 1. Embed query & search vector DB
            query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            
            retrieved_chunks = results["documents"][0]
            sources = set(meta["source"] for meta in results["metadatas"][0])

            # 2. Build prompt context
            context = "\n\n---\n\n".join(retrieved_chunks)
            prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context below. If you don't know, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {query}
"""
            # 3. Generate response
            response = ollama.generate(model=MODEL_NAME, prompt=prompt)
            answer = response["response"]
            
            # Append sources to output
            source_text = f"\n\n**Sources:** {', '.join(sources)}"
            full_response = answer + source_text
            
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})