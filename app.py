import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tempfile
import shutil

load_dotenv()

st.set_page_config(page_title="Professional RAG System", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .summary-btn > button {
        background-color: #2196F3;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 Professional RAG System")
st.markdown("**Semantic Search | No Hallucinations | Document Summary**")

@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def init_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = init_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    faiss_path = "./faiss_db"
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
    vectorstore.save_local(faiss_path)
    
    os.unlink(tmp_path)
    return len(chunks), documents

def load_vectorstore():
    faiss_path = "./faiss_db"
    if os.path.exists(faiss_path):
        return FAISS.load_local(faiss_path, init_embeddings(), allow_dangerous_deserialization=True)
    return None

def semantic_search(vectorstore, query, k=5):
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return results

def check_if_relevant(results, threshold=0.15):
    if not results:
        return False, 0
    max_score = results[0][1]
    return max_score >= threshold, max_score

def generate_summary(documents):
    """Generate a summary of the document"""
    # Take first 3 chunks for summary
    full_text = " ".join([doc.page_content[:500] for doc in documents[:3]])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a document summarizer. Provide a 2-3 sentence overview of what this document is about."),
        ("human", f"Document content:\n{full_text}\n\nSummary:")
    ])
    
    llm = init_llm()
    chain = prompt | llm
    response = chain.invoke({})
    return response.content

# Sidebar
with st.sidebar:
    st.header("⚙️ How to Use")
    st.markdown("""
    1. **Upload** a PDF document
    2. Click **Process Document**
    3. **Ask questions** or **Generate Summary**
    
    **Features:**
    - 🔍 Semantic search (cosine similarity)
    - 📝 Document summary for overview
    - 🛡️ No hallucinations (confidence threshold)
    - ⚡ Fast Groq inference
    """)
    
    st.divider()
    st.caption("Made with LangChain + Groq + FAISS")

# Main area
uploaded_file = st.file_uploader("📁 Upload a PDF document", type="pdf")

if uploaded_file:
    if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        if os.path.exists("./faiss_db"):
            shutil.rmtree("./faiss_db")
            st.info("🔄 Cache cleared for new document")
        st.session_state.documents = None
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📑 Process Document", type="primary"):
            with st.spinner("Creating embeddings..."):
                num_chunks, documents = process_pdf(uploaded_file)
                st.session_state.documents = documents
                st.success(f"✅ Processed! {num_chunks} chunks indexed")
                st.info("🎯 Using MPNet embeddings (768 dimensions)")
    
    with col2:
        if st.button("📝 Generate Summary", type="secondary"):
            if st.session_state.get('documents'):
                with st.spinner("Generating summary..."):
                    summary = generate_summary(st.session_state.documents)
                    st.success("📋 Document Summary")
                    st.write(summary)
            else:
                st.warning("⚠️ Please process the document first (click Process Document)")
    
    st.divider()
    
    # Search section with button
    st.subheader("🔍 Ask a Question")
    
    # Create row with input and button
    query = st.text_input("", placeholder="Type your question here...", label_visibility="collapsed")
    
    # Search button next to input
    col_search, col_empty = st.columns([1, 4])
    with col_search:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_clicked and query:
        vectorstore = load_vectorstore()
        
        if vectorstore is None:
            st.error("⚠️ Please process a document first.")
        else:
            with st.spinner("🔎 Searching semantically..."):
                results = semantic_search(vectorstore, query)
                is_relevant, confidence = check_if_relevant(results)
            
            if results:
                st.subheader(f"📄 Retrieved Chunks (Confidence: {confidence:.2f})")
                for i, (doc, score) in enumerate(results[:3]):
                    with st.expander(f"📌 Match {i+1} - Similarity: {score:.3f}"):
                        st.write(doc.page_content[:400])
            
            if not is_relevant:
                st.warning("❓ No sufficiently relevant information found. Try rephrasing your question.")
            else:
                with st.spinner("💡 Generating answer..."):
                    context = "\n\n".join([doc.page_content for doc, _ in results[:3]])
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """Answer based ONLY on the provided context. 
                        Be specific and factual. Keep answers to 2-3 sentences.
                        If the context doesn't fully answer, say what you can infer."""),
                        ("human", f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
                    ])
                    
                    llm = init_llm()
                    chain = prompt | llm
                    response = chain.invoke({})
                    
                    st.subheader("💡 Answer")
                    st.success(response.content)
    
    elif search_clicked and not query:
        st.warning("⚠️ Please type a question first.")

elif not uploaded_file:
    st.info("👈 Start by uploading a PDF document in the sidebar")

# Footer
st.divider()
st.caption("✨ Pure semantic search using cosine similarity | No keyword matching | FAISS vector database")