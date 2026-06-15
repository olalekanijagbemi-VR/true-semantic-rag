import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tempfile

load_dotenv()

st.set_page_config(page_title="Professional RAG System", layout="wide")
st.title("Professional RAG System")
st.markdown("**Semantic Search | No Hallucinations | Production Ready**")

@st.cache_resource
def init_embeddings():
    # Using OpenAI embeddings (works with Groq's API format)
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1"
    )

@st.cache_resource
def init_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

@st.cache_resource
def init_vectorstore():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=init_embeddings(),
        collection_metadata={"hnsw:space": "cosine"}
    )

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    vectorstore = init_vectorstore()
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    
    os.unlink(tmp_path)
    return len(chunks)

def semantic_search(query, vectorstore, k=5):
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return results

def check_if_relevant(results, threshold=0.35):
    if not results:
        return False, 0
    max_score = results[0][1]
    return max_score >= threshold, max_score

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db")
            st.info("Cache cleared for new document")
    
    if st.button("Process Document"):
        with st.spinner("Creating embeddings..."):
            num_chunks = process_pdf(uploaded_file)
            st.success(f" Document processed! {num_chunks} chunks indexed")
    
    st.divider()
    query = st.text_input("Ask a question:")
    
    if query and st.button("Search"):
        vectorstore = init_vectorstore()
        
        with st.spinner("Searching..."):
            results = semantic_search(query, vectorstore)
            is_relevant, confidence = check_if_relevant(results)
        
        st.subheader(f"Retrieved Chunks (Confidence: {confidence:.2f})")
        for i, (doc, score) in enumerate(results[:3]):
            with st.expander(f"Similarity Score: {score:.3f}"):
                st.write(doc.page_content[:400])
        
        if not is_relevant:
            st.warning("No sufficiently relevant information found.")
        else:
            with st.spinner("Generating answer..."):
                context = "\n\n".join([doc.page_content for doc, _ in results[:3]])
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer based ONLY on context. Be specific and factual. Keep answers to 2-3 sentences."),
                    ("human", f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
                ])
                
                llm = init_llm()
                chain = prompt | llm
                response = chain.invoke({})
                
                st.subheader("Answer")
                st.success(response.content)

with st.sidebar:
    st.header("Production Features")
    st.markdown("""
    - Semantic search with cosine similarity
    - No hallucinations (confidence threshold)
    - Groq API for fast inference
    - ChromaDB vector storage
    """)