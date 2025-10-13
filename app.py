"""
Streamlit UI for Intelligent Programming Documentation Search Engine
---------------------------------------------------------------------
Beautiful, interactive interface for demonstrating the system capabilities.
Perfect for your demo video!
"""

import streamlit as st
import requests
import time
import os
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="DevDocs AI - Intelligent Programming Documentation Search",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_example_queries() -> List[str]:
    """Get example queries from API."""
    try:
        response = requests.get(f"{API_BASE}/examples")
        return response.json().get("example_queries", [])
    except:
        return []

def upload_document(file, document_name: str = None) -> Dict[str, Any]:
    """Upload document to the knowledge base."""
    files = {"file": (file.name, file.getvalue(), file.type)}
    data = {"document_name": document_name or file.name}
    
    response = requests.post(f"{API_BASE}/upload", files=files, data=data)
    return response.json()

def query_knowledge_base(question: str, filters: Dict = None) -> Dict[str, Any]:
    """Query the knowledge base."""
    payload = {"question": question, "filters": filters}
    response = requests.post(f"{API_BASE}/query", json=payload)
    return response.json()

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics."""
    response = requests.get(f"{API_BASE}/stats")
    return response.json()

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">🧠 Intelligent Programming Documentation Search Engine</div>', 
                unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("🚨 API server is not running! Please start the server with:")
        st.code("uvicorn src.api:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Info")
        
        try:
            stats = get_system_stats()
            st.metric("Documents in Knowledge Base", 
                     stats.get("knowledge_base", {}).get("total_chunks", 0))
            st.metric("System Version", stats.get("system", {}).get("version", "1.0.0"))
        except:
            st.info("System stats unavailable")
        
        st.header("📁 Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a document", 
            type=['pdf', 'txt'],
            help="Upload programming documentation (PDF or text files)"
        )
        
        if uploaded_file:
            document_name = st.text_input("Document name (optional)", uploaded_file.name)
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        result = upload_document(uploaded_file, document_name)
                        st.success(f"✅ {result['message']}")
                        st.info(f"Processed {result['chunks_processed']} chunks")
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Example queries
        example_queries = get_example_queries()
        if example_queries:
            st.write("**Try these examples:**")
            for i, example in enumerate(example_queries[:3]):
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.question = example
        
        # Question input
        question = st.text_area(
            "Ask about programming documentation:",
            height=100,
            placeholder="e.g., How do I create a FastAPI endpoint with path parameters?",
            key="question"
        )
        
        # Query options
        with st.expander("Advanced Options"):
            filter_type = st.selectbox(
                "Filter by content type:",
                ["All", "API Documentation", "Code Examples", "Setup Guides"]
            )
            
            filters = None
            if filter_type == "API Documentation":
                filters = {"content_type": "api_documentation"}
            elif filter_type == "Code Examples":
                filters = {"content_type": "code_example"}
            elif filter_type == "Setup Guides":
                filters = {"content_type": "setup_guide"}
        
        # Query button
        if st.button("Get Answer", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("🔍 Searching documentation..."):
                    try:
                        start_time = time.time()
                        result = query_knowledge_base(question, filters)
                        processing_time = time.time() - start_time
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("💡 Answer")
                        
                        # Confidence indicator
                        confidence = result.get("confidence", 0)
                        if confidence > 0.7:
                            confidence_class = "confidence-high"
                        elif confidence > 0.4:
                            confidence_class = "confidence-medium"
                        else:
                            confidence_class = "confidence-low"
                        
                        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confidence", f"{confidence:.1%}")
                        with col2:
                            st.metric("Sources Used", len(result.get("sources", [])))
                        with col3:
                            st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                        
                        # Sources
                        if result.get("sources"):
                            with st.expander("📚 Source Documents"):
                                for source in result["sources"]:
                                    st.write(f"• {source}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing question: {str(e)}")
            else:
                st.warning("Please enter a question")
    
    with col2:
        st.header("🎯 Quick Actions")
        
        if st.button("🔄 Refresh Knowledge Base", use_container_width=True):
            try:
                stats = get_system_stats()
                st.success("Knowledge Base refreshed!")
                st.info(f"Total chunks: {stats.get('knowledge_base', {}).get('total_chunks', 0)}")
            except:
                st.error("Failed to refresh")
        
        if st.button("📋 View All Examples", use_container_width=True):
            try:
                examples = get_example_queries()
                st.write("**Example Queries:**")
                for example in examples:
                    st.write(f"• {example}")
            except:
                st.error("Failed to load examples")
        
        st.header("ℹ️ About")
        st.info("""
        This Intelligent Programming Documentation Search Engine:
        
        • **Searches** across your technical documentation
        • **Understands** programming concepts and code
        • **Provides** accurate, cited answers
        • **Supports** PDF and text files
        
        Perfect for API docs, codebases, and technical manuals!
        """)

if __name__ == "__main__":
    main()