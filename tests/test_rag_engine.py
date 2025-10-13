import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import process_document
from src.rag_engine import create_rag_engine

def test_rag_engine():
    print("🧪 Testing RAG Engine...")
    
    # Create test document
    test_content = """
# FastAPI Documentation

## Quick Start
To create a FastAPI application, first install the package:
pip install fastapi uvicorn

## Basic Example
Here's a simple FastAPI application:

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

## Important Notes
- FastAPI requires Python 3.7+
- Automatic API docs available at /docs and /redoc
- Built-in data validation using Pydantic
"""
    
    # Write test file
    test_file = "data/raw_documents/test_fastapi.txt"
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    with open(test_file, "w") as f:
        f.write(test_content)
    
    try:
        # 1. Process document
        print("📄 Processing test document...")
        chunks = process_document(test_file, "FastAPI Guide")
        print(f"✅ Created {len(chunks)} chunks")
        
        # 2. Initialize RAG engine
        print("🔧 Initializing RAG engine...")
        rag_engine = create_rag_engine()
        print("✅ RAG engine initialized")
        
        # 3. Add documents to vector DB
        print("📥 Adding chunks to vector database...")
        added_count = rag_engine.add_documents(chunks)
        print(f"✅ Added {added_count} chunks")
        
        # 4. Test search
        print("🔍 Testing search...")
        results = rag_engine.search("How to create a FastAPI endpoint")
        print(f"✅ Search returned {len(results)} results")
        
        # 5. Test full RAG pipeline
        print("🤖 Testing full RAG pipeline...")
        response = rag_engine.query("How do I create a GET endpoint in FastAPI?")
        
        print(f"\n📝 Question: {response['question']}")
        print(f"💡 Answer: {response['answer']}")
        print(f"📚 Sources: {response['sources']}")
        print(f"🎯 Confidence: {response['confidence']}")
        print(f"🔢 Chunks used: {response['chunks_used']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_rag_engine()