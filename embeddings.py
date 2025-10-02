import chromadb
from chunking import chunk_all_repos
from sentence_transformers import SentenceTransformer
import os

# Initialize embedding model - using all-MiniLM-L6-v2
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model 'all-MiniLM-L6-v2' loaded successfully")
    print(f"   - Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    exit(1)

# Initialize ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="repo_chunks",
        metadata={"description": "GitHub repository chunks with all-MiniLM-L6-v2 embeddings"}
    )
    print("✅ ChromaDB collection ready")
except Exception as e:
    print(f"❌ Error initializing ChromaDB: {e}")
    exit(1)

def create_embeddings():
    """Create embeddings for all repository chunks and store in ChromaDB"""
    # Step 1: Load chunks
    print("🔄 Loading chunks from repositories...")
    chunks = chunk_all_repos()
    
    if not chunks:
        print("❌ No chunks to process. Please run chunking first.")
        return
    
    print(f"🔄 Processing {len(chunks)} chunks...")
    
    # Clear existing collection to avoid duplicates
    try:
        collection.delete(where={})  # Clear all documents
        print("🧹 Cleared existing collection")
    except:
        print("📝 Starting with fresh collection")
    
    # Step 2: Batch processing for better performance
    batch_size = 100
    successful_embeddings = 0
    documents = []
    metadatas = []
    embeddings_list = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        try:
            content = chunk["content"].strip()
            
            # Skip very short or empty content
            if len(content) < 10:
                continue
                
            metadata = {
                "repo": chunk["repo"],
                "file": chunk["file"],
                "function": chunk.get("function", "") or "",
                "class": chunk.get("class", "") or "",
                "chunk_id": i
            }

            documents.append(content)
            metadatas.append(metadata)
            ids.append(f"chunk_{i}_{hash(content) % 100000:08x}")
            
            # Process in batches for better performance
            if len(documents) >= batch_size or i == len(chunks) - 1:
                # Generate embeddings for the batch
                batch_embeddings = embedding_model.encode(documents).tolist()
                embeddings_list.extend(batch_embeddings)
                
                # Add batch to ChromaDB
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=batch_embeddings,
                    ids=ids
                )
                
                successful_embeddings += len(documents)
                
                print(f"✅ Processed {i + 1}/{len(chunks)} chunks... (Batch of {len(documents)})")
                
                # Reset batch
                documents = []
                metadatas = []
                embeddings_list = []
                ids = []
                
        except Exception as e:
            print(f"⚠️  Error processing chunk {i}: {e}")
            continue

    print(f"🎉 Successfully embedded {successful_embeddings}/{len(chunks)} chunks in ChromaDB.")
    
    # Print collection stats
    try:
        count = collection.count()
        print(f"📊 ChromaDB collection now contains {count} documents")
    except:
        print("📊 Collection stats unavailable")

def search_similar_chunks(query, n_results=5):
    """Utility function to search for similar chunks"""
    query_embedding = embedding_model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    return results

if __name__ == "__main__":
    create_embeddings()
    
    # Test search functionality
    print("\n🧪 Testing search functionality...")
    test_results = search_similar_chunks("function definition", n_results=2)
    if test_results['documents']:
        print("✅ Search test successful!")
        print(f"Found {len(test_results['documents'][0])} relevant chunks")
    else:
        print("❌ Search test failed - no documents found")