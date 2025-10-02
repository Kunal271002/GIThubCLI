from sentence_transformers import SentenceTransformer
import chromadb

print("🧪 SUPER SIMPLE TEST")

# Just test embeddings and search
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="repo_chunks")

query = "In which repository there is hello.py"
print(f"Query: {query}")

# Test embedding
embedding = embedding_model.encode(query).tolist()
print("✅ Embedding created")

# Test search
results = collection.query(query_embeddings=[embedding], n_results=3)
print(results['documents'],'Hello')
print(f"✅ Search done. Found {len(results['documents'][0])} results")

if results['documents'][0]:
    print("📄 First result:")
    print(results['documents'][0][0][:] + "...")
else:
    print("❌ No results found - need to create embeddings first")