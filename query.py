import typer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import chromadb
import torch
import os

app = typer.Typer()

# -----------------------
# Embedding Model - Using valid models
# -----------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# ChromaDB: Persistent storage
# -----------------------
try:
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="repo_chunks")
    print("✅ ChromaDB persistent collection loaded")
except Exception as e:
    print(f"❌ ChromaDB persistent error, using in-memory: {e}")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="repo_chunks")

TOP_K = 5

# -----------------------
# LLM Setup - Using smaller, more efficient models
# -----------------------
os.makedirs("offload", exist_ok=True)

try:
    print("🔄 Loading Mistral 7B model...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_folder="offload",
        trust_remote_code=True
    )
    
    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )
    print("✅ Mistral 7B loaded successfully!")
    
except Exception as e:
    print(f"❌ Could not load Mistral 7B: {e}")
    print("🔄 Trying smaller model...")
    
    try:
        print("🔄 Loading Microsoft DialoGPT-medium...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium",
            device_map="auto",
            torch_dtype=torch.float32,
            offload_folder="offload"
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )
        print("✅ DialoGPT-medium loaded successfully!")
        
    except Exception as e2:
        print(f"❌ Could not load DialoGPT-medium: {e2}")
        print("🔄 Loading GPT2 as fallback...")
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )
        print("✅ GPT2 loaded successfully!")

# -----------------------
# CLI Commands
# -----------------------
@app.command()
def query(user_query: str):
    """Query your code repository with RAG"""
    try:
        # Encode query
        query_embedding = embedding_model.encode(user_query).tolist()
        
        # Search in ChromaDB with metadata
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'] or not results['documents'][0]:
            print("❌ No relevant data found for the query.")
            return

        # Build context with source information
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            source_info = f"From {metadata.get('repo', 'unknown')}/{metadata.get('file', 'unknown')}"
            if metadata.get('function'):
                source_info += f" (function: {metadata['function']})"
            context_parts.append(f"{source_info}:\n{doc}")
        
        context_text = "\n\n".join(context_parts)

        # Format prompt based on model
        model_name = tokenizer.name_or_path.lower()
        
        if "mistral" in model_name:
            prompt = f"""<s>[INST] You are a helpful GitHub code assistant. Use the following code context from repositories to answer the question.

CODE CONTEXT:
{context_text}

QUESTION: {user_query}

Provide a concise and accurate answer based only on the code context above. If the context doesn't contain relevant information, say "I cannot find relevant information in the code context."

ANSWER: [/INST]"""
        else:
            prompt = f"""Code Context:
{context_text}

Question: {user_query}

Based on the code context above, please answer:"""

        # Generate answer
        print("🤖 Generating response...")
        response = llm(
            prompt, 
            max_new_tokens=128,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        answer = response[0]['generated_text']

        # Clean up response
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()
        elif prompt in answer:
            answer = answer.replace(prompt, "").strip()
        answer = answer.replace("</s>", "").strip()

        print("\n" + "="*60)
        print("🤖 ANSWER:")
        print("="*60)
        print(answer)
        print("\n" + "="*60)
        print("📚 Sources used:")
        for i, metadata in enumerate(results['metadatas'][0][:3]):
            print(f"  {i+1}. {metadata.get('repo', 'unknown')}/{metadata.get('file', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Error during query: {e}")

@app.command()
def test_models():
    """Test if models are working"""
    print("🧪 Testing embedding model...")
    test_embedding = embedding_model.encode("test")
    print(f"✅ Embedding model working (shape: {test_embedding.shape})")
    
    print("🧪 Testing LLM...")
    try:
        test_response = llm("Say 'test successful' in one word:", max_new_tokens=10)
        print(f"✅ LLM working: {test_response[0]['generated_text'][:50]}...")
    except Exception as e:
        print(f"❌ LLM error: {e}")

@app.command()
def model_info():
    """Show which models are currently loaded"""
    print(f"🔤 Embedding model: all-MiniLM-L6-v2")
    print(f"🤖 LLM tokenizer: {tokenizer.name_or_path}")
    print(f"🧠 LLM model: {model.config.name_or_path}")

@app.command()
def db_status():
    """Check ChromaDB collection status"""
    try:
        count = collection.count()
        print(f"📊 ChromaDB collection has {count} documents")
    except Exception as e:
        print(f"❌ Error checking database: {e}")

@app.command()
def examples():
    """Show example queries"""
    print("💡 Example queries:")
    examples = [
        "How many repositories do I have?",
        "What is the main function in app.py?",
        "Show me all the class definitions",
        "What dependencies are used?",
        "Explain the project structure"
    ]
    for i, example in enumerate(examples, 1):
        print(f"  {i}. python query.py query \"{example}\"")

if __name__ == "__main__":
    app()