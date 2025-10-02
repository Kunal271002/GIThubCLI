import os
import json
import re

CHUNK_SIZE = 500  # Characters instead of lines for better code handling
OVERLAP = 50  # Character overlap between chunks

def chunk_repo(repo_json_path):
    with open(repo_json_path, "r", encoding="utf-8") as f:
        repo = json.load(f)
    
    chunks = []
    
    for file in repo["files"]:
        path = file["path"]
        content = file["content"]
        
        # Skip very small files
        if len(content) < 50:
            continue
            
        # Determine file type for better chunking
        file_ext = os.path.splitext(path)[1].lower()
        
        if file_ext in ['.py', '.js', '.java', '.cpp', '.c']:
            # For code files, try to chunk by functions/classes
            chunks.extend(chunk_code_file(content, path, repo["repo"]))
        else:
            # For other files, use semantic chunking
            chunks.extend(chunk_by_size(content, path, repo["repo"]))
    
    return chunks

def chunk_code_file(content, file_path, repo_name):
    """Chunk code files by functions and classes when possible"""
    chunks = []
    
    # Try to find function and class definitions
    patterns = [
        (r'def (\w+)\([^)]*\):', 'function'),
        (r'class (\w+)\([^)]*\):', 'class'),
        (r'function (\w+)\([^)]*\)', 'function'),
        (r'const (\w+)\s*=', 'variable'),
        (r'let (\w+)\s*=', 'variable'),
        (r'var (\w+)\s*=', 'variable'),
    ]
    
    lines = content.split('\n')
    current_chunk = []
    current_context = {"function": "", "class": ""}
    
    for i, line in enumerate(lines):
        current_chunk.append(line)
        
        # Check for structural elements
        for pattern, element_type in patterns:
            match = re.search(pattern, line.strip())
            if match:
                if element_type == 'function':
                    current_context["function"] = match.group(1)
                elif element_type == 'class':
                    current_context["class"] = match.group(1)
                    current_context["function"] = ""  # Reset function when new class
        
        # Chunk when we hit a reasonable size or structural boundary
        if len('\n'.join(current_chunk)) >= CHUNK_SIZE or i == len(lines)-1:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text.strip()) > 50:  # Only save substantial chunks
                chunks.append({
                    "repo": repo_name,
                    "file": file_path,
                    "function": current_context["function"],
                    "class": current_context["class"],
                    "content": chunk_text
                })
            current_chunk = []
    
    return chunks

def chunk_by_size(content, file_path, repo_name):
    """Fallback chunking by size for non-code files"""
    chunks = []
    start = 0
    
    while start < len(content):
        end = start + CHUNK_SIZE
        # Try to break at sentence end or newline
        if end < len(content):
            while end > start and content[end] not in ['\n', '.', '!', '?']:
                end -= 1
            if end == start:  # No good break point found
                end = start + CHUNK_SIZE
        
        chunk_text = content[start:end].strip()
        if len(chunk_text) > 50:  # Only save substantial chunks
            chunks.append({
                "repo": repo_name,
                "file": file_path,
                "function": "",
                "class": "",
                "content": chunk_text
            })
        
        start = end - OVERLAP if end - OVERLAP > start else end
    
    return chunks

def chunk_all_repos(json_folder="repos_json"):
    all_chunks = []
    if not os.path.exists(json_folder):
        print(f"❌ Directory {json_folder} not found. Fetch repositories first.")
        return all_chunks
        
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            repo_path = os.path.join(json_folder, filename)
            try:
                repo_chunks = chunk_repo(repo_path)
                all_chunks.extend(repo_chunks)
                print(f"✅ Processed {filename}: {len(repo_chunks)} chunks")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
    
    print(f"🎉 Total chunks created: {len(all_chunks)}")
    
    # Save all chunks for inspection
    with open("all_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    return all_chunks

if __name__ == "__main__":
    chunks = chunk_all_repos()