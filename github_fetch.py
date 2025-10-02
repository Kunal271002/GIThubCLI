from github import Github
import json
import os
import sqlite3
from auth import get_db_connection

def fetch_repos(username: str):
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get token from DB
    c.execute("SELECT github_token FROM users WHERE username=?", (username,))
    token_row = c.fetchone()
    conn.close()
    
    if not token_row or not token_row[0]:
        print("❌ GitHub token not found. Please set token first using 'set_token' command.")
        return

    token = token_row[0]

    try:
        g = Github(token)
        user = g.get_user()
        repos = user.get_repos()

        os.makedirs("repos_json", exist_ok=True)
        repo_count = 0

        for repo in repos:
            print(f"📦 Fetching: {repo.name}")
            repo_data = {"repo": repo.name, "files": []}
            
            try:
                contents = repo.get_contents("")
                while contents:
                    file_content = contents.pop(0)
                    if file_content.type == "dir":
                        contents.extend(repo.get_contents(file_content.path))
                    elif file_content.type == "file":
                        # Skip large files and binary files
                        if file_content.size > 1000000:  # 1MB limit
                            continue
                        try:
                            content_data = file_content.decoded_content.decode('utf-8')
                            repo_data["files"].append({
                                "path": file_content.path,
                                "content": content_data,
                                "size": file_content.size
                            })
                        except (UnicodeDecodeError, AttributeError):
                            continue  # Skip binary files
            except Exception as e:
                print(f"⚠️  Could not process {repo.name}: {e}")
                continue

            # Save repo as JSON
            with open(f"repos_json/{repo.name}.json", "w", encoding="utf-8") as f:
                json.dump(repo_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {repo.name}.json")
            repo_count += 1

        print(f"🎉 Successfully fetched {repo_count} repositories!")

    except Exception as e:
        print(f"❌ Error fetching repositories: {e}")