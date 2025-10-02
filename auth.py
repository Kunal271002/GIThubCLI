import sqlite3
import hashlib
import os

# Ensure DB directory exists
os.makedirs("DB", exist_ok=True)

def get_db_connection():
    return sqlite3.connect("DB/users.db")

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        password_hash TEXT,
        github_token TEXT
    )
    """)
    conn.commit()
    conn.close()

# Initialize database on import
init_db()

def Signup(username: str, password: str):
    conn = get_db_connection()
    c = conn.cursor()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        print(f"User {username} created successfully!")
    except sqlite3.IntegrityError:
        print("Username already exists!")
    finally:
        conn.close()

def Login(username: str, password: str) -> bool:
    conn = get_db_connection()
    c = conn.cursor()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password_hash=?", (username, password_hash))
    result = c.fetchone() is not None
    conn.close()
    return result

def save_token(username: str, token: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE users SET github_token=? WHERE username=?", (token, username))
    conn.commit()
    print("GitHub token saved successfully!")
    conn.close()