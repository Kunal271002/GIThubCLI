import typer
from auth import Signup, Login, save_token
from github_fetch import fetch_repos

app = typer.Typer()

@app.command()
def signup():
    username = typer.prompt("Username")
    password = typer.prompt("Password", hide_input=True)
    Signup(username, password)

@app.command()
def login():
    username = typer.prompt("Username")
    password = typer.prompt("Password", hide_input=True)
    if Login(username, password):
        typer.echo("Login successful!")
        return username  # Return username for token setting
    else:
        typer.echo("Invalid username or password")
        return None

@app.command()
def set_token():
    username = typer.prompt("Username")
    token = typer.prompt("Enter your GitHub Personal Access Token", hide_input=True)
    save_token(username, token)

@app.command()
def fetch():
    username = typer.prompt("Username")
    fetch_repos(username)

if __name__ == "__main__":
    app()