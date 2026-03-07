import os
import re

REPO_DIR = "/home/ubuntu/football-prediction-xgboost"
API_KEY = "c1b837c37df33c46d475f5a67c346c22"

def sanitize_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the specific API key with os.getenv
    # We look for the pattern API_KEY = "..." or '...'
    # and replace it with API_KEY = os.getenv("FOOTBALL_API_KEY")
    
    # First, ensure os is imported if we're going to use os.getenv
    if API_KEY in content and "import os" not in content:
        content = "import os\n" + content
        
    new_content = content.replace(f'"{API_KEY}"', 'os.getenv("FOOTBALL_API_KEY")')
    new_content = new_content.replace(f"'{API_KEY}'", 'os.getenv("FOOTBALL_API_KEY")')
    
    # Also look for the headers pattern if it's hardcoded
    new_content = new_content.replace(f"'x-apisports-key': '{API_KEY}'", "'x-apisports-key': os.getenv('FOOTBALL_API_KEY')")
    new_content = new_content.replace(f'"x-apisports-key": "{API_KEY}"', '"x-apisports-key": os.getenv("FOOTBALL_API_KEY")')

    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Sanitized: {file_path}")

for root, dirs, files in os.walk(REPO_DIR):
    for file in files:
        if file.endswith(".py"):
            sanitize_file(os.path.join(root, file))

# Create .env.example
with open(os.path.join(REPO_DIR, ".env.example"), "w") as f:
    f.write("FOOTBALL_API_KEY=your_api_key_here\n")

# Create .gitignore
with open(os.path.join(REPO_DIR, ".gitignore"), "w") as f:
    f.write(".env\n__pycache__/\n*.pyc\n.DS_Store\nraw_fixtures_*.json\n")

print("Sanitization complete.")
