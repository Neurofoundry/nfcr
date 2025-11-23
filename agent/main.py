import os

def main():
    repo_url = os.environ.get("REPO_URL", "<not set>")
    print("Agent starting up...")
    print(f"REPO_URL seen by agent: {repo_url}")

if __name__ == "__main__":
    main()
