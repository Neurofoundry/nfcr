import os
from http.server import BaseHTTPRequestHandler, HTTPServer

def agent_run():
    repo_url = os.environ.get("REPO_URL", "<not set>")
    print("Agent starting up...")
    print(f"REPO_URL seen: {repo_url}")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        agent_run()
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Agent ran successfully.\n")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Server listening on port {port}")
    server.serve_forever()
