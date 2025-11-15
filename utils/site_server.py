import http.server
import socketserver

PORT = 8083
DIRECTORY = "site"

"""
Configures the Python http server to serve the sites folder without caching.
"""


class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def run_server():
    with socketserver.TCPServer(("", PORT), NoCacheHTTPRequestHandler) as server:
        print("To rebuild the docs run:")
        print(" - task build-docs")
        print("Or live-reload the docs with: ")
        print(" - task serve-docs\n")
        print(f"Serving '{DIRECTORY}' directory at http://localhost:{PORT}\n")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.socket.close()


if __name__ == "__main__":
    run_server()
