"""Local static server for the draft page.   python3 serve.py [port]

Sends COOP/COEP headers so the page is cross-origin isolated, which lets onnxruntime-web use
WebAssembly threads (COEP "credentialless" still allows Scryfall card images). Browsers without
credentialless support (Safari) fall back to one thread.
"""
import http.server
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
                      ".onnx": "application/octet-stream", ".json": "application/json"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"http://localhost:{port}")
    http.server.ThreadingHTTPServer(("", port), Handler).serve_forever()
