#!/usr/bin/env python3
"""
Simple HTTP server for the Graph RAG frontend.
Serves static files and handles CORS for API requests.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 3000

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support."""
    
    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight OPTIONS requests."""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[Frontend] {self.address_string()} - {format % args}")


def main():
    """Start the frontend server."""
    # Change to frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    # Create server
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║  📚 Graph RAG Query System - Frontend Server              ║
║                                                            ║
║  Server running at: http://localhost:{PORT}                ║
║                                                            ║
║  Make sure the backend API is running at:                 ║
║  http://localhost:8000                                     ║
║                                                            ║
║  Press Ctrl+C to stop the server                          ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
        """)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n[Frontend] Server stopped by user")
            httpd.shutdown()


if __name__ == "__main__":
    main()
