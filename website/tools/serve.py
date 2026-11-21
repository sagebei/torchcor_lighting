"""Serve the website to your local network, with byte-range support for the videos.

    python3 tools/serve.py           # http://<this computer's IP>:8000
    python3 tools/serve.py 8080      # another port

Python's built-in `http.server` ignores Range requests, so Safari (macOS and iPhone)
won't play the videos from it; this server answers them with 206 Partial Content.
It is for sharing on a trusted network. For a permanent public site, use a real host
(see ../README.md).
"""
import os
import re
import socket
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the website folder
RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class Handler(SimpleHTTPRequestHandler):
    """Static files plus single byte ranges (`Range: bytes=a-b`, `a-`, `-n`)."""

    _left = None   # bytes still to send for a partial response

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")   # browsers re-check, so edits show at once
        super().end_headers()

    def send_head(self):
        match = RANGE.fullmatch(self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not match or match.group(1) == match.group(2) == "" or os.path.isdir(path):
            return super().send_head()              # whole file, directory or no/odd range
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        stat = os.fstat(f.fileno())
        size = stat.st_size
        first, last = match.groups()
        if first == "":                             # suffix: the last N bytes
            start, end = max(0, size - int(last)), size - 1
        else:
            start, end = int(first), min(int(last) if last else size - 1, size - 1)
        if start >= size or start > end:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.end_headers()
        f.seek(start)
        self._left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        if self._left is None:
            return super().copyfile(source, outputfile)
        while self._left > 0:
            chunk = source.read(min(64 * 1024, self._left))
            if not chunk:
                break
            outputfile.write(chunk)
            self._left -= len(chunk)


def lan_ip():
    """This computer's address on the local network (no packets are sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    with ThreadingHTTPServer(("0.0.0.0", port), partial(Handler, directory=str(ROOT))) as httpd:
        print(f"Serving {ROOT}")
        print(f"  on your network:  http://{lan_ip()}:{port}/")
        print(f"  on this computer: http://localhost:{port}/")
        print("Anyone on the same network can open the first address. Stop with Ctrl-C.", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
