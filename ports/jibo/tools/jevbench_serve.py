#!/usr/bin/env python3
"""Stand in for `strands-decider serve` so evaluation/jevbench/jevbench.sh scores the C runtime.

    HOBSON=ports/jibo/tools/jevbench_serve.py JIBO_EXPORT=out/v22-q4 JIBO_BIN=build/jibo-decider \\
      PY=python evaluation/jevbench/jevbench.sh CHECKPOINT OUT_DIR

jevbench.sh runs `$HOBSON serve CHECKPOINT --host H --port P` and checks that /health names that
checkpoint and its window. This starts `jibo-decider serve` on a private Unix socket with the
export in JIBO_EXPORT (model.jdw, tokenizer.jdt) and answers HTTP on H:P: POST /v1/systemone is
forwarded to the socket, GET /health reports CHECKPOINT, the export and the window. JIBO_WINDOW
sets the window (default: the checkpoint's max_length, as the Python server uses).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["serve"])
    ap.add_argument("checkpoint")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args, _ = ap.parse_known_args()
    export = os.environ["JIBO_EXPORT"]
    binary = os.environ["JIBO_BIN"]
    cfg_path = os.path.join(args.checkpoint, "strands_decider_config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(args.checkpoint, "hobson_config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        window = int(os.environ.get("JIBO_WINDOW") or json.load(fh)["max_length"])
    sock = os.path.join(tempfile.mkdtemp(prefix="jibo-"), "decider.sock")
    proc = subprocess.Popen([binary, "serve", os.path.join(export, "model.jdw"),
                             os.path.join(export, "tokenizer.jdt"), "--socket", sock,
                             "--window", str(window), "--model-name", os.path.basename(args.checkpoint.rstrip("/"))])
    for _ in range(600):
        if os.path.exists(sock):
            break
        time.sleep(0.1)

    def ask(payload: bytes) -> bytes:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(sock)
            s.sendall(payload)
            s.shutdown(socket.SHUT_WR)
            chunks = []
            while chunk := s.recv(65536):
                chunks.append(chunk)
        return b"".join(chunks).strip()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send(404, b"{}")
                return
            h = json.loads(ask(b'{"health": true}'))
            h.update({"checkpoint": args.checkpoint, "max_length": window, "export": export,
                      "device": "cpu (jibo-decider)"})
            self._send(200, json.dumps(h).encode())

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/systemone":
                self._send(404, b"{}")
                return
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            out = ask(body)
            self._send(422 if out.startswith(b'{"detail"') else 200, out)

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    finally:
        proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
