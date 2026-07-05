#!/usr/bin/env python3
"""Постоянный сервер ЛИЧНОСТИ: грузит 3 MLX-фасета ОДИН раз и держит тёплыми.
connectome шлёт POST /color {moment} -> {lead, disagree, reacts}. Живые прогоны больше
не перегружают модель каждый раз. Запуск: being/venv/bin/python being/persona_server.py"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import persona


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/reload":                       # после «сна» перечитать дообученные адаптеры
            persona._models.clear()
            for name, _ in persona.FACETS: persona._ensure(name)
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"reloaded":true}')
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        moment = json.loads(self.rfile.read(n) or b"{}").get("moment", "")
        lead, dis, reacts = persona._color_local(moment)
        out = json.dumps({"lead": lead, "disagree": dis, "reacts": reacts}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(out)
    def log_message(self, *a): pass


if __name__ == "__main__":
    print("persona server: warming 3 MLX facets...", flush=True)
    for name, _ in persona.FACETS:
        persona._ensure(name)
    print(f"persona server ready on :{persona.PORT}", flush=True)
    HTTPServer(("127.0.0.1", persona.PORT), H).serve_forever()
