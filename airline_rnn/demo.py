from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .analysis import AnalysisRunner

from .inference import SentimentPredictor

WEB = Path(__file__).with_name("web")
PAGE = (WEB / "index.html").read_text(encoding="utf-8")
ASSETS = {"/assets/app.js": ("app.js", "application/javascript; charset=utf-8"),
          "/assets/style.css": ("style.css", "text/css; charset=utf-8"),
          "/assets/school-logo.png": ("school-logo.png", "image/png")}



def serve(root=None, port=8765):
    predictor = SentimentPredictor(root)
    runner = AnalysisRunner(predictor)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value, mime="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else value.encode("utf-8") if isinstance(value, str) else json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/":
                self.respond(200, PAGE, "text/html; charset=utf-8")
            elif route == "/api/health":
                self.respond(200, {"status": "ready", "model": "RNN",
                                  "config_id": predictor.manifest["config_id"],
                                  "seed": predictor.manifest["seed"],
                                  "vocabulary_size": predictor.tokenizer.vocabulary_size,
                                  "sequence_length": predictor.representation["sequence_length"],
                                  "embedding_dimension": predictor.config["embedding_dimension"],
                                  "hidden_units": predictor.config["hidden_units"],
                                  "dense_units": predictor.config["dense_units"],
                                  "label_names": predictor.names,
                                  "parameters": predictor.model.count_params()})
            elif route == "/api/project":
                self.respond(200, json.loads((WEB / "project-info.json").read_text(encoding="utf-8")))
            elif route in ASSETS:
                filename, mime = ASSETS[route]
                self.respond(200, (WEB / filename).read_bytes(), mime)
            else:
                self.respond(404, {"error": "Không tìm thấy trang."})

        def do_POST(self):
            route = urlsplit(self.path).path
            if route not in ("/api/predict", "/api/analyze"):
                return self.respond(404, {"error": "Không tìm thấy endpoint."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32000:
                    raise ValueError("Request không hợp lệ hoặc quá dài.")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Request phải là JSON object.")
                text = request.get("text")
                runner.validate(text)
                if route == "/api/predict":
                    return self.respond(200, predictor.predict(text))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                return self.respond(400, {"error": str(error)})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for event in runner.events(text):
                    self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser cancelled this request; do not send another response
            except Exception as error:
                self.log_error("Analysis failed: %s", error)
                try:
                    self.wfile.write((json.dumps({"type": "error", "error": "Mô hình không hoàn tất phân tích. Hãy thử lại."}, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Demo ready: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
