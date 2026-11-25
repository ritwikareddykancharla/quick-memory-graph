"""HTTP Server & AWS CloudFront `/api/*` Origin Handler for QuickMemory."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, urlparse

from .graph_engine import QuickMemoryEngine

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"

ENGINE = QuickMemoryEngine()

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    """Pure router shared by local HTTP server and AWS Lambda `/api/*` origin behind CloudFront."""
    try:
        clean_path = path.rstrip("/") if path != "/" else "/"

        if method == "GET" and clean_path == "/api/health":
            counts = ENGINE.health_summary()
            return 200, {
                "status": "ok",
                "service": "quick-memory-graph",
                "delivery": os.environ.get("QUICKMEMORY_DELIVERY", "cloudfront-s3-oac + api-origin"),
                "aws_region": os.environ.get("AWS_REGION", "us-west-2"),
                "bedrock_model": os.environ.get("QUICKMEMORY_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"),
                "python_version": platform.python_version(),
                "request_id": request_id,
                **counts,
            }

        if method == "GET" and clean_path == "/api/state":
            persona = query.get("persona") or "principal_pm"
            snap = ENGINE.get_state(persona_id=persona)
            snap["request_id"] = request_id
            return 200, snap

        if method == "POST" and clean_path == "/api/sessions/ingest":
            persona = str(body.get("persona") or query.get("persona") or "principal_pm")
            session_title = str(body.get("session_title") or "Live Contextual Work Session").strip()
            transcript = str(body.get("transcript") or "").strip()
            if not transcript:
                return 400, {"error": "transcript is required"}
            out = ENGINE.ingest_session(
                persona_id=persona,
                session_title=session_title,
                transcript=transcript,
            )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and clean_path == "/api/companion/ask":
            persona = str(body.get("persona") or query.get("persona") or "principal_pm")
            prompt = str(body.get("prompt") or "").strip()
            if not prompt:
                return 400, {"error": "prompt is required"}
            out = ENGINE.ask_companion(persona_id=persona, prompt=prompt)
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and clean_path == "/api/memory/toggle":
            persona = str(body.get("persona") or query.get("persona") or "principal_pm")
            memory_id = str(body.get("memory_id") or "").strip()
            action = str(body.get("action") or "").strip()
            if not memory_id or not action:
                return 400, {"error": "Both memory_id and action (pin|forget|restore) are required"}
            out = ENGINE.toggle_memory(persona_id=persona, memory_id=memory_id, action=action)
            out["request_id"] = request_id
            return 200, out

        if method in ("POST", "DELETE") and clean_path == "/api/memory/forget":
            persona = str(body.get("persona") or query.get("persona") or "principal_pm")
            memory_id = str(body.get("memory_id") or query.get("memory_id") or "").strip()
            if not memory_id:
                return 400, {"error": "memory_id is required"}
            out = ENGINE.toggle_memory(persona_id=persona, memory_id=memory_id, action="forget")
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and clean_path == "/api/reset":
            ENGINE.reset()
            persona = str(body.get("persona") or query.get("persona") or "principal_pm")
            snap = ENGINE.get_state(persona_id=persona)
            return 200, {"status": "reset_complete", **snap, "request_id": request_id}

        return 404, {"error": f"Unknown API route: {method} {path}"}
    except KeyError as exc:
        return 404, {"error": str(exc)}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda handler mounted behind Amazon CloudFront `/api/*` origin."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    query_params = {
        k: str(v)
        for k, v in (event.get("queryStringParameters") or {}).items()
        if v is not None
    }
    req_id = getattr(context, "aws_request_id", "cloudfront-api")

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if raw_path.startswith("/api/"):
        body: Dict[str, Any] = {}
        raw_body = event.get("body")
        if raw_body:
            if event.get("isBase64Encoded") and isinstance(raw_body, str):
                raw_body = base64.b64decode(raw_body).decode("utf-8")
            body = json.loads(raw_body) if isinstance(raw_body, str) else dict(raw_body)
        status, payload = route_api_request(method, raw_path, query_params, body, request_id=req_id)
        return {
            "statusCode": status,
            "headers": {
                **CORS_HEADERS,
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
            "body": json.dumps(payload, default=str),
        }

    # Diagnostic static fallback if direct Function URL is opened
    target = WEB_DIR / ("index.html" if raw_path in ("/", "") else raw_path.lstrip("/"))
    if not target.is_file() and raw_path.endswith(".svg"):
        target = DOCS_DIR / Path(raw_path).name
    if target.is_file():
        ctype = mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8"
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": ctype},
            "body": target.read_text(encoding="utf-8"),
        }
    return {
        "statusCode": 404,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps({"error": "Not found", "path": raw_path}),
    }


class LocalQuickMemoryHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}

        if path.startswith("/api/"):
            status, payload = route_api_request("GET", path, query, {}, request_id="local-dev")
            self._send_json(status, payload)
            return

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        candidate = (WEB_DIR / rel).resolve()
        if not candidate.is_file() and path.endswith(".svg"):
            candidate = (DOCS_DIR / Path(path).name).resolve()
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"

        data = candidate.read_bytes()
        ctype = mimetypes.guess_type(str(candidate))[0] or "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, query, body, request_id="local-dev")
        self._send_json(status, payload)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("DELETE", parsed.path, query, body, request_id="local-dev")
        self._send_json(status, payload)


def run_local_server(port: int = 3096) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), LocalQuickMemoryHandler)
    print(f"QuickMemory live studio running at http://localhost:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "3096"))
    run_local_server(port=port_num)
