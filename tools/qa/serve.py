"""QA review server: one page, keyboard verdicts, every verdict appended to <batch>.verdicts.jsonl at once.

usage (from the repo root):  python tools/qa/serve.py <batch.jsonl|.parquet> [--port 8765] [--root .] [--reviewer NAME]
                                                     [--host 127.0.0.1|0.0.0.0] [--token T]
then open http://127.0.0.1:8765/ . Stdlib only. Binds to 127.0.0.1 by default.
With --host 0.0.0.0 (review from another device on the LAN) a random access token is required: open the printed
http://<lan-ip>:<port>/?t=<token> once; it sets a cookie, and requests without it get 403.

Endpoints
  GET  /                  the review page (tools/qa/index.html)
  GET  /api/batch         {name, items (meta.hidden stripped), verdicts {item_id: record}, rosters {EV: {num: team}}}
  POST /api/verdict       {item_id, verdict: correct|wrong|not_car|unsure|undo, number?, boxes_wrong?, dt_ms?}
  GET  /img/<path>        an image under --root (crop / context paths in the batch are relative to --root)
"""
import argparse
import hmac
import json
import mimetypes
import os
import secrets
import socket
import sys
import threading
from datetime import datetime
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qalib  # noqa: E402

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


class State:
    def __init__(self, batch, root, reviewer):
        self.batch = Path(batch).resolve()
        self.root = Path(root).resolve()
        self.reviewer = reviewer
        self.items = qalib.load_batch(self.batch)
        self.ids = {it["item_id"] for it in self.items}
        self.vpath = qalib.verdicts_path(self.batch)
        self.lock = threading.Lock()
        events = sorted({it["event"] for it in self.items})
        self.rosters = {ev: qalib.load_roster(self.root, ev) for ev in events}
        for ev, r in self.rosters.items():
            if not r:
                print(f"warning: no roster rosters/{ev}.csv under {self.root}", file=sys.stderr)

    def public_items(self):
        out = []
        for it in self.items:
            it = dict(it)
            meta = dict(it.get("meta") or {})
            meta.pop("hidden", None)          # GT / anything the reviewer must not see
            it["meta"] = meta
            out.append(it)
        return out

    def append(self, rec):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self.lock:
            with open(self.vpath, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            return len(qalib.load_verdicts(self.batch))


class Handler(BaseHTTPRequestHandler):
    S: State = None
    token: str = None   # set when serving beyond localhost

    def log_message(self, fmt, *args):   # quiet: only errors
        if args and str(args[1])[:1] in "45":
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def authed(self):
        """No token -> open (localhost only). Else the ?t= query (which also sets the cookie) or the cookie."""
        self.set_cookie = False
        if not self.token:
            return True
        q = parse_qs(urlparse(self.path).query).get("t", [""])[0]
        if q and hmac.compare_digest(q, self.token):
            self.set_cookie = True
            return True
        c = SimpleCookie(self.headers.get("Cookie", ""))
        return "qa_t" in c and hmac.compare_digest(c["qa_t"].value, self.token)

    def send(self, code, body, ctype="application/json; charset=utf-8", cache=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", ("private, max-age=3600" if self.token else "max-age=3600") if cache
                         else "no-store")
        if getattr(self, "set_cookie", False):
            self.send_header("Set-Cookie", f"qa_t={self.token}; Path=/; HttpOnly; SameSite=Strict")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.authed():
            return self.send(403, "forbidden: open the URL with ?t=<token> printed by serve.py\n",
                             "text/plain; charset=utf-8")
        path = urlparse(self.path).path
        S = self.S
        if path in ("/", "/index.html"):
            return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/batch":
            return self.send(200, dict(name=S.batch.name, reviewer=S.reviewer, items=S.public_items(),
                                       verdicts=qalib.load_verdicts(S.batch), rosters=S.rosters))
        if path.startswith("/img/"):
            rel = unquote(path[len("/img/"):])
            p = (S.root / rel).resolve()
            if not p.is_relative_to(S.root) or p.suffix.lower() not in IMG_EXT or not p.is_file():
                return self.send(404, {"error": "not found"})
            return self.send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream",
                             cache=True)
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.authed():
            return self.send(403, {"ok": False, "error": "forbidden"})
        if urlparse(self.path).path != "/api/verdict":
            return self.send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            r = json.loads(self.rfile.read(n) or b"{}")
            iid, v = r.get("item_id"), r.get("verdict")
            if iid not in self.S.ids:
                raise ValueError(f"unknown item_id {iid!r}")
            if v not in qalib.VERDICTS + ("undo",):
                raise ValueError(f"bad verdict {v!r}")
            num = r.get("number")
            if num not in (None, ""):
                num = int(num)
            else:
                num = None
            rec = dict(item_id=iid, verdict=v, number=num if v == "wrong" else None,
                       boxes_wrong=bool(r.get("boxes_wrong")) if v != "undo" else None,
                       dt_ms=r.get("dt_ms"), reviewer=self.S.reviewer,
                       ts=datetime.now().astimezone().isoformat(timespec="milliseconds"))
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return self.send(400, {"ok": False, "error": str(e)})
        n_rev = self.S.append(rec)
        return self.send(200, {"ok": True, "n_reviewed": n_rev, "record": rec})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=".", help="directory that crop paths are relative to (default: cwd = repo root)")
    ap.add_argument("--reviewer", default=os.environ.get("USERNAME") or os.environ.get("USER") or "")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to review from another device (token required)")
    ap.add_argument("--token", default=None, help="access token when --host is not localhost (default: random)")
    a = ap.parse_args()
    local = a.host in ("127.0.0.1", "localhost", "::1")
    Handler.token = None if local else (a.token or secrets.token_urlsafe(9))
    Handler.S = State(a.batch, a.root, a.reviewer)
    S = Handler.S
    missing = [c for it in S.items[:20] for c in it.get("crops", []) if not (S.root / c).is_file()]
    if missing:
        print(f"warning: {len(missing)} crops of the first 20 items not found under {S.root}, e.g. {missing[0]}",
              file=sys.stderr)
    done = len(qalib.load_verdicts(S.batch))
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"{S.batch.name}: {len(S.items)} items, {done} already reviewed; verdicts -> {S.vpath}")
    if local:
        print(f"open http://127.0.0.1:{a.port}/   (Ctrl+C to stop)", flush=True)
    else:
        try:   # the address this PC uses to reach the LAN (no packet is sent)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("192.0.2.1", 9))
            ip = s.getsockname()[0]
            s.close()
        except OSError:
            ip = socket.gethostbyname(socket.gethostname())
        print(f"open http://{ip}:{a.port}/?t={Handler.token}   from any device on the LAN (Ctrl+C to stop)",
              flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
