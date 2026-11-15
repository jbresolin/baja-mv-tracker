"""Baja MV launcher: the tray app + page, the self-test and the installer's shutdown request.

usage:
  python -m launcher                     tray icon + page on http://<this PC>:8780/ (the Start-menu entry runs this
                                         with pythonw.exe: no console window)
  python -m launcher --no-tray           page only (e.g. in a terminal, or a PC without a tray)
  python -m launcher --self-test         check this installation end to end; exit code 0 = all good (CI uses it)
  python -m launcher --shutdown          ask a running launcher to stop both roles and exit (installer, upgrades)
options: --port N (default 8780), --data-root DIR (default: the installer's choice, C:\\BajaMV, or out/bajamv in a
  repo checkout), --no-browser (don't open the page at start when there is no tray icon)
"""
import argparse
import datetime as dt
import json
import sys
import threading
import urllib.request
import webbrowser

from launcher.paths import DEFAULT_LAUNCHER_PORT, resolve_paths


def ping(port, timeout=1.0):
    """The answer of a launcher already running on this port, or None."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/api/ping" % port, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data if data.get("app") == "bajamv-launcher" else None
    except Exception:  # noqa: BLE001
        return None


def request_shutdown(port, timeout=30.0):
    """--shutdown: ask the running launcher to stop the roles and exit. 0 = done or nothing was running."""
    if ping(port) is None:
        print("nenhum Baja MV rodando na porta %d" % port)
        return 0
    request = urllib.request.Request("http://127.0.0.1:%d/api/shutdown" % port, data=b"", method="POST")
    try:
        urllib.request.urlopen(request, timeout=timeout).read()
    except Exception as e:  # noqa: BLE001
        print("pedido de parada falhou: %s" % e)
        return 1
    # wait until it is gone (it stops the roles first)
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and ping(port, timeout=0.5) is not None:
        time.sleep(0.5)
    print("Baja MV parado")
    return 0


def redirect_output_to_log(paths):
    """Under pythonw.exe there is no console (sys.stdout is None): send prints and tracebacks to launcher.log."""
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    stream = open(paths.logs_dir / "launcher.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream


def log(message):
    print("[%s] %s" % (dt.datetime.now().isoformat(timespec="seconds"), message), flush=True)


def serve(launcher, port):
    """The page's web server in a background thread. Returns the uvicorn server (to stop it)."""
    import uvicorn
    from launcher.server import create_app
    server = uvicorn.Server(uvicorn.Config(create_app(launcher), host="0.0.0.0", port=port, log_level="warning",
                                           log_config=None))
    thread = threading.Thread(target=server.run, daemon=True, name="launcher-web")
    thread.start()
    return server, thread


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m launcher", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=DEFAULT_LAUNCHER_PORT)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--no-tray", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--keep", action="store_true", help="--self-test: keep its temporary folder")
    ap.add_argument("--shutdown", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        from launcher.selftest import run_self_test
        return run_self_test(keep=args.keep)
    if args.shutdown:
        return request_shutdown(args.port)

    paths = resolve_paths(args.data_root)
    paths.make_folders()
    if sys.stdout is None:
        redirect_output_to_log(paths)
    url = "http://127.0.0.1:%d/" % args.port
    if ping(args.port) is not None:                   # already running: just show it
        log("Baja MV já está rodando: abrindo %s" % url)
        webbrowser.open(url)
        return 0

    from launcher.server import Launcher
    launcher = Launcher(paths, args.port, log=log)
    log("Baja MV %s, dados em %s, painel em http://<este PC>:%d/" % (paths.version, paths.data_root, args.port))
    launcher.roles.adopt_running()
    server, thread = serve(launcher, args.port)
    launcher.start_autostart_roles()

    shown = False
    if not args.no_tray:
        shown = run_tray_or_false(launcher, url)
    if not shown:
        if not args.no_browser:
            webbrowser.open(url)
        try:
            while not launcher.exit_requested.wait(1.0):
                if not thread.is_alive():
                    log("o servidor do painel parou (porta %d ocupada?)" % args.port)
                    break
        except KeyboardInterrupt:
            log("Ctrl+C: o painel fecha, os papéis continuam rodando")
    server.should_exit = True
    thread.join(timeout=5)
    log("Baja MV fechado")
    return 0


def run_tray_or_false(launcher, url):
    from launcher.tray import run_tray
    return run_tray(launcher, url, webbrowser.open, log=log)


if __name__ == "__main__":
    sys.exit(main())
