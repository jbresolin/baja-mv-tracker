"""Baja MV launcher (AUTOPTZ-26): a small tray app + local web page that sets up and runs this laptop's role(s).

It only orchestrates the existing entry points; it never reimplements them:
  * Lap line  -> rt/supervise.py (which runs rt/run.py and restarts it), status page :8765
  * Zone PTZ  -> zone/supervisor.py (which runs zone.app and restarts it), page :8090, studio agent API :8890

Modules (each one small, read them in this order):
  paths.py        where the code, the bundled tools and the writable data folder are (repo mode vs installed mode)
  tomlwrite.py    a tiny TOML writer (Python's tomllib can only read)
  settings.py     the essential settings per role: defaults, validation, writing the TOML configs
  procs.py        one managed child process: start / stop / state / log file
  roles.py        the two roles: their commands, environment and the zone's safe stop
  rosterfetch.py  the "Baixar lista de equipes" button
  cameras.py      camera list and snapshot preview
  tools.py        the PTZ characterization tool in a visible console window
  exportlogs.py   "Exportar logs": a zip to send back
  server.py       the web page's HTTP API (FastAPI) and static/ (plain HTML + JS, pt-BR)
  tray.py         the tray icon (pystray), with a fallback when there is no tray
  selftest.py     `--self-test`: checks an installation end to end (used by CI)
  __main__.py     the command line: python -m launcher [--self-test] [--no-tray] [--port N] [--data-root DIR]
"""
