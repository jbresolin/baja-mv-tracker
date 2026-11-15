"""Event rosters (car number -> team) from the official results site.

The roster belongs to the event, not the prova: every prova of an event uses the same car numbers. Source is the
"Geral" page (<EV>_GER), which lists every registered car (26BR: 51 cars, vs 38 that ran the enduro). Falls back to
the enduro page when Geral is missing. Stdlib only.

usage: python teacher/rosters.py [--out-dir DIR] [EVENT ...]   (default: every event linked from the site's home page)
writes rosters/<EVENT>.csv (or DIR/<EVENT>.csv) with columns numero,equipe,escola,fonte
The installed app's launcher (launcher/rosterfetch.py) calls roster() and write_roster() directly.
"""
import csv
import html
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://resultados.bajasaebrasil.net"
OUT = Path(__file__).resolve().parent.parent / "rosters"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (baja lapvision roster fetch)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def events():
    return sorted(set(re.findall(r'href="/(\d\d[A-Z]{2})"', get(BASE + "/"))))


def text(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def parse(page):
    """Rows of the results table: [pos, number, team<br><i>school</i>, ...]."""
    body = page[page.find("<tbody"):page.find("</tbody")]
    out = []
    for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S):
        td = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(td) < 3 or not text(td[1]).isdigit():
            continue
        team, _, school = td[2].partition("<br")
        out.append(dict(numero=int(text(td[1])), equipe=text(team), escola=text("<" + school) if school else ""))
    return out


def roster(ev, get_page=None, sleep=time.sleep):
    """-> (rows, prova) for one event, or ([], None). get_page(url) -> html (default: get); tests pass fakes."""
    get_page = get_page or get
    for prova in ("GER", "END"):
        rows = parse(get_page("%s/%s/prova.php?id=%s_%s" % (BASE, ev, ev, prova)))
        if rows:
            return rows, prova
        sleep(1)
    return [], None


def write_roster(rows, src, path):
    """Write one roster CSV (one row per car number, sorted). Returns the number of cars written."""
    rows = sorted({r["numero"]: r for r in rows}.values(), key=lambda r: r["numero"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["numero", "equipe", "escola", "fonte"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, fonte=src))
    return len(rows)


def main():
    args = sys.argv[1:]
    out_dir = OUT
    if len(args) >= 2 and args[0] == "--out-dir":
        out_dir, args = Path(args[1]), args[2:]
    evs = args or events()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ev in evs:
        try:
            rows, src = roster(ev)
        except Exception as e:  # one bad event must not stop the rest
            print(ev, "ERROR", e, flush=True)
            continue
        if not rows:
            print(ev, "no roster", flush=True)
            continue
        n = write_roster(rows, src, out_dir / ("%s.csv" % ev))
        print(ev, src, n, "cars", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
