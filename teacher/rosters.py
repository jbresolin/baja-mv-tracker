"""Event rosters (car number -> team) from the official results site.

The roster belongs to the event, not the prova: every prova of an event uses the same car numbers. Source is the
"Geral" page (<EV>_GER), which lists every registered car (26BR: 51 cars, vs 38 that ran the enduro). Falls back to
the enduro page when Geral is missing. Stdlib only.

usage: python teacher/rosters.py [EVENT ...]      (default: every event linked from the site's home page)
writes rosters/<EVENT>.csv with columns numero,equipe,escola,fonte
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


def roster(ev):
    for prova in ("GER", "END"):
        rows = parse(get("%s/%s/prova.php?id=%s_%s" % (BASE, ev, ev, prova)))
        if rows:
            return rows, prova
        time.sleep(1)
    return [], None


def main():
    evs = sys.argv[1:] or events()
    OUT.mkdir(exist_ok=True)
    for ev in evs:
        try:
            rows, src = roster(ev)
        except Exception as e:  # one bad event must not stop the rest
            print(ev, "ERROR", e, flush=True)
            continue
        if not rows:
            print(ev, "no roster", flush=True)
            continue
        rows = sorted({r["numero"]: r for r in rows}.values(), key=lambda r: r["numero"])
        with open(OUT / ("%s.csv" % ev), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, ["numero", "equipe", "escola", "fonte"])
            w.writeheader()
            for r in rows:
                w.writerow(dict(r, fonte=src))
        print(ev, src, len(rows), "cars", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
