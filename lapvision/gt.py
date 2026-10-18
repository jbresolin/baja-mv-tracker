"""Ground truth loader: parses the MySQL dump (no schema present, columns inferred).

contagem_passagem columns (schema provided by user):
  0 passagem_id, 1 fonte_id (1..3 = human counters, 4 = box), 2 dispositivo_id,
  3 uuid_cliente, 4 equipe_numero (as typed), 5 ts_ms (THE pass time, server
  clock), 6 qualidade_tempo, 7 ts_dispositivo_ms (audit only, ~7s off),
  8 ancora_servidor_ms, 9 ancora_rtt_ms, 10 ancora_idade_ms, 11 digitacao_ms,
  12 modo_entrada, 13 recebido_em_ms
contagem_volta columns (inferred):
  prova, evento, car, lap, ts_ms, dispersion_ms, n_sources, lap_ms, ?, status
"""
import re
import json
import datetime as dt
from pathlib import Path

SQL = Path(__file__).resolve().parent.parent / "lap_count_dump_26se.sql"
TZ = dt.timezone(dt.timedelta(hours=-3))

_tok = re.compile(r"'((?:[^'\\]|\\.)*)'|(NULL)|(-?\d+(?:\.\d+)?)|(\()|(\))|,|;")


def _rows(table):
    txt = SQL.read_text(encoding="utf-8")
    m = re.search(r"INSERT INTO `%s` VALUES (.*?);\n" % table, txt, re.S)
    if not m:
        return []
    body = m.group(1)
    rows, cur, i = [], None, 0
    while i < len(body):
        mm = _tok.match(body, i)
        if not mm:
            i += 1
            continue
        s, null, num, op, cp = mm.groups()
        if op:
            cur = []
        elif cp:
            rows.append(cur)
            cur = None
        elif s is not None:
            cur.append(s.encode().decode("unicode_escape") if "\\" in s else s)
        elif null:
            cur.append(None)
        elif num is not None:
            cur.append(float(num) if "." in num else int(num))
        i = mm.end()
    return rows


def passagens():
    out = []
    for r in _rows("contagem_passagem"):
        out.append(dict(id=r[0], fonte=r[1], disp=r[2], car=r[4], ts=r[5] / 1000.0,
                        ts_dev=r[7] / 1000.0, method=r[12]))
    return out


def voltas():
    out = []
    for r in _rows("contagem_volta"):
        out.append(dict(car=r[2], lap=r[3], ts=r[4] / 1000.0, disp_ms=r[5], nsrc=r[6],
                        lap_ms=r[7], status=r[9]))
    return out


def equipes():
    return {r[1]: (r[2], r[5]) for r in _rows("equipe")}


def local(ts):
    return dt.datetime.fromtimestamp(ts, TZ)


if __name__ == "__main__":
    p = passagens()
    v = voltas()
    print(len(p), "passagens;", len(v), "voltas")
    from collections import Counter
    print("by fonte", Counter(x["fonte"] for x in p))
    print("status", Counter(x["status"] for x in v))
    print("cars", sorted(set(x["car"] for x in v)))
    print("first/last passagem", local(min(x["ts"] for x in p)), local(max(x["ts"] for x in p)))
    d = [x["ts"] - x["ts_dev"] for x in p]
    print("ts - ts_dev: min %.2f max %.2f" % (min(d), max(d)))
    print(json.dumps(v[:3], default=str))


def tod(ts):
    """epoch seconds -> local seconds-of-day (matches clock.read())."""
    return (ts - 3 * 3600) % 86400
