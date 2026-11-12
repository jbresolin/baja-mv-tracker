"""Output sinks. Every event goes through `Emitter.emit`, which stamps it and hands it to each sink (thread-safe:
the detection loop and the identity worker both emit). Add a sink by subclassing `Sink` (event() / close()).

Event types (one JSON object per line, see rt/README.md for every field):
  pass_start        a new foreground car entered (PTZ entry signal), ~0.6 s after it appeared
  pass_provisional  first confident number while the pass is still open (PTZ entry camera)
  pass_end          the pass closed (re-attach windows expired): final t_enter / t_cross / t_exit
  pass_identity     the number read from the pass's crops (number, confidence, auto_confirm, top3, latency)
  info / warning    run start / end, clock warnings
  beam, beam_edge, beam_break, beam_stuck, beam_poll, pass_beam, camera_latency   beam input (rt/beam/, WP-L2)
"""
import csv
import datetime as dt
import json
import threading
import time
import uuid
from pathlib import Path

# Times rounded to the ms and given an _iso twin. t_edge / t_beam / t_last / t_since are the beam events' (WP-L2).
T_KEYS = ("t_enter", "t_cross", "t_exit", "t_video", "t_edge", "t_beam", "t_last", "t_since")


def iso(t):
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds") if t is not None else None


class Sink:
    def event(self, ev):
        pass

    def close(self):
        pass


class Emitter:
    def __init__(self, sinks=()):
        self.sinks = list(sinks)
        self.lock = threading.Lock()
        self.counts = {}

    def emit(self, ev):
        ev = dict(ev)
        ev["wall"] = round(time.time(), 3)
        for k in T_KEYS:
            if ev.get(k) is not None:
                ev[k] = round(float(ev[k]), 3)
                ev[k + "_iso"] = iso(ev[k])
        with self.lock:
            self.counts[ev["type"]] = self.counts.get(ev["type"], 0) + 1
            for s in self.sinks:
                try:
                    s.event(ev)
                except Exception as e:  # noqa: BLE001 (a broken sink must not stop counting)
                    print("sink %s failed: %s" % (type(s).__name__, e), flush=True)

    def close(self):
        for s in self.sinks:
            s.close()


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    return str(o)


class JsonlSink(Sink):
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def event(self, ev):
        self.f.write(json.dumps(ev, default=_default, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


CSV_COLS = ["pass_id", "t_enter_iso", "t_cross_iso", "t_exit_iso", "t_enter", "t_cross", "t_exit", "dwell_s", "n_frames",
            "n_tracks", "exited", "number", "team", "confidence", "auto_confirm", "support", "n_crops", "n_boxed",
            "n_boxes", "alt2", "alt2_share", "alt3", "alt3_share", "provisional_number", "identity_latency_s"]


class CsvSink(Sink):
    """One row per pass, written when its identity is known."""

    def __init__(self, path):
        new = not Path(path).exists() or Path(path).stat().st_size == 0
        self.f = open(path, "a", encoding="utf-8", newline="")
        self.w = csv.DictWriter(self.f, CSV_COLS, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def event(self, ev):
        if ev["type"] != "pass_identity":
            return
        row = dict(ev)
        for k, a in enumerate(ev.get("top3", [])[1:3], start=2):
            row["alt%d" % k], row["alt%d_share" % k] = a["number"], a["share"]
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


class ConsoleSink(Sink):
    def __init__(self, quiet=False):
        self.quiet = quiet

    def event(self, ev):
        t = ev["type"]
        if t in ("warning", "info"):
            print("[%s] %s" % (t, ev.get("msg", "")), flush=True)
            return
        if self.quiet:
            return
        tt = (ev.get("t_enter_iso") or ev.get("t_video_iso") or "")[11:23]
        if t == "pass_start":
            print("%s  pass %4d  START" % (tt, ev["pass_id"]), flush=True)
        elif t == "pass_provisional":
            print("%s  pass %4d  provisional #%s (%.2f)" % ((ev.get("t_video_iso") or "")[11:23], ev["pass_id"],
                                                            ev["number"], ev["confidence"]), flush=True)
        elif t == "pass_end":
            print("%s  pass %4d  END    dwell %.1f s, %d frames, %d crops" % (tt, ev["pass_id"], ev["dwell_s"],
                                                                           ev["n_frames"], ev["n_crops"]), flush=True)
        elif t == "pass_identity":
            print("%s  pass %4d  #%-4s conf %.2f %s  (%d/%d crops boxed, %.2f s after close)" % (
                tt, ev["pass_id"], ev["number"], ev["confidence"], "AUTO" if ev["auto_confirm"] else "review",
                ev["n_boxed"], ev["n_crops"], ev["identity_latency_s"]), flush=True)


class LapAppSink(Sink):
    """STUB of the third counting source ("fonte") for the lap app. Nothing is sent: each finished pass becomes the
    `contagem_passagem` row the camera would insert, appended to an outbox JSONL file. `deliver()` is where the real
    integration goes once the user decides between the app's API and a direct DB insert.

    contagem_passagem columns (lapvision/gt.py, from the user): passagem_id (auto), fonte_id, dispositivo_id,
    uuid_cliente, equipe_numero, ts_ms (server clock: THE pass time), qualidade_tempo ('exato' in all 586 rows of
    26SE), ts_dispositivo_ms (device clock), ancora_servidor_ms / ancora_rtt_ms / ancora_idade_ms (the web client's
    server-clock anchor), digitacao_ms (typing time), modo_entrada ('blocos' / 'teclado_tela' in 26SE),
    recebido_em_ms (server receive time). contagem_fonte 26SE: 1-3 'Contagem N' (manual, humana), 4 'Box'; the camera
    needs a new row, e.g. (5, EV, 'END', 'Câmera', 'camera', 'maquina', 1, 4, NULL) -- column meanings inferred.

    Open points for the user (also in rt/README.md): equipe_numero is NOT NULL in every 26SE row, so an unread pass
    (number None) cannot be inserted as is (default: skipped, kept in the outbox with skipped=true); how auto_confirm
    maps to the reviewer flow (`contagem_decisao` 'confirmar' rows by 'revisor'); ts_ms needs the server clock
    (clock.video_minus_server_s, or the app's anchor protocol)."""

    def __init__(self, cfg, outbox, event_code, source_name):
        self.c = cfg["lapapp"]
        self.offset = float(cfg["clock"]["video_minus_server_s"])
        self.f = open(outbox, "a", encoding="utf-8")
        self.ev, self.src = event_code, source_name

    def row(self, ev):
        t = ev["t_enter"]  # the human click is closest to "car enters the frame" (LESSONS: +0.35 s median)
        key = "%s|%s|%d|%.3f" % (self.ev, self.src, ev["pass_id"], t)
        return dict(
            passagem_id=None, fonte_id=int(self.c["fonte_id"]), dispositivo_id=int(self.c["dispositivo_id"]),
            uuid_cliente=str(uuid.uuid5(uuid.NAMESPACE_URL, "baja-rt:" + key)),  # stable -> idempotent retries
            equipe_numero=ev["number"], ts_ms=int(round((t - self.offset) * 1000)), qualidade_tempo="exato",
            ts_dispositivo_ms=int(round(t * 1000)), ancora_servidor_ms=None, ancora_rtt_ms=None, ancora_idade_ms=None,
            digitacao_ms=None, modo_entrada="camera", recebido_em_ms=None)

    def deliver(self, row, meta):
        """Integration point (not implemented): POST to the lap app's API, or INSERT INTO contagem_passagem.
        Must be idempotent on uuid_cliente and must never block the detection loop (queue + retry thread)."""
        return False

    def event(self, ev):
        if ev["type"] != "pass_identity":
            return
        row = self.row(ev)
        meta = dict(pass_id=ev["pass_id"], confidence=ev["confidence"], auto_confirm=ev["auto_confirm"],
                    top3=ev["top3"], t_cross=ev["t_cross"], t_exit=ev["t_exit"])
        skipped = ev["number"] is None and bool(self.c["only_numbered"])
        sent = False if skipped else self.deliver(row, meta)
        self.f.write(json.dumps(dict(table="contagem_passagem", row=row, meta=meta, skipped=skipped, sent=sent),
                                ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
