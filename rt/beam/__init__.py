"""Beam input (WP-L2, AUTOPTZ-23): the event timing light-beam sensors at the start/finish line, read through a USB
serial adapter, as precise crossing times next to rt's vision passes. Vision stays responsible for identity and
counting; the beam only gives times. Shadow mode: nothing goes to the lap app.

  sources.py    drivers: modem-line (CTS/DSR/DCD/RI polled at ~1 kHz), byte-stream (bytes per pulse), fake
  grouping.py   raw edges -> one break per car (group_gap_s), stuck-beam detection
  associate.py  live pass <-> break rule, LED latency detector
  service.py    BeamService: reader + dispatcher threads, JSONL events, status page numbers
Tools: rt/tools/beam_probe.py (which line / bytes does the adapter use?), rt/tools/fuse_beam.py (offline fusion).
"""
from .associate import LedWatcher, associate_pass
from .grouping import BreakGrouper
from .service import BeamService, PassBeamSink
from .sources import (BeamPortError, BeamSource, ByteStreamSource, FakeBeamSource, ModemLineSource, RawEdge,
                      make_source)

__all__ = ["BeamService", "PassBeamSink", "BreakGrouper", "LedWatcher", "associate_pass", "BeamPortError",
           "BeamSource", "ByteStreamSource", "FakeBeamSource", "ModemLineSource", "RawEdge", "make_source"]
