"""Tests for the tray's watch on the page server (launcher/tray.py: watch_server).

If the page's web server cannot start (its port is taken by another program), uvicorn ends its thread. A tray icon
whose page never opens would only confuse the operator, so the tray notices, shows a notification and the launcher
exits. watch_server is the small loop that notices; the test feeds it a fake "is the server alive?" answer.
"""
import threading

from launcher.tray import watch_server


def test_a_dead_page_server_is_reported_once():
    answers = iter([True, True, False])          # alive, alive, then dead
    reported = []
    watch_server(lambda: next(answers), lambda: reported.append("dead"), threading.Event(), poll_s=0)
    assert reported == ["dead"]


def test_no_report_when_the_launcher_is_closing_anyway():
    closing = threading.Event()
    closing.set()                                 # the launcher is exiting: stop watching, report nothing
    reported = []
    watch_server(lambda: False, lambda: reported.append("dead"), closing, poll_s=0)
    assert reported == []
