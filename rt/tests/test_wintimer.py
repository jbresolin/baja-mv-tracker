"""Tests of rt/wintimer.py: on Windows, rt asks for a 1 ms timer tick so the beam can be polled 1000 times a second.

For a beginner: Windows normally wakes sleeping programs only every 15.6 ms. A program that wants to wake up every
1 ms must ask for it (timeBeginPeriod(1) in winmm.dll) and give it back when it ends (timeEndPeriod(1)). These tests
run on any OS: they pretend to be Windows and pass a fake winmm library that only records the calls.
"""
from rt.wintimer import python_check, raise_timer_resolution


class FakeWinmm:
    """Records the calls instead of changing the real Windows timer."""

    def __init__(self, begin_result=0):
        self.calls = []
        self.begin_result = begin_result

    def timeBeginPeriod(self, ms):  # noqa: N802 (the Windows function name)
        self.calls.append(("begin", ms))
        return self.begin_result

    def timeEndPeriod(self, ms):  # noqa: N802
        self.calls.append(("end", ms))
        return 0


def test_not_windows_does_nothing():
    assert raise_timer_resolution(winmm=FakeWinmm(), platform="linux") is None


def test_windows_asks_for_1_ms_and_gives_it_back_at_exit(monkeypatch):
    registered = []
    monkeypatch.setattr("atexit.register", lambda fn, *args: registered.append((fn, args)))
    winmm = FakeWinmm()
    note = raise_timer_resolution(winmm=winmm, platform="win32")
    assert winmm.calls == [("begin", 1)]
    assert "1 ms" in note
    # At exit the period is given back: the registered function is timeEndPeriod with 1.
    fn, args = registered[0]
    fn(*args)
    assert winmm.calls[-1] == ("end", 1)


def test_windows_refusal_is_reported_not_raised():
    note = raise_timer_resolution(winmm=FakeWinmm(begin_result=97), platform="win32")
    assert "refused" in note


def test_python_older_than_3_11_is_a_warning():
    info, warning = python_check((3, 10, 9), "3.10.9 (main)")
    assert info == "python 3.10.9 (main)"
    assert "older than 3.11" in warning
    assert python_check((3, 12, 1), "3.12.1")[1] is None
