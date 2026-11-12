"""Windows timer resolution for rt (WP-L2): make time.sleep(0.001) really sleep ~1 ms.

Why: the beam's modem-line driver polls the adapter 1000 times a second and sleeps ~1 ms between polls. Windows'
default timer tick is 15.6 ms; Python 3.11+ uses a high-resolution waitable timer for time.sleep(), but other waits
(and older Pythons) still follow the system tick. timeBeginPeriod(1) (winmm.dll) asks Windows for a 1 ms tick while
rt runs; timeEndPeriod(1) gives it back at exit. It costs a little battery on a laptop, nothing else.

python_check() returns the warning to log when the Python is older than 3.11 (sleep then follows the 15.6 ms tick
on Windows unless the period was raised).
"""
import atexit
import sys


def raise_timer_resolution(winmm=None, platform=None):
    """On Windows: timeBeginPeriod(1) now and timeEndPeriod(1) at exit. Returns a short text for the log (what was
    done), or None off Windows. `winmm` / `platform` are for the tests (a fake library object). Never raises."""
    platform = platform or sys.platform
    if platform != "win32":
        return None
    try:
        if winmm is None:
            import ctypes
            winmm = ctypes.WinDLL("winmm")
        result = winmm.timeBeginPeriod(1)
        if result != 0:                                  # TIMERR_NOERROR = 0
            return "timeBeginPeriod(1) refused (code %s): sleeps follow the default timer tick" % result
        atexit.register(winmm.timeEndPeriod, 1)
        return "Windows timer resolution set to 1 ms (timeBeginPeriod(1); undone at exit)"
    except Exception as e:  # noqa: BLE001 (a missing winmm must not stop rt)
        return "timeBeginPeriod(1) not available (%s): sleeps follow the default timer tick" % e


def python_check(version_info=None, version=None):
    """(info line with sys.version, warning or None). The warning when Python < 3.11."""
    version_info = version_info or sys.version_info
    version = version or sys.version
    info = "python %s" % version.replace("\n", " ")
    if tuple(version_info[:2]) < (3, 11):
        return info, ("python %d.%d is older than 3.11: time.sleep() is coarse on Windows, the beam poll rate may be "
                      "far below 1 kHz; install Python 3.11+" % tuple(version_info[:2]))
    return info, None
