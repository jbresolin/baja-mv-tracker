"""Is this PC's clock trustworthy? (Session 1: the recording PC ran 80 s ahead of the lap server.)

Windows: `w32tm /query /status` (local query, no network): warns when the source is the local CMOS clock / free-running,
the leap indicator says "not synchronized", or the last successful sync is older than warn_sync_age_h. Linux:
`timedatectl show`. Optional SNTP offset check against clock.ntp_server (off by default: no network unless set).
"""
import datetime as dt
import re
import socket
import struct
import subprocess
import sys
import time


def windows_status():
    try:
        out = subprocess.run(["w32tm", "/query", "/status"], capture_output=True, text=True, timeout=10).stdout
    except Exception as e:  # noqa: BLE001
        return None, "w32tm not available (%s)" % e
    return out, None


def check(cfg, log):
    """Returns a list of warning strings (also logged)."""
    c = cfg["clock"]
    warns = []
    if sys.platform == "win32":
        out, err = windows_status()
        if err:
            warns.append("could not verify Windows time sync: " + err)
        else:
            src = re.search(r"^\s*(Source|Fonte)\s*:\s*(.+)$", out, re.M | re.I)
            leap = re.search(r"^\s*Leap Indicator\s*:\s*(\d)", out, re.M | re.I)
            last = re.search(r"^\s*Last Successful Sync Time\s*:\s*(.+)$", out, re.M | re.I)
            s = src.group(2).strip() if src else ""
            if not out.strip():
                warns.append("could not verify Windows time sync (w32tm printed nothing; is the time service running?)")
            if re.search(r"Local CMOS Clock|Free-running|Rel[oó]gio CMOS", s, re.I):
                warns.append("Windows time is NOT synchronised (source: %s). Session 1's PC was 80 s off. Fix: "
                             "Settings > Time > 'Sync now', or `w32tm /resync` as admin, or set clock.offset_s." % s)
            elif leap and leap.group(1) == "3":
                warns.append("Windows time service reports 'not synchronized' (leap indicator 3).")
            if last and "unspecified" not in last.group(1).lower():
                age = _age_hours(last.group(1).strip())
                if age is not None and age > float(c["warn_sync_age_h"]):
                    warns.append("last Windows time sync was %.0f h ago (> %.0f h)." % (age, c["warn_sync_age_h"]))
    else:
        try:
            out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized"], capture_output=True, text=True,
                                 timeout=10).stdout
            if "NTPSynchronized=yes" not in out:
                warns.append("system clock is not NTP-synchronised (timedatectl).")
        except Exception as e:  # noqa: BLE001
            warns.append("could not verify time sync: %s" % e)
    if c.get("ntp_server"):
        off = sntp_offset(c["ntp_server"])
        if off is None:
            warns.append("SNTP check against %s failed" % c["ntp_server"])
        else:
            log("SNTP: this clock - %s = %+.3f s" % (c["ntp_server"], off))
            if abs(off) > 0.5:
                warns.append("this PC's clock is %+.2f s off %s; set clock.offset_s = %.2f or resync." %
                             (off, c["ntp_server"], -off))
    return warns


def _age_hours(s):
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return (dt.datetime.now() - dt.datetime.strptime(s, fmt)).total_seconds() / 3600
        except ValueError:
            continue
    return None


def sntp_offset(server, timeout=2.0):
    """local clock - server clock (s), one SNTP request (RFC 4330)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        t1 = time.time()
        s.sendto(b"\x1b" + 47 * b"\0", (server, 123))
        data, _ = s.recvfrom(48)
        t4 = time.time()
        s.close()
        sec, frac = struct.unpack("!II", data[40:48])
        t3 = sec - 2208988800 + frac / 2**32
        sec, frac = struct.unpack("!II", data[32:40])
        t2 = sec - 2208988800 + frac / 2**32
        return ((t1 - t2) + (t4 - t3)) / 2
    except Exception:  # noqa: BLE001
        return None
