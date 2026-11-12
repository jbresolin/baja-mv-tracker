"""Is this PC's clock trustworthy? (Session 1: the recording PC ran 80 s ahead of the lap server.)

The primary check is an SNTP request to `clock.ntp_server` (at the event: the router's address): a real measurement
of this PC's offset. Without it (or on top of it), the OS time service is asked, locally, without network:
Windows `w32tm /query /status [/verbose]`, Linux `timedatectl`.

w32tm prints its labels in the language of Windows ("Leap Indicator" in English, Portuguese labels on a pt-BR
laptop), in the console's code page. So the w32tm output is read by its STRUCTURE, which is the same in every
language: one "label: value" line per field, always in the same order, with values that are numbers, hex ids,
dates and seconds. Known labels (English, Portuguese, French) are used when they match; otherwise each field is
found by its position and the shape of its value. No sample of a real pt-BR w32tm output was available when this
was written: the Portuguese test sample is built from the English layout with likely translations, and the
parser does not depend on them.
"""
import datetime as dt
import locale
import re
import socket
import struct
import subprocess
import sys
import time

# Labels as printed by w32tm in some languages (lower case, accents optional). Only a shortcut: the parser falls
# back to the structure when none matches.
LABELS = {
    "leap": ("leap indicator", "indicador de salto", "indicateur de saut"),
    "stratum": ("stratum", "estrato", "strate"),
    "last_sync": ("last successful sync time", "hora da ultima sincronizacao", "heure de la derniere synchronisation"),
    "source": ("source", "fonte"),
    "phase_offset": ("phase offset", "deslocamento de fase", "desvio de fase", "decalage de phase"),
}
FIELD_LINE = re.compile(r"^\s*([^:\r\n]+?)\s*:\s?(.*?)\s*$")      # "label: value" (the label never has a colon)
LEAP_VALUE = re.compile(r"^([0-3])\s*\(")                          # "3(not synchronized)" / "0(nenhum aviso)"
STRATUM_VALUE = re.compile(r"^(\d+)\s*\(")                         # "4 (secondary reference ...)"
REFERENCE_ID_VALUE = re.compile(r"^0x([0-9A-Fa-f]{8})")            # "0x14659A44 (source IP: ...)"
DATE_VALUE = re.compile(r"\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\s+\d{1,2}:\d{2}:\d{2}")
POLL_INTERVAL_VALUE = re.compile(r"^\d+\s*\(\d+s\)$")              # "10 (1024s)"
SECONDS_VALUE = re.compile(r"^([-+]?\d+(?:[.,]\d+)?)s$")           # "-0.0012345s" / "0,0012345s"
LOCAL_CLOCK_REFERENCE_ID = "4C4F434C"                              # "LOCL": the local CMOS clock
UNSYNCED_SOURCE = re.compile(r"CMOS|Free-running|livre|libre", re.I)


def _plain(text):
    """Lower case without accents, so 'Última Sincronização' matches 'ultima sincronizacao'."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def decode_console_output(raw):
    """w32tm writes in the console code page (cp850 on a pt-BR Windows), not in UTF-8 nor in the ANSI code page
    Python would assume, so accented labels can come out garbled or even fail to decode. Never raise: the parser
    does not need the labels."""
    if isinstance(raw, str):
        return raw
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    for encoding in ("oem", "cp850", locale.getpreferredencoding(False)):
        try:
            return raw.decode(encoding, errors="replace")
        except LookupError:            # the "oem" codec only exists on Windows
            continue
    return raw.decode("latin-1", errors="replace")


def parse_w32tm_status(text):
    """`w32tm /query /status [/verbose]` output, in any language -> dict with leap (0-3), stratum, reference_id
    (8 hex digits), last_sync (the date text, None if never / unspecified), source, phase_offset_s (only with
    /verbose). A field that cannot be found is None."""
    fields = []                                    # [(plain label, value)] in printed order
    for line in (text or "").splitlines():
        match = FIELD_LINE.match(line)
        if match:
            fields.append((_plain(match.group(1)), match.group(2)))
    result = dict(leap=None, stratum=None, reference_id=None, last_sync=None, source=None, phase_offset_s=None)

    def by_label(key):
        for i, (label, value) in enumerate(fields):
            if any(label.startswith(known) for known in LABELS[key]):
                return i, value
        return None, None

    # Leap indicator: the first field. Stratum: the second.
    _, value = by_label("leap")
    for candidate in ([value] if value is not None else []) + [v for _, v in fields[:1]]:
        match = LEAP_VALUE.match(candidate)
        if match:
            result["leap"] = int(match.group(1))
            break
    _, value = by_label("stratum")
    for candidate in ([value] if value is not None else []) + [v for _, v in fields[1:2]]:
        match = STRATUM_VALUE.match(candidate)
        if match:
            result["stratum"] = int(match.group(1))
            break
    for _, value in fields:
        match = REFERENCE_ID_VALUE.match(value)
        if match:
            result["reference_id"] = match.group(1).upper()
            break
    # Last successful sync: the field with a date in it (unspecified = never synced).
    sync_index, value = by_label("last_sync")
    if sync_index is None:
        sync_index = next((i for i, (_, v) in enumerate(fields) if DATE_VALUE.search(v)), None)
        value = fields[sync_index][1] if sync_index is not None else None
    if value is not None and DATE_VALUE.search(value):
        result["last_sync"] = value
    # Source: by label, else the field after the last-sync field, else the one before the poll interval.
    poll_index = next((i for i, (_, v) in enumerate(fields) if POLL_INTERVAL_VALUE.match(v)), None)
    source_index, value = by_label("source")
    if source_index is None:
        if sync_index is not None and sync_index + 1 < len(fields):
            source_index = sync_index + 1
        elif poll_index:
            source_index = poll_index - 1
        value = fields[source_index][1] if source_index is not None else None
    result["source"] = value
    # Phase offset (/verbose): by label, else the first "<seconds>s" value after the poll interval.
    _, value = by_label("phase_offset")
    if value is None and poll_index is not None:
        value = next((v for _, v in fields[poll_index + 1:] if SECONDS_VALUE.match(v)), None)
    match = SECONDS_VALUE.match(value or "")
    if match:
        result["phase_offset_s"] = float(match.group(1).replace(",", "."))
    return result


def windows_sync_warnings(text, warn_sync_age_h, now=None):
    """Warnings (strings) from a `w32tm /query /status` output; [] = synchronised and recent."""
    if not (text or "").strip():
        return ["could not verify Windows time sync (w32tm printed nothing; is the time service running?)"]
    status = parse_w32tm_status(text)
    warnings = []
    source = status["source"] or ""
    if status["reference_id"] == LOCAL_CLOCK_REFERENCE_ID or UNSYNCED_SOURCE.search(source):
        warnings.append("Windows time is NOT synchronised (source: %s). Session 1's PC was 80 s off. Fix: "
                        "Settings > Time > 'Sync now', or `w32tm /resync` as admin, or set clock.offset_s." % source)
    elif status["leap"] == 3:
        warnings.append("Windows time service reports 'not synchronized' (leap indicator 3).")
    elif status["leap"] is None:
        warnings.append("could not read the Windows time service status (unexpected w32tm output)")
    if status["last_sync"]:
        age = _age_hours(status["last_sync"], now)
        if age is not None and age > float(warn_sync_age_h):
            warnings.append("last Windows time sync was %.0f h ago (> %.0f h)." % (age, warn_sync_age_h))
    return warnings


def run_w32tm(args, runner=subprocess.run):
    """Run w32tm and return its output as text (decoded safely); raises if w32tm cannot run."""
    result = runner(["w32tm"] + list(args), capture_output=True, timeout=10)
    return decode_console_output(result.stdout or b"")


def check(cfg, log, platform=None, runner=subprocess.run, sntp=None, now=None):
    """Returns a list of warning strings (also used for the status page). `platform`, `runner` and `sntp` are
    replaceable for tests."""
    clock_cfg = cfg["clock"]
    platform = platform or sys.platform
    sntp = sntp or sntp_offset
    warns = []
    if platform == "win32":
        try:
            text = run_w32tm(["/query", "/status"], runner)
        except Exception as e:  # noqa: BLE001
            warns.append("could not verify Windows time sync: w32tm not available (%s)" % e)
        else:
            warns += windows_sync_warnings(text, clock_cfg["warn_sync_age_h"], now)
    else:
        try:
            out = runner(["timedatectl", "show", "-p", "NTPSynchronized"], capture_output=True, text=True,
                         timeout=10).stdout
            if "NTPSynchronized=yes" not in (out or ""):
                warns.append("system clock is not NTP-synchronised (timedatectl).")
        except Exception as e:  # noqa: BLE001
            warns.append("could not verify time sync: %s" % e)
    if clock_cfg.get("ntp_server"):
        offset = sntp(clock_cfg["ntp_server"])
        if offset is None:
            warns.append("SNTP check against %s failed" % clock_cfg["ntp_server"])
        else:
            log("SNTP: this clock - %s = %+.3f s" % (clock_cfg["ntp_server"], offset))
            if abs(offset) > 0.5:
                warns.append("this PC's clock is %+.2f s off %s; set clock.offset_s = %.2f or resync." %
                             (offset, clock_cfg["ntp_server"], -offset))
    return warns


def measure_offset(cfg, runner=subprocess.run, platform=None, sntp=None):
    """This PC's clock minus the reference, in seconds, for the status page. Returns (offset_s or None, source).

    With clock.ntp_server set: one SNTP request to it (the real measurement). Otherwise what the OS time service
    reports: Windows `w32tm /query /status /verbose` phase offset, Linux `timedatectl timesync-status` "Offset".
    None = unknown (the page shows it in amber)."""
    server = cfg["clock"].get("ntp_server")
    if server:
        return (sntp or sntp_offset)(server), "SNTP %s" % server
    try:
        if (platform or sys.platform) == "win32":
            return parse_w32tm_offset(run_w32tm(["/query", "/status", "/verbose"], runner)), "w32tm phase offset"
        out = runner(["timedatectl", "timesync-status"], capture_output=True, text=True, timeout=10).stdout
        return parse_timedatectl_offset(out), "timedatectl"
    except Exception:  # noqa: BLE001
        return None, "no time service query"


def parse_w32tm_offset(out):
    """The phase offset of `w32tm /query /status /verbose` in seconds ('Phase Offset: 0.0012345s' in English, any
    other language by position), or None."""
    return parse_w32tm_status(out)["phase_offset_s"]


def parse_timedatectl_offset(out):
    """'Offset: +1.234ms' / '-25us' / '+1.2s' -> seconds."""
    match = re.search(r"^\s*Offset:\s*([-+]?\d+(?:\.\d+)?)\s*(us|µs|ms|s|min)\s*$", out or "", re.M)
    if not match:
        return None
    scale = {"us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0}[match.group(2)]
    return float(match.group(1)) * scale


def _age_hours(text, now=None):
    """Hours since the date in `text` (w32tm's last sync time, in the PC's date format), or None if unreadable.
    US English prints 11/12/2026 10:15:02 AM, pt-BR 12/11/2026 10:15:02: the AM/PM decides the order; without it
    day/month is tried first (pt-BR), and an impossible result (in the future) is rejected."""
    match = re.search(r"(\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\s+\d{1,2}:\d{2}:\d{2}(?:\s*[AaPp]\.?\s*[Mm]\.?)?)", text)
    if not match:
        return None
    stamp = match.group(1).replace(".", "").replace("-", "/")
    now = now or dt.datetime.now()
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            age = (now - dt.datetime.strptime(stamp, fmt)).total_seconds() / 3600
        except ValueError:
            continue
        if age >= -1:                  # a date in the future = wrong day/month order: try the next format
            return age
    return None


def sntp_offset(server, timeout=2.0):
    """local clock - server clock (s), one SNTP request (RFC 4330). None if the server does not answer."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:     # closed even on timeout
            sock.settimeout(timeout)
            t_sent = time.time()
            sock.sendto(b"\x1b" + 47 * b"\0", (server, 123))
            data, _ = sock.recvfrom(48)
            t_received = time.time()
        seconds, fraction = struct.unpack("!II", data[40:48])
        t_server_sent = seconds - 2208988800 + fraction / 2**32
        seconds, fraction = struct.unpack("!II", data[32:40])
        t_server_received = seconds - 2208988800 + fraction / 2**32
        return ((t_sent - t_server_received) + (t_received - t_server_sent)) / 2
    except Exception:  # noqa: BLE001
        return None
