"""Tests of the clock checks (rt/timesync.py): reading the Windows time service's answer in English AND Portuguese,
the Linux one, and the SNTP check. Nothing here runs w32tm or touches the network: the commands are replaced by
fakes that return sample outputs.

Why it matters: rt stamps every pass with this PC's clock. Session 1's PC was 80 s off and nobody knew. rt asks
Windows "is your clock synchronised?" with `w32tm /query /status`, and Windows answers in its own language: on the
race laptop (pt-BR) the labels are Portuguese, so a parser that only knows "Leap Indicator" would see nothing.

About the Portuguese samples: they follow the exact layout of the English output with likely Portuguese labels. A
real pt-BR capture was not available, so the parser is built to work on the layout (field order, value shapes)
and the tests also check a sample whose labels are pure nonsense.
"""
import datetime as dt

import pytest

from rt import timesync

# ------------------------------------------------------------------------------------------------ sample outputs
# `w32tm /query /status /verbose` on an English Windows that is synchronised (layout from Microsoft's docs).
EN_SYNCED = """Leap Indicator: 0(no warning)
Stratum: 4 (secondary reference - syncd by (S)NTP)
Precision: -23 (119.209ns per tick)
Root Delay: 0.0312500s
Root Dispersion: 7.9002196s
ReferenceId: 0x14659A44 (source IP:  20.101.57.9)
Last Successful Sync Time: 11/12/2026 10:15:02 AM
Source: time.windows.com,0x9
Poll Interval: 10 (1024s)

Phase Offset: -0.0012345s
ClockRate: 0.0156250s
State Machine: 2 (Sync)
Time Source Flags: 0 (None)
Server Role: 0 (None)
Last Sync Error: 0 (The command completed successfully.)
Time since Last Good Sync Time: 123.4567890s
"""

# An English Windows that never synchronised: it runs on its own battery-backed clock (what session 1's PC did).
EN_LOCAL_CLOCK = """Leap Indicator: 3(not synchronized)
Stratum: 0 (unspecified)
Precision: -23 (119.209ns per tick)
Root Delay: 0.0000000s
Root Dispersion: 0.0000000s
ReferenceId: 0x00000000 (unspecified)
Last Successful Sync Time: unspecified
Source: Local CMOS Clock
Poll Interval: 10 (1024s)
"""

# The same synchronised answer as a pt-BR Windows would print it (dates day/month/year, 24 h clock, comma decimal).
PT_SYNCED = """Indicador de Salto: 0(nenhum aviso)
Estrato: 4 (referência secundária - sincronizada por (S)NTP)
Precisão: -23 (119.209ns por tique)
Atraso de Raiz: 0,0312500s
Dispersão de Raiz: 7,9002196s
ID de Referência: 0x14659A44 (IP de origem:  20.101.57.9)
Hora da Última Sincronização Bem-sucedida: 12/11/2026 10:15:02
Fonte: time.windows.com,0x9
Intervalo de Sondagem: 10 (1024s)

Deslocamento de Fase: 0,0004321s
Taxa de Relógio: 0,0156250s
Máquina de Estado: 2 (Sincronizar)
"""

# pt-BR, never synchronised.
PT_LOCAL_CLOCK = """Indicador de Salto: 3(não sincronizado)
Estrato: 0 (não especificado)
Precisão: -23 (119.209ns por tique)
Atraso de Raiz: 0,0000000s
Dispersão de Raiz: 0,0000000s
ID de Referência: 0x00000000 (não especificado)
Hora da Última Sincronização Bem-sucedida: não especificado
Fonte: Relógio CMOS Local
Intervalo de Sondagem: 10 (1024s)
"""

# The pt-BR answer with every label replaced by nonsense: what the parser sees if the real labels differ from our
# guesses, or the accents come out garbled. Only the layout and the values are left.
UNKNOWN_LABELS = """Aaa: 0(xx)
Bbb: 4 (yy)
Ccc: -23 (119.209ns)
Ddd: 0,0312500s
Eee: 7,9002196s
Fff: 0x14659A44 (zz:  20.101.57.9)
Ggg: 12/11/2026 10:15:02
Hhh: time.windows.com,0x9
Iii: 10 (1024s)

Jjj: -0,0250000s
Kkk: 0,0156250s
"""

NOW = dt.datetime(2026, 11, 12, 12, 0, 0)   # "now" for the tests: 1 h 45 min after the samples' last sync


# ------------------------------------------------------------------------------------------------ parsing
@pytest.mark.parametrize("sample, offset", [(EN_SYNCED, -0.0012345), (PT_SYNCED, 0.0004321),
                                            (UNKNOWN_LABELS, -0.025)])
def test_status_fields_are_read_in_any_language(sample, offset):
    # The same fields come out whether the labels are English, Portuguese or unknown.
    status = timesync.parse_w32tm_status(sample)
    assert status["leap"] == 0 and status["stratum"] == 4
    assert status["reference_id"] == "14659A44"
    assert status["source"] == "time.windows.com,0x9"
    assert "10:15:02" in status["last_sync"]
    assert status["phase_offset_s"] == pytest.approx(offset)       # the /verbose "phase offset"


@pytest.mark.parametrize("sample", [EN_LOCAL_CLOCK, PT_LOCAL_CLOCK])
def test_never_synced_status(sample):
    status = timesync.parse_w32tm_status(sample)
    assert status["leap"] == 3 and status["stratum"] == 0
    assert status["last_sync"] is None                              # "unspecified" / "não especificado"
    assert "CMOS" in status["source"]


def test_parse_w32tm_offset():
    assert timesync.parse_w32tm_offset(EN_SYNCED) == pytest.approx(-0.0012345)
    assert timesync.parse_w32tm_offset(PT_SYNCED) == pytest.approx(0.0004321)   # comma decimal "0,0004321s"
    assert timesync.parse_w32tm_offset(EN_LOCAL_CLOCK) is None      # no /verbose part: unknown, not 0
    assert timesync.parse_w32tm_offset("") is None


def test_parse_timedatectl_offset():
    # Linux `timedatectl timesync-status` prints the offset with a unit; the result is always in seconds.
    assert timesync.parse_timedatectl_offset("       Server: 192.168.88.1\n       Offset: +1.234ms\n") == \
        pytest.approx(0.001234)
    assert timesync.parse_timedatectl_offset("Offset: -25us") == pytest.approx(-25e-6)
    assert timesync.parse_timedatectl_offset("Offset: +1.2s") == pytest.approx(1.2)
    assert timesync.parse_timedatectl_offset("no offset here") is None


def test_console_output_in_the_windows_code_page_is_decoded_without_crashing():
    # w32tm on a pt-BR Windows writes cp850 bytes. Decoding them as UTF-8 would crash ("invalid byte"); the old
    # code then reported "w32tm not available". Now the text always comes out, and the parse still works.
    raw = PT_SYNCED.encode("cp850")
    text = timesync.decode_console_output(raw)
    assert timesync.parse_w32tm_status(text)["leap"] == 0


# ------------------------------------------------------------------------------------------------ warnings
@pytest.mark.parametrize("sample", [EN_SYNCED, PT_SYNCED, UNKNOWN_LABELS])
def test_synced_clock_gives_no_warning(sample):
    assert timesync.windows_sync_warnings(sample, 24, now=NOW) == []


@pytest.mark.parametrize("sample", [EN_LOCAL_CLOCK, PT_LOCAL_CLOCK])
def test_local_clock_gives_a_loud_warning(sample):
    warnings = timesync.windows_sync_warnings(sample, 24, now=NOW)
    assert len(warnings) == 1 and "NOT synchronised" in warnings[0]


def test_old_last_sync_is_warned_in_both_date_formats():
    # Last sync on 12 Nov 10:15 (written 11/12 in English, 12/11 in Portuguese), checked 3 days later.
    three_days_later = dt.datetime(2026, 11, 15, 10, 15, 2)
    for sample in (EN_SYNCED, PT_SYNCED):
        warnings = timesync.windows_sync_warnings(sample, 24, now=three_days_later)
        assert warnings == ["last Windows time sync was 72 h ago (> 24 h)."]


def test_empty_or_strange_output_is_a_warning_not_silence():
    assert "printed nothing" in timesync.windows_sync_warnings("", 24)[0]
    assert "unexpected w32tm output" in timesync.windows_sync_warnings("something else entirely", 24)[0]


class FakeRun:
    """Stands in for subprocess.run: returns `stdout` (bytes, as w32tm really gives) and remembers the command."""

    def __init__(self, stdout):
        self.stdout, self.commands = stdout, []

    def __call__(self, cmd, **kw):
        self.commands.append(cmd)

        class Result:
            pass

        result = Result()
        result.stdout = self.stdout
        return result


def test_check_on_windows_and_sntp_is_the_primary_measurement(cfg):
    # check() as rt calls it at start, on a (pretend) Windows: w32tm says synced, and the SNTP request to the
    # event router (clock.ntp_server) measures +0.8 s: that is the warning that counts.
    cfg["clock"]["ntp_server"] = "192.168.88.1"
    runner = FakeRun(PT_SYNCED.encode("cp850"))
    logs = []
    warnings = timesync.check(cfg, logs.append, platform="win32", runner=runner, sntp=lambda server: 0.8, now=NOW)
    assert runner.commands[0][:3] == ["w32tm", "/query", "/status"]
    assert warnings == ["this PC's clock is +0.80 s off 192.168.88.1; set clock.offset_s = -0.80 or resync."]
    assert logs == ["SNTP: this clock - 192.168.88.1 = +0.800 s"]


def test_measure_offset_prefers_sntp_then_w32tm(cfg):
    cfg["clock"]["ntp_server"] = "192.168.88.1"
    assert timesync.measure_offset(cfg, sntp=lambda server: 0.012) == (0.012, "SNTP 192.168.88.1")
    cfg["clock"]["ntp_server"] = ""
    runner = FakeRun(EN_SYNCED.encode())
    offset, source = timesync.measure_offset(cfg, runner=runner, platform="win32")
    assert offset == pytest.approx(-0.0012345) and source == "w32tm phase offset"
    assert "/verbose" in runner.commands[0]


def test_sntp_socket_is_closed_on_timeout(monkeypatch):
    # No answer from the server: sntp_offset returns None, and the socket is closed (not left open each minute).
    closed = []

    class SilentSocket:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            closed.append(True)

        def settimeout(self, seconds):
            pass

        def sendto(self, data, address):
            pass

        def recvfrom(self, size):
            raise TimeoutError("no answer")

    monkeypatch.setattr(timesync.socket, "socket", SilentSocket)
    assert timesync.sntp_offset("192.0.2.1") is None
    assert closed == [True]
