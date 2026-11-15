"""Tests for launcher/netprofile.py: the "this network is marked Public" warning.

Windows calls a network it does not know "Public", and a race LAN without internet usually is one. The real code asks
PowerShell (Get-NetConnectionProfile); here a fake `runner` returns the JSON PowerShell would print, so the tests run
on any computer.
"""
import json

from launcher.netprofile import (CachedNetworkWarning, fix_command, parse_profiles, public_network_warning,
                                 public_networks)


def powershell_says(*profiles):
    """A fake runner: whatever command it gets, it "prints" these profiles as PowerShell's JSON would."""
    text = json.dumps(profiles[0] if len(profiles) == 1 else list(profiles))
    return lambda command: text


def test_a_public_network_is_found_and_named():
    runner = powershell_says({"Name": "RedeBaja", "InterfaceAlias": "Ethernet", "NetworkCategory": "Public"},
                             {"Name": "Casa", "InterfaceAlias": "Wi-Fi", "NetworkCategory": "Private"})
    found = public_networks(runner=runner, platform="win32")
    assert found == [dict(name="RedeBaja", interface="Ethernet", category="Public")]
    warning = public_network_warning(found)
    assert "RedeBaja" in warning
    # the warning carries the exact command that fixes it
    assert "Set-NetConnectionProfile -Name 'RedeBaja' -NetworkCategory Private" in warning


def test_older_powershell_numbers_are_understood():
    # Some PowerShell versions print the category as a number: 0 = Public, 1 = Private, 2 = Domain.
    assert parse_profiles('{"Name": "X", "NetworkCategory": 0}')[0]["category"] == "Public"
    assert parse_profiles('{"Name": "X", "NetworkCategory": 1}')[0]["category"] == "Private"


def test_no_warning_when_every_network_is_private_or_off_windows():
    runner = powershell_says({"Name": "Casa", "InterfaceAlias": "Wi-Fi", "NetworkCategory": "Private"})
    assert public_network_warning(public_networks(runner=runner, platform="win32")) is None
    # Linux / macOS: nothing to check, and PowerShell is never started
    assert public_networks(runner=lambda c: 1 / 0, platform="linux") == []


def test_garbage_or_no_answer_from_powershell_is_not_an_error():
    assert parse_profiles("") == []
    assert parse_profiles("this is not json") == []


def test_a_quote_in_the_network_name_cannot_break_the_command():
    # PowerShell doubles a ' inside '...' strings
    assert fix_command("Rede d'Água") == "Set-NetConnectionProfile -Name 'Rede d''Água' -NetworkCategory Private"


def test_the_page_asks_windows_at_most_once_a_minute():
    asked = []
    now = [0.0]

    def lookup():
        asked.append(now[0])
        return [dict(name="RedeBaja", interface="Ethernet", category="Public")]
    cache = CachedNetworkWarning(lookup=lookup, clock=lambda: now[0], max_age_s=60)
    assert "RedeBaja" in cache.get()
    now[0] = 30.0
    cache.get()                 # 30 s later: the remembered answer
    assert asked == [0.0]
    now[0] = 61.0
    cache.get()                 # more than a minute later: asked again
    assert asked == [0.0, 61.0]
