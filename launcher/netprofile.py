"""Windows network category (Public / Private / Domain) of the active connections.

Why it matters: Windows treats a network it does not know as "Public", and a race LAN without internet usually ends
up Public. The installer's firewall rules cover every profile (private, public and domain), but on a Public network
Windows may still hide this PC from the others (network discovery off) and some company laptops block every incoming
connection on Public networks by policy. So the self-test and the launcher's page say so, with the command that marks
the network as Private.

Everything that touches Windows (running PowerShell) is injectable, so the tests run anywhere.
"""
import json
import subprocess
import sys
import time

POWERSHELL_QUERY = ("Get-NetConnectionProfile | Select-Object Name, InterfaceAlias, NetworkCategory | "
                    "ConvertTo-Json -Compress")

# Get-NetConnectionProfile gives NetworkCategory as a name ("Public") or, in older PowerShell, as a number.
CATEGORY_NAMES = {0: "Public", 1: "Private", 2: "DomainAuthenticated"}


def fix_command(name):
    """The PowerShell command (run as administrator) that marks one network as Private."""
    return "Set-NetConnectionProfile -Name '%s' -NetworkCategory Private" % str(name).replace("'", "''")


def run_powershell(command, timeout=15):
    """Run one PowerShell command without a console window flashing up; -> its stdout ("" on any failure)."""
    from launcher.procs import windows_flags
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                capture_output=True, text=True, timeout=timeout,
                                creationflags=windows_flags(visible=False))
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def parse_profiles(text):
    """PowerShell's JSON (one object, or a list of them) -> [dict(name, interface, category)]."""
    try:
        data = json.loads(text) if text and text.strip() else []
    except ValueError:
        return []
    if isinstance(data, dict):
        data = [data]
    profiles = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        category = item.get("NetworkCategory")
        category = CATEGORY_NAMES.get(category, category) if isinstance(category, int) else str(category or "")
        profiles.append(dict(name=str(item.get("Name") or ""), interface=str(item.get("InterfaceAlias") or ""),
                             category=category))
    return profiles


def public_networks(runner=run_powershell, platform=None):
    """The active connections Windows calls Public ([] elsewhere, or when it cannot be asked)."""
    if (platform or sys.platform) != "win32":
        return []
    return [p for p in parse_profiles(runner(POWERSHELL_QUERY)) if p["category"].lower() == "public"]


def public_network_warning(networks):
    """A pt-BR warning for the page / self-test, or None when no network is Public."""
    if not networks:
        return None
    names = ", ".join("%s (%s)" % (n["name"] or "?", n["interface"] or "?") for n in networks)
    commands = "; ".join(fix_command(n["name"]) for n in networks if n["name"])
    return ("Rede marcada como Pública pelo Windows: %s. Se outro computador não conseguir abrir as páginas deste, "
            "marque a rede como Privada (PowerShell como administrador): %s" % (names, commands))


class CachedNetworkWarning:
    """public_network_warning() asked at most once a minute (the page polls every few seconds and PowerShell takes
    about half a second to start)."""

    def __init__(self, lookup=public_networks, clock=time.monotonic, max_age_s=60.0):
        self.lookup, self.clock, self.max_age_s = lookup, clock, max_age_s
        self.value, self.when = None, None

    def get(self):
        now = self.clock()
        if self.when is None or now - self.when > self.max_age_s:
            try:
                self.value = public_network_warning(self.lookup())
            except Exception:  # noqa: BLE001 (a warning must never break the page)
                self.value = None
            self.when = now
        return self.value
