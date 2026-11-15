"""A tiny TOML writer for the launcher's config files. Python's tomllib reads TOML but cannot write it.

Supports what rt and zone configs use: strings, booleans, integers, floats, lists of those (and lists of lists),
and nested tables ([camera.settings], [rt.capture], [profiles.strict.controller]). Comments of an existing file are
not kept (the launcher writes a header saying so); keys are, because the file is read, changed and written back.
"""
import json
import math
import re

BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def format_key(key):
    """A key as TOML wants it: bare when it only has letters, digits, _ and -, else quoted."""
    key = str(key)
    return key if BARE_KEY.match(key) else json.dumps(key, ensure_ascii=False)


def format_value(value):
    """One value as TOML text. JSON string escapes (\\" \\\\ \\n \\uXXXX) are valid TOML basic-string escapes."""
    if isinstance(value, bool):                      # before int: bool is a subclass of int
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("TOML writer: %r is not supported" % value)
        text = repr(value)
        return text if any(c in text for c in ".eE") else text + ".0"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    if isinstance(value, dict):                      # only reached for a dict inside a list: inline table
        return "{" + ", ".join("%s = %s" % (format_key(k), format_value(v)) for k, v in value.items()) + "}"
    raise ValueError("TOML writer: cannot write a %s (%r)" % (type(value).__name__, value))


def dumps(data, header=""):
    """dict -> TOML text. Plain keys of a table come first, then its sub-tables, each with a [dotted.header]."""
    lines = [("# " + line).rstrip() for line in header.splitlines()] if header else []
    _write_table(lines, data, [])
    return "\n".join(lines).strip("\n") + "\n"


def _write_table(lines, table, path):
    plain = {k: v for k, v in table.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in table.items() if isinstance(v, dict)}
    if path and (plain or not tables):
        lines.append("")
        lines.append("[" + ".".join(format_key(part) for part in path) + "]")
    for key, value in plain.items():
        lines.append("%s = %s" % (format_key(key), format_value(value)))
    for key, value in tables.items():
        _write_table(lines, value, path + [key])
