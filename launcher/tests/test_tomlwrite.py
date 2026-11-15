"""Tests for launcher/tomlwrite.py, the small TOML writer.

The safest check for a writer is a "round trip": write a dict as TOML text, read that text back with Python's own
TOML reader (tomllib), and compare with the original dict. If they are equal, the writer wrote valid TOML that
means the same thing.
"""
import tomllib

import pytest

from launcher.tomlwrite import dumps, format_key, format_value


def round_trip(data):
    return tomllib.loads(dumps(data))


def test_simple_values_survive_a_round_trip():
    data = {"name": "lapline", "port": 8765, "ratio": 0.5, "on": True, "off": False, "big": 1e-7,
            "list": [1, 2, 3], "nested_list": [[0, 1], [2, 3]], "words": ["car", "truck"]}
    assert round_trip(data) == data


def test_tables_and_sub_tables():
    # [camera.settings] inside [camera], and a table that only holds tables ([profiles] here)
    data = {"profile": "", "camera": {"device": 0, "settings": {"exposure": -10, "gain": 64}},
            "profiles": {"strict": {"controller": {"k_quant": 0.0}}}}
    assert round_trip(data) == data


def test_windows_paths_and_special_characters_in_strings():
    # Backslashes, quotes, accents and the "#" of a DirectShow device path must all come back unchanged.
    data = {"paths": {"out": "C:\\BajaMV\\out\\lapline", "device": "usb#vid_046d&pid_085e&mi_00#7&2f0c1a3b",
                      "quote": 'say "hi"', "accent": "Câmera ção", "newline": "a\nb"}}
    assert round_trip(data) == data


def test_top_level_keys_are_written_before_any_table():
    # In TOML a key written after [a table] belongs to that table, so plain keys must come first.
    text = dumps({"zone": {"file": "x"}, "profile": "bench"})
    assert text.index("profile") < text.index("[zone]")


def test_keys_that_need_quotes_get_them():
    assert format_key("ntp_server") == "ntp_server"
    assert format_key("has space") == '"has space"'


def test_floats_always_look_like_floats():
    # 1.0 must not become "1" (that would read back as an integer)
    assert format_value(1.0) == "1.0"
    assert isinstance(round_trip({"x": 1.0})["x"], float)


def test_values_toml_cannot_hold_are_refused():
    with pytest.raises(ValueError):
        format_value(float("nan"))
    with pytest.raises(ValueError):
        format_value(object())


def test_the_header_becomes_comments():
    text = dumps({"a": 1}, header="line one\nline two")
    assert text.startswith("# line one\n# line two\n")
    assert tomllib.loads(text) == {"a": 1}
