"""Tests for zone/zonefile.py: zone.json save / backup / load."""
import json

import pytest

from zone import zonefile
from zone.tests.z2_fixtures import straight_zone


def test_round_trip_keeps_everything(tmp_path):
    z = straight_zone()
    z.notes = "teste"
    z.report = {"branch": "A"}
    f = tmp_path / "zone.json"
    zonefile.save(z, f)
    back = zonefile.load(f)
    assert back.to_dict() == z.to_dict()
    assert back.entry_preset == 10                     # the first keyframe's preset
    assert back.points[1].keyframe.pan == 0 and back.points[3].keyframe.zoom == 2000


def test_every_save_backs_up_the_previous_file(tmp_path):
    f = tmp_path / "zone.json"
    z = straight_zone()
    assert zonefile.save(z, f) is None                 # nothing to back up the first time
    z.corridor_half_width_px = 77
    b1 = zonefile.save(z, f)
    z.corridor_half_width_px = 88
    b2 = zonefile.save(z, f)
    assert b1 != b2 and b1.exists() and b2.exists()
    # the backups hold the PREVIOUS versions: 60, then 77
    assert json.loads(b1.read_text())["corridor_half_width_px"] == 60
    assert json.loads(b2.read_text())["corridor_half_width_px"] == 77
    assert zonefile.load(f).corridor_half_width_px == 88
    assert not (tmp_path / "zone.json.tmp").exists()   # the temporary file was renamed into place


def test_next_free_preset_starts_at_base_and_skips_used():
    z = straight_zone()                                # presets 10, 11, 12
    assert z.next_free_preset(10) == 13
    assert z.next_free_preset(20) == 20


def test_newer_format_is_refused(tmp_path):
    f = tmp_path / "zone.json"
    f.write_text(json.dumps({"format_version": 99, "points": []}))
    with pytest.raises(ValueError):
        zonefile.load(f)
