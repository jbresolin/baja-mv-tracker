"""Two bugs found on the zone bench of 2026-11-15 (the installed laptop's zone log):

1. Every request for the page's state failed with "Object of type bool is not JSON serializable": a track's
   "predicted" flag was a numpy bool, which Python's JSON encoder refuses. The page then stopped refreshing.
2. The zone ran on the simulator's DEFAULT speed tables because no characterization report was set in the config,
   although a fresh report.json was sitting in the data folder. Default tables are in other units than the real
   camera's, so every speed the controller asked for mapped to the top speed step and the PTZ ran to its end stops.
"""
import json
import os
import time

import numpy as np

from zone import config as C
from zone.app import plain_json


def test_page_state_with_numpy_values_turns_into_plain_json():
    # A made-up state with the kinds of numpy values the tracker produces: a numpy bool (the one that broke the
    # page), a numpy float, a numpy int and a small numpy array, nested inside dicts and lists.
    state = dict(track=dict(predicted=np.bool_(True), s=np.float64(0.25), id=np.int64(7)),
                 boxes=[np.array([1.0, 2.0])], label="ok")
    plain = plain_json(state)
    # json.dumps raises an error for numpy types; after plain_json it must work and keep the values
    text = json.dumps(plain)
    assert json.loads(text) == dict(track=dict(predicted=True, s=0.25, id=7), boxes=[[1.0, 2.0]], label="ok")


def write_report(folder, name, mtime):
    # A minimal characterization report: just a verdict. Its modification time decides which one is "newest".
    report_dir = folder / "reports" / name
    report_dir.mkdir(parents=True)
    path = report_dir / "report.json"
    path.write_text(json.dumps({"verdict": {"branch": "B", "reason": name}}), encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_without_a_configured_report_the_newest_one_in_the_data_folder_is_used(tmp_path):
    # Two reports in the data folder (an older and a newer run); none configured. The zone must pick the newer one
    # instead of silently falling back to the default speed tables.
    now = time.time()
    write_report(tmp_path, "ptz_old", now - 3600)
    newest = write_report(tmp_path, "ptz_new", now)
    report = C.load_report("", data_root=str(tmp_path))
    assert report.path == str(newest)
    assert "newest report in the data folder" in report.reason


def test_with_no_report_anywhere_the_reason_says_default_tables_loudly(tmp_path):
    # An empty data folder and no configured report: defaults are still used, and the reason says so clearly.
    report = C.load_report("", data_root=str(tmp_path))
    assert report.path == ""
    assert "DEFAULT speed tables" in report.reason
