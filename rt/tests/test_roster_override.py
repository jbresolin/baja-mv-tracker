"""identity.rosters_dir (AUTOPTZ-26): an extra roster folder that rt searches first.

The installed app keeps the code (with its bundled rosters/) in a read-only Program Files folder. Rosters fetched on
site go to the data folder instead, and identity.rosters_dir points rt there. The rule tested here:
  * a file for the event in that folder wins;
  * no file there -> the roster next to the code is used, exactly as before;
  * the setting is empty by default, so a repo run behaves as it always did.
"""
from rt import config as C


def write_roster(folder, event, numbers):
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["numero,equipe"] + ["%d,Equipe %d" % (n, n) for n in numbers]
    (folder / ("%s.csv" % event)).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_extra_folder_wins_when_it_has_the_event(tmp_path):
    write_roster(tmp_path / "bundled", "27SE", [1, 2, 3])
    write_roster(tmp_path / "fetched", "27SE", [10, 20])
    roster = C.load_roster("27SE", rosters_dir=tmp_path / "bundled", override_dir=tmp_path / "fetched")
    assert roster["numbers"] == [10, 20]


def test_without_a_file_there_the_bundled_roster_is_used(tmp_path):
    write_roster(tmp_path / "bundled", "27SE", [1, 2, 3])
    (tmp_path / "fetched").mkdir()
    roster = C.load_roster("27SE", rosters_dir=tmp_path / "bundled", override_dir=tmp_path / "fetched")
    assert roster["numbers"] == [1, 2, 3]


def test_the_setting_is_empty_by_default(cfg):
    # `cfg` is rt/config.toml freshly loaded (rt/tests/conftest.py): nothing changes for a repo run
    assert cfg["identity"]["rosters_dir"] == ""
    assert C.pick_rosters_dir("26SE", None, cfg["identity"]["rosters_dir"] or None) is None


def test_the_no_roster_warning_names_the_launcher_button_in_the_installed_app(tmp_path):
    # Repo runs keep the old advice (the command line); the installed app points at its page's button.
    repo_run = C.load_roster("99XX", rosters_dir=tmp_path)
    installed = C.load_roster("99XX", rosters_dir=tmp_path, override_dir=tmp_path / "data_rosters")
    assert "python teacher/rosters.py 99XX" in repo_run["warning"]
    assert "Baixar lista de equipes" in installed["warning"]
