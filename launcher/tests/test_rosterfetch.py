"""Tests for the "Baixar lista de equipes" button (launcher/rosterfetch.py) and for how rt then finds the roster.

The results site is replaced by a fake page function, so no network is used. The fake page has the same table
layout as the real one (position, car number, team<br><i>school</i>), which teacher/rosters.py parses.
"""
from rt import config as rt_config

from launcher.rosterfetch import fetch_roster, roster_status

RESULTS_PAGE = """<html><table><tbody>
<tr><td>1</td><td>12</td><td>Equipe Doze<br><i>Escola A</i></td></tr>
<tr><td>2</td><td>7</td><td>Equipe Sete<br><i>Escola B</i></td></tr>
<tr><td>3</td><td>12</td><td>Equipe Doze<br><i>Escola A</i></td></tr>
</tbody></table></html>"""
EMPTY_PAGE = "<html><table><tbody></tbody></table></html>"


def no_sleep(seconds):
    pass


def test_fetched_roster_is_saved_in_the_data_folder(paths):
    asked = []

    def fake_page(url):
        asked.append(url)
        return RESULTS_PAGE
    result = fetch_roster("27se", paths, get_page=fake_page, sleep=no_sleep)
    assert result["ok"] is True
    assert result["cars"] == 2                                 # car 12 listed twice = one car
    saved = paths.rosters_dir / "27SE.csv"
    assert result["path"] == str(saved)
    assert saved.read_text(encoding="utf-8").splitlines()[0] == "numero,equipe,escola,fonte"
    assert "27SE_GER" in asked[0]                              # the "Geral" page first
    assert not (paths.bundled_rosters_dir / "27SE.csv").exists()   # the code's own rosters are not touched


def test_no_roster_yet_is_a_normal_answer_with_an_explanation(paths):
    result = fetch_roster("27SE", paths, get_page=lambda url: EMPTY_PAGE, sleep=no_sleep)
    assert result["ok"] is False
    assert "Ainda não há lista" in result["message"]
    assert not (paths.rosters_dir / "27SE.csv").exists()


def test_network_problems_become_a_message_not_a_crash(paths):
    def offline(url):
        raise OSError("no route to host")
    result = fetch_roster("27SE", paths, get_page=offline, sleep=no_sleep)
    assert result["ok"] is False
    assert "no route to host" in result["message"]


def test_an_empty_event_code_is_refused(paths):
    assert fetch_roster("", paths)["ok"] is False


def test_status_prefers_the_downloaded_roster_over_the_bundled_one(paths):
    # 26SE is bundled with the code. Before any download, the status says "embutida" (bundled) ...
    assert roster_status("26SE", paths)["where"] == "embutida"
    # ... after a download into the data folder, that one wins.
    fetch_roster("26SE", paths, get_page=lambda url: RESULTS_PAGE, sleep=no_sleep)
    status = roster_status("26SE", paths)
    assert status["where"] == "baixada" and status["cars"] == 2
    assert roster_status("99XX", paths)["where"] is None


def test_rt_reads_the_downloaded_roster_first(paths):
    # rt/config.load_roster with override_dir (identity.rosters_dir, written by the launcher) uses the downloaded
    # file when there is one, and the bundled roster otherwise.
    bundled = rt_config.load_roster("26SE", override_dir=str(paths.rosters_dir))
    assert bundled["open_set"] is False and len(bundled["numbers"]) > 2
    fetch_roster("26SE", paths, get_page=lambda url: RESULTS_PAGE, sleep=no_sleep)
    downloaded = rt_config.load_roster("26SE", override_dir=str(paths.rosters_dir))
    assert downloaded["numbers"] == [7, 12]
    assert downloaded["path"] == str(paths.rosters_dir / "26SE.csv")


def test_rt_without_any_roster_still_runs_in_open_set_mode(paths):
    result = rt_config.load_roster("99XX", override_dir=str(paths.rosters_dir))
    assert result["open_set"] is True
