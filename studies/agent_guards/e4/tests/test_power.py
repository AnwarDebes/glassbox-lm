"""Tests of E4.3's power (scripts/power_e43.py; revision 3; the decision 1; re-review item 0): the exact
trinomial sum on hand-checked tiny tables, a probability that rises with the favourable rate, need(a) equal to the
published power table, the review's numbers (about 12% at the go-bar corner G 8 / L 4, about 17% at a net 4 of 74
with nothing against, about +13.4 points for 80% power with nothing against, about +20 with 2.5% against), and
gobar.power_block on the pinned file. Revision 4 (round-3 review m1, m5(a); scripts/power_rev4.py): E4.2's need table
(equal to the published one, and to power_table.min_favourable beyond it), its 80% power effects (the review's +4.8,
+9.8 and +12.1 points), the go-bar boundary G - L = 4 from G 8 to G 39 (down to 4.2%), and the pinned rev4 file.
Run: offline_env.sh .../taubench/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider tests/test_power.py"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import json
import math
import sys
from pathlib import Path

import pytest

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

import power_e43 as P  # noqa: E402

PINNED = E4 / "results/power_e43_rev3.json"


@pytest.fixture(scope="module")
def need():
    return P.need_table()


def _brute(g, l, need, n):  # noqa: E741
    tot = 0.0
    for a in range(n + 1):
        for f in range(n - a + 1):
            if need[a] is not None and f >= need[a]:
                tot += (math.factorial(n) / (math.factorial(f) * math.factorial(a) * math.factorial(n - f - a))
                        * g ** f * l ** a * (1 - g - l) ** (n - f - a))
    return tot


def test_exact_power_on_tiny_tables():
    for need in ([1, 2, None, None, None, None], [0, 0, 0, 0, 0, 0], [3, None, None, None, None, None]):
        for g, l in ((0.3, 0.1), (0.5, 0.0), (0.0, 0.4), (0.2, 0.2)):  # noqa: E741
            assert P.power(g, l, need, 5) == pytest.approx(_brute(g, l, need, 5), abs=1e-12)
    assert P.power(0.2, 0.1, [0] * 41, 40) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        P.power(0.7, 0.4, [0] * 41, 40)


def test_power_rises_with_g_and_falls_with_l(need):
    ps = [P.power(g / 100, 0.02, need) for g in range(2, 60, 4)]
    assert all(b >= a for a, b in zip(ps, ps[1:]))
    qs = [P.power(0.2, l / 100, need) for l in range(0, 20, 2)]  # noqa: E741
    assert all(b <= a for a, b in zip(qs, qs[1:]))


def test_need_table_equals_the_published_power_table(need):
    pt = json.loads((E4 / "results/power_table_rev2.json").read_text())["E4.3_40_tasks"]
    assert [need[r["against"]] for r in pt] == [r["favourable_needed"] for r in pt] == [4, 7, 9, 10, 13, 14, 16]
    assert need[40] is None and all(x is None or x > a for a, x in enumerate(need))


def test_the_reviewers_numbers(need):
    assert P.power_at_counts(8, 4, need) == pytest.approx(0.12, abs=0.005)        # "12% at G 8, L 4"
    assert P.power(4 / 74, 0.0, need) == pytest.approx(0.17, abs=0.005)           # "about 17% at that threshold"
    e0 = P.effect_for_power(0.80, 0.0, need)
    assert e0["net_points"] == pytest.approx(13.4, abs=0.1) and e0["net_tasks_of_40"] == pytest.approx(5.3, abs=0.1)
    e1 = P.effect_for_power(0.80, 0.025, need)
    assert e1["net_points"] == pytest.approx(20.0, abs=0.6)                       # "about +20 points"
    assert P.power(e0["g"], 0.0, need) >= 0.80
    assert P.threshold_points()[:5] == [(8, 0), (8, 1), (8, 2), (8, 3), (8, 4)] and (9, 5) in P.threshold_points()


@pytest.mark.skipif(not PINNED.exists(), reason="results/power_e43_rev3.json not written yet")
def test_pinned_file_and_the_gobar_power_block(need):
    import gobar as GB

    d = json.loads(PINNED.read_text())
    assert d["need_favourable_by_against"] == need
    assert d["at_go_bar_corner_G8_L4"] == pytest.approx(P.power_at_counts(8, 4, need))
    pb = GB.power_block({"tm_seed1": {"G": 8, "L": 4}, "lr": {"G": 0, "L": 0}}, 74)
    assert pb["at_observed_rates"]["tm_seed1"] == pytest.approx(d["at_go_bar_corner_G8_L4"])
    assert pb["at_observed_rates"]["lr"] == 0.0 and pb["power_file_sha256"] and "low power" in pb["what"]
    mc = d["monte_carlo_check"]["G8_L4"]
    assert abs(mc["power"] - mc["exact"]) < 4 * mc["se"] + 0.01            # the layout effect is small


# ---------------------------------------------------- revision 4 (round-3 review m1 and m5(a)): scripts/power_rev4.py
PINNED4 = E4 / "results/power_rev4.json"


def test_e42_need_table_and_its_80pct_effects():
    import power_rev4 as P4

    need42 = P4.need42_table()
    pt = json.loads((E4 / "results/power_table_rev2.json").read_text())["E4.2_114_tasks"]
    assert [need42[r["against"]] for r in pt] == [r["favourable_needed"] for r in pt] == [4, 7, 9, 11, 13, 14, 16]
    from power_table import min_favourable

    for a in (7, 10, 15):                                     # the skipped bootstrap calls change nothing
        assert need42[a] == min_favourable(114, a, 3.0)["favourable_needed"]
    e = [P.effect_for_power(0.80, l, need42, n=114) for l in (0.0, 0.025, 0.05)]  # noqa: E741
    # the round-3 review's own recompute: +4.8 (5.5 net tasks), +9.8, +12.1 points
    assert [round(x["net_points"], 1) for x in e] == [4.8, 9.8, 12.1]
    assert 114 * e[0]["g"] == pytest.approx(5.45, abs=0.05)


def test_the_go_bar_boundary_runs_to_g39():
    import power_rev4 as P4

    pts = P4.boundary_points()
    assert pts[0] == (8, 4) and pts[-1] == (39, 35) and len(pts) == 32 and all(g - l == 4 for g, l in pts)  # noqa: E741
    need = P.need_table()
    assert P.power_at_counts(39, 35, need) == pytest.approx(0.042, abs=0.0005)      # "down to about 4%"
    assert P.power_at_counts(24, 20, need) == pytest.approx(0.0576, abs=0.0005)


@pytest.mark.skipif(not PINNED4.exists(), reason="results/power_rev4.json not written yet")
def test_pinned_rev4_file():
    d = json.loads(PINNED4.read_text())
    assert [round(x["net_points"], 1) for x in d["E4.2"]["effect_for_80pct_power"]] == [4.8, 9.8, 12.1]
    lo = d["E4.3_go_bar_boundary"]["lowest"]
    assert (lo["G"], lo["L"]) == (39, 35) and round(100 * lo["power"], 1) == 4.2
    assert d["E4.3_go_bar_boundary"]["first"]["power"] == pytest.approx(P.power_at_counts(8, 4, P.need_table()))
    mc = d["E4.2"]["monte_carlo_check_80pct_l0"]
    assert abs(mc["power"] - mc["exact"]) < 4 * mc["se"] + 0.01
