"""CPU tests of the closed-loop result statistics (synthetic episodes)."""
import json
import os

import pytest

from tacinno.evaluation.results import compare, first_results, first_valid_seeds, mcnemar_exact, success_rate


def _write(run_dir, episodes):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "episodes.json"), "w") as f:
        json.dump(episodes, f)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    # 32 vs 4 discordant pairs: two-sided exact binomial tail
    from math import comb
    expected = 2 * sum(comb(36, i) for i in range(5)) / 2 ** 36
    assert mcnemar_exact(32, 4) == pytest.approx(expected)
    assert mcnemar_exact(4, 32) == mcnemar_exact(32, 4)


def test_first_finished_result_per_seed(tmp_path):
    # run 1 starts earlier; seed 2 errors there and is filled by seed 3
    _write(tmp_path / "2026-01-01_10:00:00", [
        {"seed": 1, "result": "success", "seconds": 10},
        {"seed": 2, "result": "error"},
        {"seed": 3, "result": "failed", "seconds": 10},
    ])
    # run 2 re-evaluates seed 1 later (ignored) and evaluates seed 2
    _write(tmp_path / "2026-01-01_11:00:00", [
        {"seed": 1, "result": "failed", "seconds": 10},
        {"seed": 2, "result": "success", "seconds": 10},
    ])
    res = first_results([tmp_path], seeds=range(1, 5))
    assert {s: e["result"] for s, e in res.items()} == {1: "success", 2: "success", 3: "failed"}
    s = success_rate(res, range(1, 5))
    assert (s["success"], s["valid"], s["missing"]) == (2, 3, [4])


def test_finished_at_takes_precedence(tmp_path):
    _write(tmp_path / "runA", [{"seed": 7, "result": "failed", "seconds": 5, "finished_at": 200.0}])
    _write(tmp_path / "runB", [{"seed": 7, "result": "success", "seconds": 5, "finished_at": 100.0}])
    res = first_results([tmp_path / "runA", tmp_path / "runB" / "episodes.json"])
    assert res[7]["result"] == "success"


def test_paired_comparison(tmp_path):
    a = [{"seed": s, "result": "success" if s % 2 == 0 or s < 4 else "failed", "seconds": 1} for s in range(10)]
    b = [{"seed": s, "result": "success" if s % 2 == 0 else "failed", "seconds": 1} for s in range(8)]
    _write(tmp_path / "a" / "2026-01-01_10:00:00", a)
    _write(tmp_path / "b" / "2026-01-01_10:00:00", b)
    ra, rb = first_results([tmp_path / "a"]), first_results([tmp_path / "b"])
    c = compare(ra, rb)
    assert c.paired == 8 and c.only_a == 2 and c.only_b == 0
    assert c.success_a == 6 and c.success_b == 4
    assert c.p_value == pytest.approx(mcnemar_exact(2, 0))


def test_first_valid_seeds_replace_errors_by_the_next_seed(tmp_path):
    _write(tmp_path / "2026-01-01_10:00:00", [
        {"seed": 10, "result": "success", "seconds": 1},
        {"seed": 11, "result": "error"},
        {"seed": 12, "result": "failed", "seconds": 1},
        {"seed": 13, "result": "success", "seconds": 1},
        {"seed": 14, "result": "success", "seconds": 1},
    ])
    res = first_results([tmp_path])
    assert first_valid_seeds(res, start=10, num=3) == [10, 12, 13]
    s = success_rate(res, first_valid_seeds(res, start=10, num=3))
    assert (s["success"], s["valid"]) == (2, 3)
