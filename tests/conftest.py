"""Options shared by the test suite.

Tests marked `acceptance` take minutes, so `pytest` skips them; select them with
`pytest -m acceptance`.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("acceptance", "acceptance tests (pytest -m acceptance)")
    group.addoption(
        "--acceptance-baseline",
        choices=["compare", "write"],
        default="compare",
        help="compare results with the stored baselines (default), or write new "
        "baselines, e.g. from master with --acceptance-stlearn",
    )
    group.addoption(
        "--acceptance-stlearn",
        metavar="PATH",
        help="stLearn source tree to test, e.g. a git worktree of master "
        "(default: this checkout)",
    )
    group.addoption(
        "--acceptance-results",
        metavar="DIR",
        default="acceptance-results",
        help="folder for the latest results, history.csv and benchmark.json",
    )
    group.addoption(
        "--acceptance-bins",
        type=int,
        default=20000,
        help="number of bins in the synthetic Visium HD sample",
    )
    group.addoption(
        "--acceptance-atol",
        type=float,
        default=0.0,
        help="largest difference allowed from the baseline (default 0: identical)",
    )
    group.addoption(
        "--acceptance-max-time-ratio",
        type=float,
        help="fail if the run takes longer than this times the baseline's",
    )
    group.addoption(
        "--acceptance-max-memory-ratio",
        type=float,
        help="fail if the peak memory is more than this times the baseline's",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "acceptance: takes minutes; run with pytest -m acceptance"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if "acceptance" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="takes minutes; run with pytest -m acceptance")
    for item in items:
        if "acceptance" in item.keywords:
            item.add_marker(skip)
