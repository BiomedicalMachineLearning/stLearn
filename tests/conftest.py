"""Options shared by the test suite.

Tests marked `acceptance` take minutes, so pyproject.toml deselects them by
default; select them with `pytest -m acceptance`.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("acceptance", "acceptance tests (pytest -m acceptance)")
    group.addoption(
        "--acceptance-baseline",
        choices=["compare", "write"],
        default="compare",
        help="compare results with the stored baselines (default), or write new "
        "baselines from this checkout's code",
    )
    group.addoption(
        "--acceptance-results",
        metavar="DIR",
        default="tests/acceptance-results",
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
        "--acceptance-max-minutes",
        type=float,
        default=30.0,
        help="stop a scenario still running after this many minutes (default 30)",
    )
    group.addoption(
        "--acceptance-max-memory-gb",
        type=float,
        default=16.0,
        help="stop a scenario whose memory goes over this many GB (default: 16.0GB "
        "limit)",
    )
