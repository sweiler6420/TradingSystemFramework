"""
Shared research orchestration driven by each project's ``tests/config.py``.

``run.py`` calls :func:`run_project_from_config` so logic is not duplicated in
per-project ``main.py`` files.
"""

from __future__ import annotations

import importlib
import os
import sys
from datetime import date, datetime
from typing import Any

import polars as pl

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_import_paths(project_dir: str, repo_root: str) -> None:
    pd = os.path.abspath(project_dir)
    rr = os.path.abspath(repo_root)
    if sys.path[0] != pd:
        sys.path.insert(0, pd)
    if rr not in sys.path:
        sys.path.insert(0, rr)


def _parse_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d).strip()[:10])


def _session_policy(name: str):
    from framework.data_handling.market_session import SessionPolicy

    try:
        return SessionPolicy[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown session_policy {name!r}; use a SessionPolicy member name, e.g. CRYPTO_UTC_24H"
        ) from e


def _filter_insample_window(df: pl.DataFrame, start_d: date, end_inclusive: date) -> pl.DataFrame:
    return df.filter(
        (pl.col("timestamp").dt.date() >= start_d)
        & (pl.col("timestamp").dt.date() <= end_inclusive)
    )


def _load_strategy(class_spec: str, data: pl.DataFrame):
    """
    ``class_spec``: ``"strategies.foo_strategy:BarStrategy"`` (module path under project root).
    """
    if ":" not in class_spec:
        raise ValueError(
            f"strategy must be 'module.submodule:ClassName', got {class_spec!r}"
        )
    mod_name, _, cls_name = class_spec.partition(":")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(data)


def _make_market_data_provider(provider_name: str, session):
    p = provider_name.lower()
    if p == "massive":
        from framework.data_sources import MassiveProvider

        return MassiveProvider(session_policy=session)
    raise ValueError(f"Unsupported provider {provider_name!r} (supported: massive)")


def run_insample_excellence(
    project_root: str,
    cfg: dict[str, Any],
    *,
    strategy_spec: str,
) -> list[dict[str, Any]]:
    """
    In-sample excellence suite: one run per symbol, driven by ``cfg`` (``TEST_CONFIG['insample_excellence']``).
    """
    from framework import DataHandler
    from framework.data_sources import ensure_cached
    from framework.data_sources.cache import safe_symbol_label

    from suites.insample_excellence import InSampleExcellenceSuite

    if not cfg.get("enabled", False):
        print("insample_excellence is disabled in tests/config.py — nothing to run.")
        return []

    symbols: list[str] = list(cfg.get("symbols") or [])
    if not symbols:
        print("insample_excellence.symbols is empty in tests/config.py — nothing to run.")
        return []

    interval = str(cfg.get("interval") or "1h")
    start_d = _parse_date(cfg["start"])
    end_inclusive = _parse_date(cfg["end"])
    if start_d > end_inclusive:
        raise ValueError(f"config start {start_d} must be <= end {end_inclusive}")

    cache_start = _parse_date(cfg["cache_start"]) if cfg.get("cache_start") else start_d
    cache_end = _parse_date(cfg["cache_end"]) if cfg.get("cache_end") else end_inclusive

    provider_name = str(cfg.get("provider") or "massive")
    session = _session_policy(str(cfg.get("session_policy") or "CRYPTO_UTC_24H"))
    provider = _make_market_data_provider(provider_name, session)

    cache_dir = os.path.join(project_root, "data")
    project_label = os.path.basename(project_root)

    print(f"=== {project_label.upper()} — IN-SAMPLE EXCELLENCE SUITE ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {symbols}  interval: {interval}")
    print(f"In-sample window (inclusive): {start_d} .. {end_inclusive}")
    print(f"Session policy: {session.value}")

    all_metadata: list[dict[str, Any]] = []

    for symbol in symbols:
        print(f"\n--- Symbol: {symbol} ---")
        data_path = ensure_cached(
            provider,
            symbol=symbol,
            interval=interval,
            start=cache_start,
            end=cache_end,
            cache_dir=cache_dir,
            session_policy=session,
        )
        print(f"Parquet: {data_path}")
        data_handler = DataHandler(str(data_path), session_policy=session)
        data_handler.load_data()
        raw = data_handler.get_data()
        filtered = _filter_insample_window(raw, start_d, end_inclusive)
        data_handler.data = filtered
        data = data_handler.get_data()

        if data.is_empty():
            print(f"Warning: no rows after filtering for {symbol}; skipping.")
            continue

        print(
            f"Data loaded: {data.height} rows from {data['timestamp'][0]} to {data['timestamp'][-1]}"
        )

        strategy = _load_strategy(strategy_spec, data)
        slug = safe_symbol_label(symbol)
        test_name = f"insample_excellence_{slug}"

        suite = InSampleExcellenceSuite(project_root, strategy)
        test_metadata = suite.run_test(data_handler, test_name)

        signal_result = strategy.generate_signals()
        suite.create_performance_plots(
            data,
            signal_result,
            test_metadata["performance_results"],
            test_name=test_name,
            test_metadata=test_metadata,
        )
        suite.generate_test_report(test_metadata, test_name=test_name)

        test_metadata["symbol"] = symbol
        test_metadata["test_name"] = test_name
        all_metadata.append(test_metadata)

    print(f"\n=== {project_label.upper()} — IN-SAMPLE EXCELLENCE COMPLETED ===")
    print("Outputs under results/<V####>/ (see tests/config.py).")

    return all_metadata


def _any_suite_enabled(tc: dict[str, Any]) -> bool:
    for v in tc.values():
        if isinstance(v, dict) and v.get("enabled"):
            return True
    return False


def run_project_from_config(project_dir: str, *, repo_root: str | None = None) -> list[dict[str, Any]]:
    """
    Load ``tests/config.py`` from ``project_dir`` and run enabled suites.

    ``repo_root`` is the trading framework repo root (parent of ``research/``). If omitted,
    inferred from this file's location.
    """
    rr = repo_root if repo_root is not None else _REPO_ROOT
    _ensure_import_paths(project_dir, rr)

    try:
        from tests import config as tests_config
    except ImportError as e:
        raise RuntimeError(
            f"No tests/config.py (or import error) in {project_dir!r}"
        ) from e

    tc = getattr(tests_config, "TEST_CONFIG", None)
    if tc is None:
        raise RuntimeError(f"tests/config.py must define TEST_CONFIG in {project_dir!r}")

    if not _any_suite_enabled(tc):
        print("No enabled suites in TEST_CONFIG — nothing to run.")
        return []

    out: list[dict[str, Any]] = []
    ie = tc.get("insample_excellence") or {}
    if ie.get("enabled"):
        strategy_spec = ie.get("strategy") or getattr(tests_config, "INSAMPLE_STRATEGY", None)
        if not strategy_spec:
            raise RuntimeError(
                "insample_excellence is enabled but no strategy is set. "
                "Add 'strategy': 'strategies.module:ClassName' to insample_excellence "
                "or define INSAMPLE_STRATEGY in tests/config.py."
            )
        out.extend(run_insample_excellence(project_dir, ie, strategy_spec=strategy_spec))

    return out


def run_cli(project_dir: str, *, repo_root: str | None = None) -> None:
    """CLI entry: run config and print completion."""
    pd = os.path.abspath(project_dir)
    label = os.path.basename(pd)
    print(f"Starting research: {label}")
    run_project_from_config(pd, repo_root=repo_root)
    print(f"\n{label} research completed.")
