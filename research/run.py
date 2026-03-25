"""
Run a research project by mach number or folder name
====================================================

From the repository root::

    uv run python research/run.py 4
    uv run python research/run.py mach4
    uv run python research/run.py mach4_ema_band_ep1

Resolves ``research/mach{N}_*`` when you pass a numeric id or ``mach{N}``. If more
than one folder matches, pass the **full** project directory name.

Loads ``configs/config.py`` and runs enabled suites (e.g. in-sample excellence) via
:mod:`research.research_runner`. Project ``main.py`` is optional; you can keep it as a
thin wrapper or for legacy use.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys


def _research_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _glob_unique_mach(research_dir: str, mach_index: str) -> str:
    pattern = os.path.join(research_dir, f"mach{mach_index}_*")
    paths = sorted(p for p in glob.glob(pattern) if os.path.isdir(p))
    if not paths:
        raise FileNotFoundError(
            f"No directory matching {os.path.basename(pattern)!r} under {research_dir!r}"
        )
    if len(paths) > 1:
        names = [os.path.basename(p) for p in paths]
        raise RuntimeError(
            "Multiple projects match this mach id — use the full folder name:\n  "
            + "\n  ".join(names)
        )
    return paths[0]


def resolve_project_dir(research_dir: str, target: str) -> str:
    """
    Resolve ``target`` to an absolute project directory under ``research_dir``.

    Accepts:

    - Full folder name: ``mach4_ema_band_ep1`` (must exist)
    - Integer string: ``4`` → unique ``mach4_*``
    - Short form: ``mach4`` → unique ``mach4_*``
    """
    t = target.strip()
    if not t:
        raise ValueError("empty target")

    direct = os.path.join(research_dir, t)
    if os.path.isdir(direct):
        return os.path.abspath(direct)

    if t.isdigit():
        return _glob_unique_mach(research_dir, t)

    m = re.match(r"^mach(\d+)$", t, re.IGNORECASE)
    if m:
        return _glob_unique_mach(research_dir, m.group(1))

    raise FileNotFoundError(
        f"No research project {t!r} (expected a folder under {research_dir!r}, "
        "a number like 4, or mach4)"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a research project using configs/config.py (enabled suites)."
    )
    parser.add_argument(
        "target",
        help="Project folder name (e.g. mach4_ema_band_ep1), or mach index (4 / mach4)",
    )
    args = parser.parse_args(argv)

    research_dir = _research_dir()
    repo_root = os.path.dirname(research_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    project_dir = resolve_project_dir(research_dir, args.target)
    project_dir = os.path.abspath(project_dir)
    if sys.path[0] != project_dir:
        sys.path.insert(0, project_dir)

    from research.research_runner import run_cli

    run_cli(project_dir, repo_root=repo_root)


if __name__ == "__main__":
    main()
