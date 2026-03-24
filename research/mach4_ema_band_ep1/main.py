"""
Mach4 EMA Band EP1 — research entry (optional)
============================================

Preferred: from repo root run ``python research/run_project.py mach4_ema_band_ep1``
(or ``4`` / ``mach4`` if unambiguous). That uses :mod:`research.research_runner` and
``tests/config.py`` — same behavior as this file.

Set ``MASSIVE_API_KEY`` for live fetches when the Parquet cache is missing.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if sys.path[0] != project_dir:
        sys.path.insert(0, project_dir)

    from research.research_runner import run_cli

    run_cli(project_dir, repo_root=repo_root)


if __name__ == "__main__":
    main()
