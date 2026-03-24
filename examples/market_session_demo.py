"""
Demonstrate SessionPolicy: what gets fetched vs what rows the strategy keeps.

Run from repo root::

    python examples/market_session_demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import polars as pl

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from framework.data_handling.market_session import SessionPolicy, apply_session_policy


def main() -> None:
    # Naive UTC (typical for Yahoo intraday): 14:30 UTC = 10:30 AM EDT mid-RTH.
    sample = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 6, 3, 11, 0),  # Mon pre-market ET (UTC)
                datetime(2024, 6, 3, 14, 30),  # Mon mid-RTH ET
                datetime(2024, 6, 1, 16, 0),  # Sat UTC (weekend)
            ],
            "close": [100.0, 101.0, 99.0],
        }
    )

    print(
        "Synthetic timestamps are naive UTC (like Yahoo). "
        "US_EQUITY_RTH converts to America/New_York before the 09:30-16:00 filter.\n"
    )

    for policy in (
        SessionPolicy.CRYPTO_UTC_24H,
        SessionPolicy.US_EQUITY_EXTENDED,
        SessionPolicy.US_EQUITY_RTH,
    ):
        out = apply_session_policy(
            sample,
            policy,
            naive_timestamp_tz="UTC",
        )
        print(f"{policy.value}: {out.height} row(s) kept")
        if not out.is_empty():
            print(f"  -> timestamps kept: {out['timestamp'].to_list()}")
        print()

    print(
        "Yahoo (yfinance): prepost=False for RTH and crypto; prepost=True only for "
        "US_EQUITY_EXTENDED. DataHandler applies US_EQUITY_RTH after load so extended "
        "bars are dropped if they appear in the file."
    )


if __name__ == "__main__":
    main()
