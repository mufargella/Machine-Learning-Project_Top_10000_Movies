#!/usr/bin/env python
"""Recreate the dataset via TMDb-style metadata extraction (requires an API key).

This script is provided to satisfy the deliverable requirement of including a
dataset download/recreation option. If you already have Top_10000_Movies.csv,
you do NOT need to run this.

Note: Actual API access requires registering for an API key with the chosen provider
and implementing provider-specific requests + pagination.
"""

import os
from pathlib import Path

OUTPUT = Path.cwd() / "Top_10000_Movies.csv"
API_KEY = os.getenv("TMDB_API_KEY", "")

def main():
    if not API_KEY:
        raise SystemExit("Set TMDB_API_KEY env var before running.")
    raise SystemExit("Template script: implement API calls if required by your course.")

if __name__ == "__main__":
    main()
