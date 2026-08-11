"""
data_loader.py

The type-normalization boundary between the canonical CSVs and scoring.py.
scoring.py owns the *model* (columns, composite, comparisons) in canonical
snake_case; this module owns *I/O and dtype hygiene* only. It does NOT rename
columns to any legacy vocabulary -- the canonical snake_case names flow through
untouched. Its one substantive job is the season-typing coercion below.

Why season must be a string:
  The canonical CSVs store `season` as an int64 (e.g. 1997). Every era/cutoff
  routine in scoring.py compares seasons as strings against boundary literals
  in each league's native format (WNBA bare year '1997', NBA cross-year
  '1979-80'). Comparing a str boundary against an int season is a TypeError in
  Python 3, so we coerce `season` to its native-format string once, here, at the
  load boundary -- and keep all three frames on the same season dtype so the
  (season, team_code) merges in scoring.py line up.
"""

from pathlib import Path
import pandas as pd

PROCESSED_ROOT = Path("data/processed")


def _to_native_season(df):
    """Coerce `season` to the league's native-format string.

    WNBA seasons are single calendar years, so an int 1997 -> '1997'. (NBA
    canonical files, when they exist, already carry the '1979-80' cross-year
    string, so str() is a no-op there.)"""
    if "season" in df.columns and df["season"].dtype != object:
        df["season"] = df["season"].astype("Int64").astype(str)
    return df


def load_league_data(league, processed_root=PROCESSED_ROOT):
    """Load and type-normalize the three canonical CSVs for a league.

    Returns (team_stats, player_stats, playoff_results) with:
      - season coerced to native-format string on all three
      - team_code / player_name coerced to str (guards against all-numeric
        codes being read back as ints, and NaN team_codes becoming the float
        'nan')
    scoring.py can then consume these directly -- no adapter, no aliasing.
    """
    base = Path(processed_root) / league
    team_stats = pd.read_csv(base / "team_season_stats.csv")
    player_stats = pd.read_csv(base / "player_season_stats.csv")
    playoff_results = pd.read_csv(base / "playoff_results.csv")

    for df in (team_stats, player_stats, playoff_results):
        _to_native_season(df)

    for df in (team_stats, player_stats, playoff_results):
        if "team_code" in df.columns:
            df["team_code"] = df["team_code"].astype(str)
    if "player_name" in player_stats.columns:
        player_stats["player_name"] = player_stats["player_name"].astype(str)

    return team_stats, player_stats, playoff_results
