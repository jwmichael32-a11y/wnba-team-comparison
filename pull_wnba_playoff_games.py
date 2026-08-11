"""
pull_wnba_playoff_games.py  --  LAYER 1 (raw pull only)

Pulls WNBA playoff GAME-level data (LeagueGameLog, Playoffs) per season and
caches it under data/raw/WNBA/playoff_games/. Pull-and-cache only; round
reconstruction is build_canonical_playoffs.py (Layer 2).

Covers 2001-2025 (the API-covered range). 1997-2000 is handled by the
existing manual file (playoff_results_manual.csv), reconstructed earlier.

Run locally:
    python pull_wnba_playoff_games.py --seasons 2001-2025
"""
import argparse
from nba_api.stats.endpoints import leaguegamelog
from raw_cache import cached_fetch

WNBA = '10'


def _fetch(season):
    def _do():
        ep = leaguegamelog.LeagueGameLog(
            league_id=WNBA, season=str(season),
            season_type_all_star='Playoffs',
        )
        return ep.get_data_frames()[0].to_dict('records')
    return _do


def pull_season(season, force=False):
    cached_fetch('WNBA', 'playoff_games', str(season), _fetch(season), force=force)


def parse_seasons(arg):
    if '-' in arg and ',' not in arg:
        a, b = arg.split('-')
        return [str(y) for y in range(int(a), int(b) + 1)]
    return [s.strip() for s in arg.split(',')]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', required=True)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    for s in parse_seasons(args.seasons):
        print(f"[{s}] pulling playoff games...")
        pull_season(s, force=args.force)
    print("\nDone. Next: build_canonical_playoffs.py")
