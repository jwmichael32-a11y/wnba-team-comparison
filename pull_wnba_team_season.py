"""
pull_wnba_team_season.py  --  LAYER 1 (raw pull only)

Pulls WNBA team-season data from nba_api and caches each raw measure-type
response permanently under data/raw/WNBA/. That is ALL this script does --
it does NOT transform, shape, or write canonical CSVs. Canonical shaping is
LAYER 2 (build_canonical_team.py), which reads the cache this produces.

This separation means:
  - re-running the pull only re-hits the API for seasons not yet cached
  - re-shaping canonical costs zero API calls (build step reads cache)
  - a pull bug and a transform bug can never masquerade as each other

Run locally:
    python pull_wnba_team_season.py --seasons 1997-2025
    python pull_wnba_team_season.py --seasons 2024,2025 --force
"""
import argparse
from nba_api.stats.endpoints import leaguedashteamstats
from raw_cache import cached_fetch

WNBA = '10'

# measure types pulled and cached IN FULL (pull wide, derive narrow).
MEASURE_TYPES = {
    'base': 'Base',
    'advanced': 'Advanced',
    'four_factors': 'Four Factors',
    'opponent': 'Opponent',
}


def _fetch_measure(season, measure_type):
    def _do():
        ep = leaguedashteamstats.LeagueDashTeamStats(
            league_id_nullable=WNBA,
            season=str(season),               # WNBA bare-year format
            season_type_all_star='Regular Season',
            per_mode_detailed='Totals',
            measure_type_detailed_defense=measure_type,
        )
        return ep.get_data_frames()[0].to_dict('records')
    return _do


def pull_season(season, force=False):
    for key, mt in MEASURE_TYPES.items():
        cached_fetch('WNBA', f'team_{key}', str(season),
                     _fetch_measure(season, mt), force=force)


def parse_seasons(arg):
    if '-' in arg and ',' not in arg:
        start, end = arg.split('-')
        return [str(y) for y in range(int(start), int(end) + 1)]
    return [s.strip() for s in arg.split(',')]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', required=True, help='e.g. 1997-2025 or 2024,2025')
    ap.add_argument('--force', action='store_true', help='re-pull even if cached')
    args = ap.parse_args()

    for season in parse_seasons(args.seasons):
        print(f"[{season}] pulling team measure types...")
        pull_season(season, force=args.force)
    print("\nDone. Raw cache populated. Run build_canonical_team.py to shape canonical CSV.")
