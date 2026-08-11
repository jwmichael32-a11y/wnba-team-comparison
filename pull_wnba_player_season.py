"""
pull_wnba_player_season.py  --  LAYER 1 (raw pull only)

Player-season analog of pull_wnba_team_season.py. Pulls WNBA player-season
data via LeagueDashPlayerStats and caches each raw measure-type response
under data/raw/WNBA/player_*. Pull-and-cache ONLY; canonical shaping is
build_canonical_player.py (Layer 2).

Measure types: Base (counting totals), Advanced (rate/efficiency incl. PIE),
Usage (USG% and usage breakdown). Pulled in full -- pull wide, derive narrow.

Run locally:
    python pull_wnba_player_season.py --seasons 1997-2025
    python pull_wnba_player_season.py --seasons 2024 --force
"""
import argparse
from nba_api.stats.endpoints import leaguedashplayerstats
from raw_cache import cached_fetch

WNBA = '10'

MEASURE_TYPES = {
    'base': 'Base',
    'advanced': 'Advanced',
    'usage': 'Usage',
}


def _fetch_measure(season, measure_type):
    def _do():
        ep = leaguedashplayerstats.LeagueDashPlayerStats(
            league_id_nullable=WNBA,
            season=str(season),
            season_type_all_star='Regular Season',
            per_mode_detailed='Totals',
            measure_type_detailed_defense=measure_type,
        )
        return ep.get_data_frames()[0].to_dict('records')
    return _do


def pull_season(season, force=False):
    for key, mt in MEASURE_TYPES.items():
        cached_fetch('WNBA', f'player_{key}', str(season),
                     _fetch_measure(season, mt), force=force)


def parse_seasons(arg):
    if '-' in arg and ',' not in arg:
        start, end = arg.split('-')
        return [str(y) for y in range(int(start), int(end) + 1)]
    return [s.strip() for s in arg.split(',')]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', required=True, help='e.g. 1997-2025 or 2024')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    for season in parse_seasons(args.seasons):
        print(f"[{season}] pulling player measure types...")
        pull_season(season, force=args.force)
    print("\nDone. Raw cache populated. Next: inspect columns, then build_canonical_player.py")
