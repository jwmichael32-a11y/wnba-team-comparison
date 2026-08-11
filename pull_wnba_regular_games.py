"""
pull_wnba_regular_games.py  --  LAYER 1 (raw pull only)

Pulls WNBA REGULAR-SEASON game-level data (LeagueGameLog, one row per team
per game) and caches it under data/raw/WNBA/regular_games/. Pull-and-cache
only.

WHY THIS EXISTS (the SOS / SRS unlock):
The v1 team file uses net_rating as a stand-in for SRS. A *true* SRS needs
strength-of-schedule, and every non-degenerate way to compute SOS -- the
real iterative SRS solver, or any rating-network method (Massey, Colley,
Elo, Bradley-Terry) -- requires game-by-game results (who played whom, and
the margin). Zero-new-data approximations (balanced-schedule / league-avg)
are mathematically degenerate: they assign ~equal SOS to everyone and add
no information. A conference-weighted proxy adds *some* signal but is
approximate and era-limited. Game logs are the input that unlocks the
proper methods, so we pull them once and cache.

This endpoint returns ONE ROW PER TEAM PER GAME: each game appears twice
(once per team), each row carrying that team's PTS + WL but NOT the
opponent's points. Margin is recovered downstream by pairing the two rows
that share a GAME_ID (Layer 2), same as the playoff reconstruction did.

Run locally:
    python pull_wnba_regular_games.py --seasons 1997-2025
    python pull_wnba_regular_games.py --seasons 2024 --force
"""
import argparse
from nba_api.stats.endpoints import leaguegamelog
from raw_cache import cached_fetch

WNBA = '10'


def _fetch(season):
    def _do():
        ep = leaguegamelog.LeagueGameLog(
            league_id=WNBA, season=str(season),
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='T',   # team-level rows
        )
        return ep.get_data_frames()[0].to_dict('records')
    return _do


def pull_season(season, force=False):
    cached_fetch('WNBA', 'regular_games', str(season), _fetch(season), force=force)


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
        print(f"[{s}] pulling regular-season games...")
        pull_season(s, force=args.force)
    print("\nDone. Raw cache populated. Inspect columns, then build the SOS/SRS layer.")
