"""
Probe: do CommonPlayoffSeries + LeagueGameLog return usable WNBA playoff
game data across the full 2001-2025 range, with the MATCHUP field we need
for opponent-pairing? Verify coverage BEFORE building the reconstruction
pipeline on top of these endpoints.

Tests a spread of seasons across the WNBA's different bracket-format eras:
  2001 (conference era), 2016 (reseeded single-elim early rounds era),
  2023 (straight 8-team seeded bracket era).

Run locally:
    python probe_wnba_playoff_games.py
"""
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

WNBA = '10'

def probe(season):
    print(f"\n=== {season} ===")
    try:
        log = leaguegamelog.LeagueGameLog(
            league_id=WNBA, season=str(season),
            season_type_all_star='Playoffs',
        )
        df = log.get_data_frames()[0]
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    if df.empty:
        print("  no playoff games returned")
        return
    cols = ['TEAM_ABBREVIATION', 'GAME_DATE', 'MATCHUP', 'WL']
    have = [c for c in cols if c in df.columns]
    print(f"  {len(df)} team-game rows, {df['GAME_ID'].nunique()} unique games")
    print(f"  has MATCHUP: {'MATCHUP' in df.columns}, has WL: {'WL' in df.columns}")
    print(df[have].head(6).to_string(index=False))

if __name__ == '__main__':
    for s in ['2001', '2016', '2023']:
        probe(s)
        time.sleep(1)
