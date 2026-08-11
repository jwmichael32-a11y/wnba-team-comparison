"""
build_canonical_player.py  --  LAYER 2 (raw cache -> canonical CSV)

Player-season analog of build_canonical_team.py. Reads player raw cache,
shapes canonical player_season_stats.csv (flat snake_case). Zero API calls.

Output: data/processed/WNBA/player_season_stats.csv

KEY DECISIONS (mapped against the real WNBA response columns):
  - team_code comes straight from TEAM_ABBREVIATION in the pull (players
    carry it natively; no name-map needed, unlike the team file).
  - team_count (from TEAM_COUNT) is the traded-player signal -- the API's
    clean replacement for BR's old 'TOT' combined rows. Carried into
    canonical so the '% of team wins' renorm logic can filter TEAM_COUNT>1
    exactly as the old multi-team exclusion did.
  - pie carried as the player roster-strength signal (REPLACES win_shares).
  - Full box-score totals carried for future WS reconstruction (pull wide,
    derive narrow).
  - player_id is the canonical join key (stable; names collide/change).
"""
import argparse
import json
import os
import pandas as pd
from canonical_team_codes import normalize_team_codes

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WNBA')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'WNBA')

MEASURE_KEYS = ['base', 'advanced', 'usage']


def _load_cached(season, measure_key):
    path = os.path.join(RAW_DIR, f'player_{measure_key}', f'{season}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def cached_seasons():
    base_dir = os.path.join(RAW_DIR, 'player_base')
    if not os.path.exists(base_dir):
        return []
    seasons = sorted(f.replace('.json', '') for f in os.listdir(base_dir) if f.endswith('.json'))
    complete = []
    for s in seasons:
        if all(_load_cached(s, k) is not None for k in MEASURE_KEYS):
            complete.append(s)
        else:
            missing = [k for k in MEASURE_KEYS if _load_cached(s, k) is None]
            print(f"[{s}] SKIPPED -- missing cached measure(s): {missing}")
    return complete


def build_canonical(season):
    base = _load_cached(season, 'base')
    adv = _load_cached(season, 'advanced')
    usg = _load_cached(season, 'usage')

    out = pd.DataFrame({
        'league': 'WNBA',
        'season': str(season),
        'player_id': base['PLAYER_ID'],
        'player_name': base['PLAYER_NAME'],
        'team_id': base['TEAM_ID'],
        'team_code': base['TEAM_ABBREVIATION'],   # native, no name-map needed
        'team_count': base['TEAM_COUNT'],          # traded-player signal (>1 = multi-team)
        'age': base['AGE'], 'gp': base['GP'], 'gs': None,  # GS not in league-dash; left null
        'min': base['MIN'],
        'fgm': base['FGM'], 'fga': base['FGA'], 'fg_pct': base['FG_PCT'],
        'fg3m': base['FG3M'], 'fg3a': base['FG3A'], 'fg3_pct': base['FG3_PCT'],
        'ftm': base['FTM'], 'fta': base['FTA'], 'ft_pct': base['FT_PCT'],
        'oreb': base['OREB'], 'dreb': base['DREB'], 'reb': base['REB'],
        'ast': base['AST'], 'stl': base['STL'], 'blk': base['BLK'],
        'tov': base['TOV'], 'pf': base['PF'], 'pts': base['PTS'],
        'plus_minus': base['PLUS_MINUS'],
    }).set_index('player_id')

    adv_idx = adv.set_index('PLAYER_ID')
    out['off_rating'] = adv_idx['OFF_RATING']
    out['def_rating'] = adv_idx['DEF_RATING']
    out['net_rating'] = adv_idx['NET_RATING']
    out['ts_pct'] = adv_idx['TS_PCT']
    out['efg_pct'] = adv_idx['EFG_PCT']
    out['usg_pct'] = adv_idx['USG_PCT']
    out['ast_pct'] = adv_idx['AST_PCT']
    out['oreb_pct'] = adv_idx['OREB_PCT']
    out['dreb_pct'] = adv_idx['DREB_PCT']
    out['reb_pct'] = adv_idx['REB_PCT']
    out['pie'] = adv_idx['PIE']                # roster-strength signal, REPLACES win_shares
    out['poss'] = adv_idx['POSS']

    usg_idx = usg.set_index('PLAYER_ID')       # share-of-team breakdown
    out['pct_pts'] = usg_idx['PCT_PTS']
    out['pct_reb'] = usg_idx['PCT_REB']
    out['pct_ast'] = usg_idx['PCT_AST']

    out['source'] = 'nba_api:leaguedashplayerstats'
    return out.reset_index()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', help='optional subset; default = all cached')
    args = ap.parse_args()

    seasons = ([s.strip() for s in args.seasons.split(',')] if args.seasons
               else cached_seasons())
    if not seasons:
        print("No complete cached player seasons found. Run pull_wnba_player_season.py first.")
        raise SystemExit(1)

    frames = [build_canonical(s) for s in seasons]
    result = pd.concat(frames, ignore_index=True)
    # standardize spelling inconsistencies (PHX->PHO, SAN->SAS) to match team file
    result = normalize_team_codes(result)

    id_cols = ['league', 'season', 'player_id', 'player_name', 'team_id', 'team_code', 'team_count']
    result = result[id_cols + [c for c in result.columns if c not in id_cols]]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'player_season_stats.csv')
    result.to_csv(out_path, index=False)

    print(f"\nWrote {len(result)} player-seasons to {out_path}")
    print(f"Seasons: {sorted(result['season'].unique())}")
    print(f"Columns ({len(result.columns)}): {list(result.columns)}")
    multi = result[result['team_count'] > 1]
    print(f"\nMulti-team (traded) player-seasons: {len(multi)} (team_count > 1)")
    print("\nSanity peek (top scorers 2025 if present):")
    peek = result[result['season'] == max(result['season'])].nlargest(8, 'pts')
    print(peek[['season', 'player_name', 'team_code', 'gp', 'pts', 'pie', 'usg_pct']].to_string(index=False))
