"""
build_canonical_team.py  --  LAYER 2 (raw cache -> canonical CSV)

Reads the raw measure-type responses cached by pull_wnba_team_season.py and
shapes them into the canonical team_season_stats.csv (flat snake_case).
Makes ZERO API calls -- it only reads data/raw/WNBA/. Re-run freely after
any schema/mapping change; it never re-pulls.

Output: data/processed/WNBA/team_season_stats.csv

PULL WIDE, DERIVE NARROW: the raw cache holds the FULL responses; this step
projects down to the curated canonical columns. Metric-replacement decisions
(net_rating replaces srs; pie carried as roster signal; opponent counting
totals carried for future WS reconstruction) are applied here, and the raw
cache retains everything needed to revisit them later.
"""
import argparse
import json
import os
import pandas as pd
from wnba_team_codes import fill_team_codes
from canonical_team_codes import normalize_team_codes

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WNBA')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'WNBA')

MEASURE_KEYS = ['base', 'advanced', 'four_factors', 'opponent']


def _load_cached(season, measure_key):
    path = os.path.join(RAW_DIR, f'team_{measure_key}', f'{season}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def cached_seasons():
    """Every season that has a full set of cached measure files."""
    base_dir = os.path.join(RAW_DIR, 'team_base')
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
    ff = _load_cached(season, 'four_factors')
    opp = _load_cached(season, 'opponent')

    out = pd.DataFrame({
        'league': 'WNBA',
        'season': str(season),
        'team_id': base['TEAM_ID'],
        'team_name': base['TEAM_NAME'],
        'g': base['GP'], 'w': base['W'], 'l': base['L'], 'w_pct': base['W_PCT'],
        'fgm': base['FGM'], 'fga': base['FGA'], 'fg_pct': base['FG_PCT'],
        'fg3m': base['FG3M'], 'fg3a': base['FG3A'], 'fg3_pct': base['FG3_PCT'],
        'ftm': base['FTM'], 'fta': base['FTA'], 'ft_pct': base['FT_PCT'],
        'oreb': base['OREB'], 'dreb': base['DREB'], 'reb': base['REB'],
        'ast': base['AST'], 'stl': base['STL'], 'blk': base['BLK'],
        'tov': base['TOV'], 'pf': base['PF'], 'pts': base['PTS'], 'min': base['MIN'],
    }).set_index('team_id')

    adv_idx = adv.set_index('TEAM_ID')
    out['off_rating'] = adv_idx['OFF_RATING']
    out['def_rating'] = adv_idx['DEF_RATING']
    out['net_rating'] = adv_idx['NET_RATING']   # composite input, REPLACES srs
    out['pace'] = adv_idx['PACE']
    out['poss'] = adv_idx['POSS']
    out['ts_pct'] = adv_idx['TS_PCT']
    out['efg_pct'] = adv_idx['EFG_PCT']
    out['pie'] = adv_idx['PIE']
    out['orb_pct'] = adv_idx['OREB_PCT']
    out['drb_pct'] = adv_idx['DREB_PCT']   # both reb% live in Advanced, not Four Factors

    ff_idx = ff.set_index('TEAM_ID')
    out['tov_pct'] = ff_idx['TM_TOV_PCT']
    out['ft_rate'] = ff_idx['FTA_RATE']
    out['opp_efg_pct'] = ff_idx['OPP_EFG_PCT']
    out['opp_tov_pct'] = ff_idx['OPP_TOV_PCT']
    out['opp_orb_pct'] = ff_idx['OPP_OREB_PCT']
    out['opp_ft_rate'] = ff_idx['OPP_FTA_RATE']

    opp_idx = opp.set_index('TEAM_ID')   # opponent COUNTING totals (WS-reconstruction inputs)
    out['opp_pts'] = opp_idx['OPP_PTS']
    out['opp_fgm'] = opp_idx['OPP_FGM']
    out['opp_fga'] = opp_idx['OPP_FGA']
    out['opp_ftm'] = opp_idx['OPP_FTM']
    out['opp_fta'] = opp_idx['OPP_FTA']
    out['opp_tov'] = opp_idx['OPP_TOV']
    out['opp_oreb'] = opp_idx['OPP_OREB']
    out['opp_dreb'] = opp_idx['OPP_DREB']

    out['source'] = 'nba_api:leaguedashteamstats'
    return out.reset_index()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', help='optional subset e.g. 2024,2025; default = all cached')
    args = ap.parse_args()

    seasons = ([s.strip() for s in args.seasons.split(',')] if args.seasons
               else cached_seasons())
    if not seasons:
        print("No complete cached seasons found. Run pull_wnba_team_season.py first.")
        raise SystemExit(1)

    frames = [build_canonical(s) for s in seasons]
    result = pd.concat(frames, ignore_index=True)

    # period-correct team_code from team_name (prints any unmapped names)
    result = fill_team_codes(result)
    # standardize spelling inconsistencies (PHX->PHO, SAN->SAS) uniformly
    result = normalize_team_codes(result)

    # order columns: identity first, then the rest
    id_cols = ['league', 'season', 'team_id', 'team_code', 'team_name']
    result = result[id_cols + [c for c in result.columns if c not in id_cols]]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'team_season_stats.csv')
    result.to_csv(out_path, index=False)

    print(f"\nWrote {len(result)} team-seasons to {out_path}")
    print(f"Seasons: {sorted(result['season'].unique())}")
    print(f"Columns ({len(result.columns)}): {list(result.columns)}")
    print("\nSanity peek:")
    print(result[['season', 'team_code', 'team_name', 'w', 'l', 'net_rating']].head(13).to_string(index=False))
