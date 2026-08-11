"""
build_canonical_playoffs.py  --  LAYER 2 (raw cache -> canonical CSV)

Reconstructs playoff_round_reached from cached game-level data (2001-2025),
folds in the manual 1997-2000 file, and writes canonical playoff_results.csv
with columns: season, team_code, playoff_round_reached, source.

METHOD (same opponent-pairing approach validated for 1997-2000):
  1. From each season's playoff game log, pair opponents via MATCHUP into
     SERIES (ignore the unreliable SERIES_ID; use who-played-whom).
  2. Order series chronologically; the LAST series is the Finals, its winner
     the Champion.
  3. DEPTH-FROM-FINAL ordinal, per that season's actual bracket:
       Champion              -> 'Champion'          (ordinal 5)
       lost the Finals       -> 'Lost Finals'       (ordinal 4)
       lost round before it  -> 'Lost Conf Finals'  (ordinal 3)
       one earlier           -> 'Lost R2'           (ordinal 2)
       two+ earlier          -> 'Lost R1'           (ordinal 1)
     Labels reuse scoring.py's PLAYOFF_ORDINAL keys, but assignment is by
     DEPTH relative to the title in that season's real bracket -- NOT by a
     fixed per-era round name. A season with only 3 rounds simply doesn't
     produce a 'Lost R2'; a season with 4 does. This keeps "how deep did
     they go relative to the championship" consistent across the WNBA's
     changing formats (conference era, single-elim-early era, seeded era).

A team appears in canonical ONLY if it made the playoffs (played >=1 series).
Missing = missed playoffs, which scoring.py reads as ordinal 0. This matches
the loader's existing "absence means missed" contract.
"""
import argparse
import json
import os
import re
import pandas as pd
from canonical_team_codes import normalize_team_codes

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WNBA', 'playoff_games')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'WNBA')
MANUAL_PATH = os.path.join(OUT_DIR, 'playoff_results_manual.csv')

# Depth-honest labels. Depth 0 = lost the final series; deeper = further from
# the title. "Champion" and "Lost Finals" keep their recognizable names; below
# that, labels state distance-from-title explicitly rather than using literal
# round numbers (which would misstate what round a team lost in, since the
# WNBA's bracket depth varies by era). Parallel to scoring.py's PLAYOFF_ORDINAL.
DEPTH_LABELS = [
    'Lost Finals',              # depth 0
    'Lost 1 Round From Title',  # depth 1
    'Lost 2 Rounds From Title', # depth 2
    'Lost 3 Rounds From Title', # depth 3+ (capped)
]


def _opponent(matchup, team):
    parts = re.split(r'\s+vs\.\s+|\s+@\s+', matchup)
    parts = [p for p in parts if p != team]
    return parts[0] if parts else None


def reconstruct_season(games, season):
    """Returns list of {season, team_code, playoff_round_reached} for one season.

    APPROACH: group series into chronological ROUNDS, then a team's depth =
    how many rounds happened AFTER the one it was eliminated in.

    Why rounds, not series-win-count: the 2016-2020 format mixed
    single-elimination rounds (1-game "series") with best-of-5 rounds, and
    gave byes to top seeds. That makes raw series-win COUNT unreliable -- a
    champion with a bye won fewer series than the bracket has rounds, and a
    1-game round win counts the same as a 5-game series win. Counting ROUNDS
    (waves of series that start together) is robust to both, because a round
    is a round regardless of how many games decide it or who had a bye.

    Rounds are detected as chronological waves: series are clustered by
    start-date proximity (a new round begins when there's a gap after the
    previous wave's series). The last round is the Finals; its winner is the
    Champion. Each team's label is by depth = (n_rounds-1) - deepest_round.
    """
    games = games.copy()
    games['OPPONENT'] = games.apply(
        lambda r: _opponent(r['MATCHUP'], r['TEAM_ABBREVIATION']), axis=1)
    games['PAIR'] = games.apply(
        lambda r: tuple(sorted([r['TEAM_ABBREVIATION'], r['OPPONENT']])), axis=1)

    # one entry per series: participants, winner, start date
    series = []
    for pair, sub in games.groupby('PAIR'):
        wins = sub[sub['WL'] == 'W']['TEAM_ABBREVIATION'].value_counts()
        winner = wins.idxmax()
        series.append({'pair': pair, 'winner': winner,
                       'start': sub['GAME_DATE'].min()})
    if not series:
        return []
    series.sort(key=lambda s: s['start'])

    # Cluster series into rounds by BRACKET PROGRESSION, not calendar dates
    # (date gaps proved unreliable: 2019 scheduled R2 and semis only 2 days
    # apart, collapsing them). A round is a maximal set of series in which no
    # team plays that already won a series in that same set. Walking series in
    # chronological order, a new round starts as soon as we hit a series whose
    # participant already won earlier in the current round -- that participant
    # advanced, so this must be the next round.
    def build_rounds(series_list):
        rounds = []
        current = []
        winners_so_far = set()
        for s in series_list:
            a, b = s['pair']
            if a in winners_so_far or b in winners_so_far:
                rounds.append(current)
                current = []
                winners_so_far = set()
            current.append(s)
            winners_so_far.add(s['winner'])
        if current:
            rounds.append(current)
        return rounds

    rounds = build_rounds(series)

    n_rounds = len(rounds)
    champion = rounds[-1][0]['winner']

    # deepest round index each team reached (appeared in)
    team_deepest = {}
    for ridx, rnd in enumerate(rounds):
        for s in rnd:
            for t in s['pair']:
                team_deepest[t] = max(team_deepest.get(t, ridx), ridx)

    results = []
    for team, deepest in team_deepest.items():
        if team == champion:
            label = 'Champion'
        else:
            depth = (n_rounds - 1) - deepest   # 0 = lost final round
            label = DEPTH_LABELS[min(depth, len(DEPTH_LABELS) - 1)]
        results.append({
            'season': str(season),
            'team_code': team,
            'playoff_round_reached': label,
            'source': 'nba_api:reconstructed',
        })
    return results


def cached_seasons():
    if not os.path.exists(RAW_DIR):
        return []
    return sorted(f.replace('.json', '') for f in os.listdir(RAW_DIR) if f.endswith('.json'))


def load_games(season):
    with open(os.path.join(RAW_DIR, f'{season}.json')) as f:
        return pd.DataFrame(json.load(f))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', help='optional subset; default = all cached')
    args = ap.parse_args()

    seasons = ([s.strip() for s in args.seasons.split(',')] if args.seasons
               else cached_seasons())

    all_rows = []
    for s in seasons:
        games = load_games(s)
        if games.empty:
            print(f"[{s}] no games cached")
            continue
        rows = reconstruct_season(games, s)
        all_rows.extend(rows)
        champ = next((r['team_code'] for r in rows if r['playoff_round_reached'] == 'Champion'), '?')
        print(f"[{s}] {len(rows)} playoff teams, champion={champ}")

    recon = pd.DataFrame(all_rows)
    recon = normalize_team_codes(recon)  # PHX->PHO, SAN->SAS to match other files

    # fold in the manual 1997-2000 file
    if os.path.exists(MANUAL_PATH):
        manual = pd.read_csv(MANUAL_PATH)
        manual['season'] = manual['Season'].astype(str)
        manual = manual.rename(columns={'team_code': 'team_code',
                                        'playoff_round_reached': 'playoff_round_reached'})
        manual['source'] = 'manual:1997-2000'
        manual = manual[['season', 'team_code', 'playoff_round_reached', 'source']]
        combined = pd.concat([manual, recon], ignore_index=True)
    else:
        print(f"WARNING: manual file not found at {MANUAL_PATH} -- 1997-2000 will be absent")
        combined = recon

    # overlap guard: no season/team_code should come from both sources
    dupes = combined.duplicated(subset=['season', 'team_code'], keep=False)
    if dupes.any():
        print("\nERROR: overlap between manual and reconstructed:")
        print(combined[dupes].to_string(index=False))
        raise SystemExit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'playoff_results.csv')
    combined = combined.sort_values(['season', 'playoff_round_reached']).reset_index(drop=True)
    combined.to_csv(out_path, index=False)

    print(f"\nWrote {len(combined)} playoff team-seasons to {out_path}")
    print(f"Seasons: {sorted(combined['season'].unique())}")
    print("\nLabel distribution:")
    print(combined['playoff_round_reached'].value_counts().to_string())
