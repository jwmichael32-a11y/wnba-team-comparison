"""Dump a season's playoff series structure from the raw cache, to see the
actual bracket shape (esp. single-elim rounds in 2016-2020).
    python dump_bracket.py 2016
"""
import sys, json, re, os
import pandas as pd

season = sys.argv[1] if len(sys.argv) > 1 else '2016'
path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'data', 'raw', 'WNBA', 'playoff_games', f'{season}.json')
g = pd.DataFrame(json.load(open(path)))

def opp(m, t):
    p = re.split(r'\s+vs\.\s+|\s+@\s+', m)
    p = [x for x in p if x != t]
    return p[0] if p else None

g['OPP'] = g.apply(lambda r: opp(r['MATCHUP'], r['TEAM_ABBREVIATION']), axis=1)
g['PAIR'] = g.apply(lambda r: tuple(sorted([r['TEAM_ABBREVIATION'], r['OPP']])), axis=1)

print(f"=== {season} series ===")
series = []
for pair, sub in g.groupby('PAIR'):
    n_games = sub['GAME_DATE'].nunique()
    wins = sub[sub['WL'] == 'W']['TEAM_ABBREVIATION'].value_counts()
    w = wins.idxmax()
    series.append((sub['GAME_DATE'].min(), pair, n_games, w))
for start, pair, n_games, w in sorted(series):
    print(f"  {start}  {pair[0]:4} vs {pair[1]:4}  games={n_games}  winner={w}")
