"""
Derives the real team_id -> {names, codes, seasons} relationships straight
from the canonical files on disk, so the canonical code map is built from
verified data rather than guessed ids. Run and paste the output.

    python derive_team_id_map.py
"""
import pandas as pd
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'data', 'processed', 'WNBA')

t = pd.read_csv(os.path.join(OUT, 'team_season_stats.csv'))
p = pd.read_csv(os.path.join(OUT, 'player_season_stats.csv'))

print("=== team_id -> (names seen, codes seen, season range) ===")
rows = []
for tid in sorted(set(t['team_id']) | set(p['team_id'])):
    t_sub = t[t['team_id'] == tid]
    p_sub = p[p['team_id'] == tid]
    names = sorted(set(t_sub['team_name'].dropna()))
    t_codes = sorted(set(t_sub['team_code'].dropna()))
    p_codes = sorted(set(p_sub['team_code'].dropna()))
    all_seasons = sorted(set(t_sub['season']) | set(p_sub['season']))
    span = f"{all_seasons[0]}-{all_seasons[-1]}" if all_seasons else "?"
    rows.append((tid, span, t_codes, p_codes, names))

for tid, span, tcodes, pcodes, names in rows:
    flag = '  <-- CODE MISMATCH' if set(tcodes) != set(pcodes) else ''
    print(f"{tid} [{span}] team={tcodes} player={pcodes}{flag}")
    print(f"           names: {names}")
