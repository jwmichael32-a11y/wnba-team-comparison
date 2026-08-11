"""
Dumps the actual column names from each cached raw measure-type response,
so we can fix inferred names against reality instead of guessing. Zero API
calls -- reads only what's already cached on disk.

    python inspect_raw_cache.py 2024
"""
import sys, json, os

season = sys.argv[1] if len(sys.argv) > 1 else '2024'
group = sys.argv[2] if len(sys.argv) > 2 else 'team'  # 'team' or 'player'
RAW = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'WNBA')

if group == 'player':
    keys = ['player_base', 'player_advanced', 'player_usage']
elif group == 'regular':
    keys = ['regular_games']
elif group == 'playoff':
    keys = ['playoff_games']
else:
    keys = ['team_base', 'team_advanced', 'team_four_factors', 'team_opponent']

for key in keys:
    path = os.path.join(RAW, key, f'{season}.json')
    if not os.path.exists(path):
        print(f"{key}: NOT CACHED ({path})")
        continue
    with open(path) as f:
        data = json.load(f)
    cols = list(data[0].keys()) if data else []
    print(f"\n=== {key} ({len(data)} rows) ===")
    print(cols)
