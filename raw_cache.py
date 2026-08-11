"""
raw_cache.py

Permanent, untouched cache for nba_api pulls. Every call goes through here
first -- if a cached raw response exists for this (league, endpoint, key),
it's read from disk instead of hitting the API again. This makes every
transform step re-runnable without re-pulling, and every re-pull an
explicit, deliberate act (delete or --force) rather than an accident of
re-running a script.

Layout:
    data/raw/{league}/{endpoint}/{key}.json

`key` is usually a season string ("2000"), but can be anything
JSON-serializable-as-filename (a game_id, a team_id+season pair joined
with an underscore, etc.) -- callers decide what's meaningful per endpoint.
"""
import json
import os
import time

# Anchor to THIS file's directory as the project root. All pipeline scripts
# and this module live flat in the project root, so raw/ sits directly
# alongside them at ./data/raw/. (Do NOT count directory levels with nested
# dirname() calls -- that silently breaks the moment the file layout changes.)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')


def _cache_path(league: str, endpoint: str, key: str) -> str:
    d = os.path.join(RAW_DIR, league, endpoint)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.json")


def cached_fetch(league: str, endpoint: str, key: str, fetch_fn, force: bool = False,
                  polite_delay: float = 1.0):
    """
    fetch_fn: zero-arg callable that hits the API and returns a
    JSON-serializable object (e.g. a DataFrame's .to_dict('records'), or
    the raw response dict). Only called on a cache miss (or force=True).

    Returns the cached-or-freshly-fetched data as a plain Python object
    (list/dict), NOT a DataFrame -- callers wrap it themselves. Keeping
    this layer dumb/generic means it works for any endpoint shape.
    """
    path = _cache_path(league, endpoint, key)
    if os.path.exists(path) and not force:
        with open(path, 'r') as f:
            return json.load(f)

    data = fetch_fn()
    print(f"    [raw_cache] fetched {len(data) if hasattr(data, '__len__') else '?'} records for {league}/{endpoint}/{key}", flush=True)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"    [raw_cache] WROTE {path}", flush=True)
    except Exception as e:
        print(f"    [raw_cache] WRITE FAILED for {path}: {type(e).__name__}: {e}", flush=True)
        raise
    time.sleep(polite_delay)  # only sleep on an actual API hit, not a cache read
    return data


def cache_status(league: str, endpoint: str) -> list:
    """Lists which keys are already cached for a league/endpoint, so you
    can see coverage at a glance before deciding what still needs pulling."""
    d = os.path.join(RAW_DIR, league, endpoint)
    if not os.path.exists(d):
        return []
    return sorted(f.replace('.json', '') for f in os.listdir(d) if f.endswith('.json'))
