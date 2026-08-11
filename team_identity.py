"""
team_identity.py

Canonical team identity for the API-native pipeline.

KEY ARCHITECTURAL NOTE (this changes something from the BR-era design):
nba_api's TEAM_ID is assigned PER FRANCHISE, not per era -- it does NOT
change across a relocation or rename. The Seattle SuperSonics and Oklahoma
City Thunder are the same TEAM_ID, same for every other relocated
franchise. scoring.py's FRANCHISE_LINEAGES dict (bridging codes like
{'SEA', 'OKC'} or {'CHH', 'NOH', 'NOK', 'NOP'}) existed specifically
because Basketball-Reference codes change per era for the same franchise.

If the canonical join key becomes team_id instead of a BR-style code,
FRANCHISE_LINEAGES-style bridging may not be needed at all for
season-to-season continuity -- team_id already IS the persistent key.
Flagging this rather than porting FRANCHISE_LINEAGES over unchanged,
since carrying it forward as-is could be solving a problem that no
longer exists in the new schema. Worth confirming once real data's
in hand, not assumed from the static table alone.

WHAT'S CONFIRMED vs. NOT YET TESTED:
- static.teams gives the current 30 franchises: team_id, current
  abbreviation, current full name. This part is confirmed (it's a
  static, hardcoded table in nba_api, not a live API call).
- Whether an endpoint returns the PERIOD-CORRECT abbreviation for a
  historical season (e.g. 'SEA' for a 1990s season query, not 'OKC')
  is NOT yet tested from here -- this needs a real local run, same as
  the coach-history and playoff-series probes earlier.
"""
from nba_api.stats.static import teams

def get_current_team_table():
    """Returns {team_id: {'abbreviation': ..., 'full_name': ..., 'city': ...}}
    for the 30 current franchises. Does NOT give period-correct historical
    abbreviations -- see module docstring."""
    return {t['id']: {
        'abbreviation': t['abbreviation'],
        'full_name': t['full_name'],
        'city': t['city'],
    } for t in teams.get_teams()}


if __name__ == '__main__':
    # Quick local sanity check -- run this first before building anything
    # on top of it, same "verify before trusting" pattern as the earlier
    # coach/playoff probes.
    table = get_current_team_table()
    print(f"{len(table)} current franchises loaded")
    for tid, info in list(table.items())[:5]:
        print(tid, info)
