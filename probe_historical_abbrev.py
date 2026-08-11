"""
Probe: does a historical-season team pull return the PERIOD-CORRECT team
abbreviation, or always the current one?

This is THE question that decides whether team_code in the canonical schema
can be trusted as season-accurate, or whether we need a separate historical
team-identity mapping layer. Static.teams can't answer it -- only a real
seasoned endpoint can.

Test cases chosen because their answer is unambiguous if you know the history:
- 1995-96 SEA/OKC: SuperSonics were in Seattle then. Period-correct = 'SEA',
  current = 'OKC'.
- 1994-95 VAN/MEM: Grizzlies didn't exist yet in Vancouver until 1995-96,
  so 1995-96 is the first testable year. Period-correct = 'VAN', current = 'MEM'.
- 2007-08 SEA: last Seattle season before the OKC move. Period-correct = 'SEA'.

Uses LeagueDashTeamStats (season-parameterized team table) as the test
endpoint since that's the closest analog to what team_season_stats.csv
is built from.

Run locally:
    python probe_historical_abbrev.py
"""
from nba_api.stats.endpoints import leaguedashteamstats

def probe(season):
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        league_id_nullable='00',
    )
    df = stats.get_data_frames()[0]
    # Look for the franchise that relocated -- print team_id + name for any
    # row whose TEAM_NAME mentions a city that moved.
    cols = [c for c in df.columns if c in ('TEAM_ID', 'TEAM_NAME')]
    print(f"\n=== {season} ===")
    watch = df[df['TEAM_NAME'].str.contains('Seattle|Oklahoma|Vancouver|Memphis|New Jersey|Brooklyn', case=False, na=False)]
    if watch.empty:
        print("  (no watched franchise found this season -- print all names:)")
        print("  " + ", ".join(df['TEAM_NAME'].tolist()))
    else:
        print(watch[cols].to_string(index=False))

if __name__ == '__main__':
    for s in ['1995-96', '2007-08', '2008-09']:
        try:
            probe(s)
        except Exception as e:
            print(f"[{s}] ERROR: {e}")
