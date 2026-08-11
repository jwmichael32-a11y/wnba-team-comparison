"""
wnba_team_codes.py

Period-correct team_code fill for the WNBA canonical table, derived from the
team_name each pull returns rather than from guessed team_ids.

WHY NAME-DERIVED, NOT id-MAPPED:
nba_api's static WNBA table has only the 13 CURRENT franchises. The league
has had ~18 franchises since 1997; defunct/relocated ones (Cleveland
Rockers, Houston Comets, Charlotte Sting, Miami Sol, Portland Fire,
Sacramento Monarchs, Utah Starzz, Orlando Miracle, Detroit Shock, etc.)
aren't in the static table at all. But we confirmed the PULL returns the
period-correct team_name for every season -- so the reliable key is the
name, not a hand-guessed id map.

This NAME_TO_CODE table is the one thing that must be enumerated by hand,
but it's bounded (~18 franchises, all known) and verifiable against the
printed team_names. Any team_name that isn't in the table gets a null code
AND is printed loudly, so a gap surfaces immediately instead of silently
producing a blank or a wrong code.
"""

# Full franchise name (as returned by the pull) -> abbreviation.
# Covers current + defunct/historical WNBA franchises. Abbreviations follow
# the codes used in the project's existing playoff_results_manual.csv (HOU,
# NYL, CHA, PHO, LAS, etc.) for consistency with data already in the repo.
NAME_TO_CODE = {
    # current franchises
    'Atlanta Dream': 'ATL',
    'Chicago Sky': 'CHI',
    'Connecticut Sun': 'CON',
    'Dallas Wings': 'DAL',
    'Golden State Valkyries': 'GSV',
    'Indiana Fever': 'IND',
    'Las Vegas Aces': 'LVA',
    'Los Angeles Sparks': 'LAS',
    'Minnesota Lynx': 'MIN',
    'New York Liberty': 'NYL',
    'Phoenix Mercury': 'PHO',
    'Seattle Storm': 'SEA',
    'Washington Mystics': 'WAS',
    # relocations (period-correct names the pull returns for older seasons)
    'Detroit Shock': 'DET',
    'Tulsa Shock': 'TUL',
    'San Antonio Stars': 'SAS',
    'San Antonio Silver Stars': 'SAS',
    'Utah Starzz': 'UTA',
    'Orlando Miracle': 'ORL',
    # defunct franchises
    'Cleveland Rockers': 'CLE',
    'Houston Comets': 'HOU',
    'Charlotte Sting': 'CHA',
    'Miami Sol': 'MIA',
    'Portland Fire': 'POR',
    'Sacramento Monarchs': 'SAC',
}


def fill_team_codes(df):
    """Adds/overwrites df['team_code'] from team_name. Prints any unmapped
    team_names loudly so historical gaps surface immediately."""
    df = df.copy()
    df['team_code'] = df['team_name'].map(NAME_TO_CODE)

    unmapped = df[df['team_code'].isna()][['team_id', 'team_name', 'season']].drop_duplicates()
    if len(unmapped):
        print("\n[wnba_team_codes] UNMAPPED team_names (add to NAME_TO_CODE):")
        print(unmapped.to_string(index=False))
        print("  ^ add each exact team_name to NAME_TO_CODE in wnba_team_codes.py\n")
    return df
