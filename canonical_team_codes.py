"""
canonical_team_codes.py

Fixes team_code SPELLING inconsistencies between the team and player files,
while PRESERVING legitimate period-correct relocation codes.

THE ACTUAL PROBLEM (from derive_team_id_map.py, verified against real data):
Only two franchises disagree between the files, and only on spelling:
  - team_id 1611661317 (Phoenix): team file 'PHO' vs player file 'PHO'/'PHX'
  - team_id 1611661319 (Aces line): team 'SAS' vs player 'SAN' for the
    San Antonio era
Everything else already agrees.

WHAT WE MUST *NOT* DO:
Collapse each team_id to a single code. Relocated franchises legitimately
have MULTIPLE period-correct codes across their history, and the era-aware
app WANTS them distinct:
  - 1611661319: UTA (Starzz) -> SAS (San Antonio) -> LVA (Aces)
  - 1611661321: DET (Detroit) -> TUL (Tulsa) -> DAL (Dallas)
  - 1611661323: ORL (Orlando) -> CON (Connecticut)
Forcing one code per team_id would erase that history.

THE FIX:
A spelling-normalization map of {inconsistent_code: canonical_code}, applied
uniformly to both files. It only touches the specific misspellings; every
correct code (including all relocation codes) passes through untouched.
Canonical spellings follow playoff_results_manual.csv (PHO, SAS).
"""

# Only the spellings that need fixing. Anything not listed passes through.
CODE_SPELLING_FIX = {
    'PHX': 'PHO',   # Phoenix: standardize to PHO (matches team file + playoff file)
    'SAN': 'SAS',   # San Antonio: standardize to SAS (matches team file + playoff file)
}


def normalize_team_codes(df):
    """Applies spelling fixes to df['team_code'] uniformly. Legitimate
    relocation codes (UTA/SAS/LVA, DET/TUL/DAL, ORL/CON) are preserved --
    only the specific misspellings in CODE_SPELLING_FIX are changed."""
    df = df.copy()
    df['team_code'] = df['team_code'].replace(CODE_SPELLING_FIX)
    return df
