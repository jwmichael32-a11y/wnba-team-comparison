"""
canonical_schema.py

The single source of truth for the API-native canonical column names, in
flat snake_case (per the naming decision). This is the contract every
transform writes to and every downstream consumer (scoring, UI) reads from.

DESIGN PRINCIPLE: PULL WIDE, DERIVE NARROW.
The raw cache stores the FULL measure-type responses (all of Base +
Advanced + Four Factors + Opponent, every column the endpoint returns).
The canonical files surface a curated set, but because raw is complete,
any currently-deferred metric (SRS via SOS solver, Win Shares, PER,
BPM/VORP) can be re-derived LATER without re-pulling. Never filter the
raw pull down to only what today's app uses.

METRIC REPLACEMENT DECISIONS (WNBA v1, per project scoping):
  - SRS (BR-computed, needs SOS solver)  -> REPLACED by net_rating (native,
    from Advanced). Raw retains everything needed to build true SRS later.
  - Win Shares (BR-computed)             -> REPLACED by pie (native, from
    player Advanced) as the roster-strength signal. Raw retains box-score
    inputs to build true WS later.
  - PER / BPM / VORP                     -> DROPPED from canonical for now
    (display-only in old app, not composite-critical). Box-score inputs
    stay in raw.

These are v1 substitutions to ship a working, ToS-clean WNBA app - NOT a
statement that the BR metrics are gone forever. The whole point of the
wide-raw / narrow-canonical split is that the door stays open.
"""

# Canonical TEAM-season columns. Maps canonical_name -> human note on source.
TEAM_SEASON_CANONICAL = {
    # identity
    'league': 'literal WNBA/NBA',
    'season': 'bare year for WNBA, hyphenated for NBA',
    'team_id': 'nba_api persistent franchise id (join key)',
    'team_code': 'period-correct abbreviation, derived per-season',
    'team_name': 'period-correct full name from the pull',
    # record
    'g': 'Base', 'w': 'Base', 'l': 'Base', 'w_pct': 'Base',
    # efficiency / pace (Advanced)  -- net_rating REPLACES srs as composite input
    'off_rating': 'Advanced', 'def_rating': 'Advanced', 'net_rating': 'Advanced',
    'pace': 'Advanced', 'ts_pct': 'Advanced', 'efg_pct': 'Advanced',
    # four factors (team + opponent)
    'tov_pct': 'Four Factors', 'orb_pct': 'Four Factors',
    'drb_pct': 'Four Factors', 'ft_rate': 'Four Factors',
    'opp_efg_pct': 'Opponent', 'opp_tov_pct': 'Opponent',
    'opp_orb_pct': 'Opponent', 'opp_drb_pct': 'Opponent', 'opp_ft_rate': 'Opponent',
    # totals (Base) -- full counting suite retained
    'fgm': 'Base', 'fga': 'Base', 'fg_pct': 'Base',
    'fg3m': 'Base', 'fg3a': 'Base', 'fg3_pct': 'Base',
    'ftm': 'Base', 'fta': 'Base', 'ft_pct': 'Base',
    'oreb': 'Base', 'dreb': 'Base', 'reb': 'Base',
    'ast': 'Base', 'stl': 'Base', 'blk': 'Base', 'tov': 'Base',
    'pf': 'Base', 'pts': 'Base', 'min': 'Base',
}

# Canonical PLAYER-season columns.
PLAYER_SEASON_CANONICAL = {
    'league': 'literal', 'season': 'bare year (WNBA)',
    'player_id': 'nba_api persistent player id (join key)',
    'player_name': 'from pull',
    'team_id': 'join key', 'team_code': 'period-correct abbrev',
    'age': 'Base', 'gp': 'Base', 'gs': 'Base', 'min': 'Base',
    # counting totals
    'fgm': 'Base', 'fga': 'Base', 'fg_pct': 'Base',
    'fg3m': 'Base', 'fg3a': 'Base', 'fg3_pct': 'Base',
    'ftm': 'Base', 'fta': 'Base', 'ft_pct': 'Base',
    'oreb': 'Base', 'dreb': 'Base', 'reb': 'Base',
    'ast': 'Base', 'stl': 'Base', 'blk': 'Base', 'tov': 'Base',
    'pf': 'Base', 'pts': 'Base',
    # advanced / rate  -- pie REPLACES win_shares as roster-strength signal
    'ts_pct': 'Advanced', 'efg_pct': 'Advanced', 'usg_pct': 'Usage',
    'oreb_pct': 'Advanced', 'dreb_pct': 'Advanced', 'reb_pct': 'Advanced',
    'ast_pct': 'Advanced', 'stl_pct': 'Advanced', 'blk_pct': 'Advanced',
    'tov_pct': 'Advanced', 'pie': 'Advanced',
    # provenance / confidence -- first-class column, per project pattern
    'source': 'which pull/derivation produced this row',
}

# Provenance is a first-class column on canonical tables (per the
# established pattern: attribution flags, multi_year_tenure_flag, etc.).
# Every canonical row carries where it came from, so a future data-quality
# surprise is visible in the data, not discovered at runtime.
