"""
scoring.py

Implements the team-comparison scoring model (canonical snake_case contract,
nba_api-sourced; Basketball-Reference fully severed).

Three scored categories, chosen for signal INDEPENDENCE rather than coverage:
- Quality  : the Massey `srs` (schedule-adjusted point differential). Falls
             back to `net_rating` for any season lacking a game-log-derived srs
             (they correlate 0.992, and net_rating covers every team-season).
- Playoffs : depth-honest postseason ordinal (see PLAYOFF_ORDINAL).
- Roster   : Sum of the roster's (pie x minutes) -- an extensive personnel
             signal, single-team rows only.

Everything else (shooting, four factors, standings, coaching, exec) is
informational/display-only, NOT part of the composite.

Why this shape (empirically grounded, not stylistic):
  The old model carried SRS (0.30) AND Net Rating (0.20) as separate categories;
  the Massey srs and net_rating correlate 0.992, so scoring both was counting
  one signal twice at half the composite. They are collapsed into one Quality
  axis. Coaching was removed from the composite entirely: any prior-record
  coaching signal is itself correlated with team quality (re-importing it), the
  award component scores a team the year AFTER recognition was earned (a lag
  artifact), and there is no WNBA coach data regardless -- so coaching is now
  qualitative/display-only. Roster (pie x min) is retained at a light weight,
  acknowledged as ~0.85 correlated with Quality; the genuinely independent axis
  is Playoffs (~0.79 with Quality), which is why it carries real weight.

Weights: the one subjective knob -- a values choice about what the score MEANS
  (regular-season dominance vs. postseason achievement). Edit WEIGHTS below to
  retune; the assertion keeps them a proper distribution.

Playoff round -> ordinal scale used for scoring:
  0 = missed playoffs, 1 = lost 3 rounds from title, 2 = lost 2 rounds from
  title, 3 = lost 1 round from title (the round before the Finals), 4 = lost
  Finals, 5 = Champion. Labels state distance-from-title rather than literal
  round numbers, because the WNBA's bracket depth varies by era, so a fixed
  round name (e.g. "R1") would misstate what round a team actually lost in.
"""

import pandas as pd
import numpy as np

# The three composite weights. This is the only values-based knob in the model;
# change these numbers to retune. They must form a distribution (sum to 1).
WEIGHTS = {
    'quality':  0.45,   # Massey srs (schedule-adjusted), net_rating fallback
    'playoffs': 0.40,   # depth-honest postseason ordinal
    'roster':   0.15,   # sum(pie x minutes) across the roster
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"


def normalize_weights(weights):
    """Coerce a user-supplied weight dict into a proper distribution (non-negative,
    sums to 1) over the three scored categories. Missing keys default to 0; an
    all-zero input falls back to the module default. This lets the app expose raw
    slider values (any scale) and hand them straight in."""
    keys = ('quality', 'playoffs', 'roster')
    w = {k: max(float(weights.get(k, 0.0)), 0.0) for k in keys}
    total = sum(w.values())
    if total <= 0:
        return dict(WEIGHTS)
    return {k: v / total for k, v in w.items()}

PLAYOFF_ORDINAL = {
    'Lost 3 Rounds From Title': 1,
    'Lost 2 Rounds From Title': 2,
    'Lost 1 Round From Title': 3,
    'Lost Finals': 4,
    'Champion': 5,
}

COACH_WIN_PSEUDO_GAMES = 41    # ~half a season of "league average" pseudo-games, used for shrinkage

# Earliest season each league's team_season_stats.csv covers, keyed by
# league. IMPORTANT: each league's season strings use that league's OWN
# native format - NBA seasons are hyphenated cross-year labels ('1979-80')
# since an NBA season spans two calendar years; WNBA seasons are bare
# years ('1997') since a WNBA season falls entirely within one calendar
# year (confirmed against the Chunk 1 pipeline's season="1999"-style
# nba_api calls). The string comparisons in get_era() and the cutoff-year
# math in compute_coach_prior_hybrid() only produce correct chronological
# ordering if every boundary/cutoff for a league matches that league's own
# format - don't mix formats within one league's list.
DATA_CUTOFF_SEASON = {
    'NBA': '1979-80',
    'WNBA': '1997',  # WNBA's inaugural season - update if File 1 coverage differs
}

STAT_DEFINITIONS = {
    'SRS': "Massey rating - point differential solved simultaneously over the whole schedule so it's adjusted for strength of schedule (blowouts capped at the 95th percentile). 0 = league average.",
    'Quality': "Composite team-strength axis: the Massey SRS (net_rating fallback where no game logs exist).",
    'Net Rating': "Offensive Rating minus Defensive Rating - points scored vs. allowed per 100 possessions.",
    'ORtg': "Offensive Rating - points scored per 100 possessions.",
    'DRtg': "Defensive Rating - points allowed per 100 possessions (lower is better).",
    'Pace': "Estimated possessions per 48 minutes - how fast a team plays.",
    'eFG%': "Effective Field Goal % - shooting accuracy, weighted to credit 3-pointers extra since they're worth more.",
    'TS%': "True Shooting % - overall scoring efficiency, factoring in 2s, 3s, AND free throws in one number.",
    'TOV%': "Turnover % - turnovers per 100 plays. Lower is better (fewer giveaways).",
    'ORB%': "Offensive Rebound % - share of available offensive rebounds a team actually grabs.",
    'DRB%': "Defensive Rebound % - share of available defensive rebounds a team actually grabs.",
    'FTr': "Free Throw Rate - free throw attempts relative to field goal attempts (how often a team gets to the line).",
    'MOV': "Margin of Victory - average point differential per game.",
    'SOS': "Strength of Schedule - how much tougher/easier than average a team's opponents were.",
    'PIE': "Player Impact Estimate - a player's share of the weighted box-score events (scoring, rebounds, assists, steals, blocks, minus misses/turnovers/fouls) in their games. An all-in-one footprint metric, outcome-agnostic.",
    'WS': "Win Shares (Oliver method) - an estimate of how many wins a player is individually responsible for; a roster's WS sum to about the team's actual win total. Outcome-anchored and efficiency-weighted (contrast PIE).",
    'OWS': "Offensive Win Shares - the offensive half of Win Shares, from points produced vs. a league-average baseline.",
    'DWS': "Defensive Win Shares - the defensive half of Win Shares, from an individual defensive rating and team defensive context.",
    'USG%': "Usage % - share of a team's offensive plays used by a player while on the floor.",
    'Playoffs': "Ordinal scale: 0 = missed playoffs, up to 5 = won the championship.",
    'Roster': "Sum of the roster's (PIE x minutes) for that season - an extensive personnel-strength signal. Single-team rows only.",
    '% of team impact': "Player's (PIE x minutes) as a share of the team's roster total (footprint). Blank for multi-team combined seasons.",
    '% of team wins': "Player's Win Shares as a share of the team's positive-WS pool (win-translation). Blank for multi-team combined seasons. Read against % of team impact: the gap is footprint vs. how much converted to winning.",
    'MP': "Minutes Played.",
}

# ----------------------------------------------------------------------
# ERA CLASSIFICATION
# Eras are computed ONCE, OFFLINE, per league - never live in the app.
# NBA boundaries below were chosen via a three-step process, documented in
# full on the app's Methodology page:
#   1. Candidate boundaries proposed from known NBA rule-change history
#   2. Visual/descriptive validation against league-wide yearly means and
#      standard deviations of pace, ORtg, DRtg, eFG%, 3PAr, TOV%, ORB%,
#      DRB%, FTr (both team and opponent side)
#   3. Objective cross-check via PELT changepoint detection (ruptures
#      library), run SEPARATELY on an offense/pace signal set and a
#      defense signal set, at multiple penalty sensitivities
# The two signal sets did not fully agree - offense-side breaks (1994,
# 2018-19) reflect offensive strategy/rule shifts with no matching
# defensive signal; the defense-side break (2003-04) reflects the 2004
# hand-checking reform and is barely visible on the offense side. Both
# are treated as legitimate boundaries rather than resolved into one
# "cleaner" answer, since they represent genuinely different structural
# forces acting on different sides of the ball.
#
# WNBA has NOT been through this offline PELT process yet - that's a
# separate future analysis, not something the app computes live. Until it
# exists, every WNBA season buckets into a single placeholder era so
# era_rank/era_total/era-adjustment inputs stay well-defined without
# implying a level of rigor that hasn't been done. Swap in real boundaries
# here once that analysis is finished - nothing else in this file needs to
# change, since every consumer looks up ERA_BOUNDARIES[league] rather than
# assuming any particular number of eras.
# ----------------------------------------------------------------------
ERA_BOUNDARIES = {
    'NBA': [
        ('1979-80', '1993-94', 'The Pace & Post Era'),
        ('1994-95', '1996-97', 'The Short Arc Era'),
        ('1997-98', '2003-04', 'The Dead-Ball Era'),
        ('2004-05', '2013-14', 'The Freedom of Movement Era'),
        ('2014-15', '2018-19', 'The Pace & Space Build-Up'),
        ('2019-20', '2025-26', 'The Efficiency Explosion'),
    ],
    'WNBA': [
        ('1997', '2025', 'Full History (era boundaries not yet validated)'),
    ],
}


def get_era(season, league):
    """Maps a season string to its era label, using that league's own
    boundary list. `season` must be in the league's native format (see the
    DATA_CUTOFF_SEASON comment above for the NBA/WNBA format difference)."""
    for start, end, label in ERA_BOUNDARIES[league]:
        if start <= season <= end:
            return label
    return 'Unclassified'


def _cutoff_year(league):
    """Converts DATA_CUTOFF_SEASON[league] to a plain integer year, handling
    both the NBA's hyphenated cross-year format ('1979-80' -> 1980, the
    second/ending year) and WNBA's bare-year format ('1997' -> 1997)."""
    cutoff_season = DATA_CUTOFF_SEASON[league]
    if '-' in cutoff_season:
        return int(cutoff_season.split('-')[0]) + 1
    return int(cutoff_season)


# ----------------------------------------------------------------------
# ERA ADJUSTMENT: quantile-mapping a team's metrics onto an equivalent
# percentile in a different ("base") era's distribution - restating a
# team's performance as if it had occurred in that era.
# ----------------------------------------------------------------------

ERA_ADJUSTABLE_METRICS = {
    'Quality': 'quality_metric',    # Massey srs (net_rating fallback), a real column
    'Roster': 'roster_strength',    # sum(pie x min)
}
# Playoffs is deliberately excluded - a championship means the same thing
# in any era, so its RAW value isn't transformed. It's still re-standardized
# against the base era's own playoff-ordinal distribution when era
# adjustment is on, so it combines consistently with the adjusted metrics.


def _tie_aware_percentile(value, population):
    """
    Tie-aware percentile rank (0-100) of `value` within `population`, using
    the same 'mean rank' convention as scipy's percentileofscore(kind='mean')
    - ties split the difference rather than breaking arbitrarily in one
    direction. Implemented directly in pandas/numpy so this module doesn't
    require scipy as a dependency.
    """
    population = population.dropna()
    if len(population) == 0 or pd.isna(value):
        return None
    less = (population < value).sum()
    equal = (population == value).sum()
    return (less + 0.5 * equal) / len(population) * 100.0


def quantile_map_to_era(value, source_era_values, target_era_values):
    """Maps a value from its own era's distribution onto the equivalent
    percentile in a target (base) era's distribution. Uses tie-aware
    ('mean' rank) percentile scoring, standard for handling ties/clumped
    values rather than an arbitrary tie-break direction."""
    source_era_values = source_era_values.dropna()
    target_era_values = target_era_values.dropna()
    if len(source_era_values) == 0 or len(target_era_values) == 0 or pd.isna(value):
        return value
    percentile = _tie_aware_percentile(value, source_era_values) / 100.0
    return float(np.quantile(target_era_values, percentile))


def apply_era_adjustment(scored_table, row_a, row_b, base_era, weights=None):
    """
    Recomputes both teams' metrics, z-scores, category edges, and composite
    score as if each team's performance were restated into the base_era's
    distribution. Returns a dict shaped like compare_two_teams' output, so
    it's a drop-in replacement for the comparison view when the toggle is on.
    """
    scored_table = scored_table.copy()

    base_pop = scored_table[scored_table['era'] == base_era]

    def adjust_team(row):
        own_era = row['era']
        own_pop = scored_table[scored_table['era'] == own_era]
        adjusted = {}
        for label, col in ERA_ADJUSTABLE_METRICS.items():
            adjusted[label] = quantile_map_to_era(row[col], own_pop[col], base_pop[col])
        return adjusted

    adj_a = adjust_team(row_a)
    adj_b = adjust_team(row_b)

    # z-score the adjusted values against the BASE ERA's own mean/std,
    # so the composite is internally consistent within the base era's terms
    z_a, z_b = {}, {}
    for label, col in ERA_ADJUSTABLE_METRICS.items():
        base_mean, base_std = base_pop[col].mean(), base_pop[col].std()
        if base_std == 0 or pd.isna(base_std):
            z_a[label], z_b[label] = 0, 0
        else:
            z_a[label] = (adj_a[label] - base_mean) / base_std
            z_b[label] = (adj_b[label] - base_mean) / base_std

    # Playoffs: NOT quantile-mapped, but re-standardized against the base
    # era's own playoff-ordinal distribution for consistent combination
    playoff_mean, playoff_std = base_pop['playoff_ordinal'].mean(), base_pop['playoff_ordinal'].std()
    z_playoffs_a = (row_a['playoff_ordinal'] - playoff_mean) / playoff_std if playoff_std else 0
    z_playoffs_b = (row_b['playoff_ordinal'] - playoff_mean) / playoff_std if playoff_std else 0

    weight_key_map = {'Quality': 'quality', 'Roster': 'roster'}
    W = normalize_weights(weights) if weights is not None else dict(WEIGHTS)
    composite_a = sum(z_a[label] * W[weight_key_map[label]] for label in z_a) + z_playoffs_a * W['playoffs']
    composite_b = sum(z_b[label] * W[weight_key_map[label]] for label in z_b) + z_playoffs_b * W['playoffs']

    edges = {label: ('A' if adj_a[label] > adj_b[label] else 'B') for label in ERA_ADJUSTABLE_METRICS}
    edges['Playoffs'] = 'A' if row_a['playoff_ordinal'] > row_b['playoff_ordinal'] else ('B' if row_b['playoff_ordinal'] > row_a['playoff_ordinal'] else 'Tie')

    return {
        'base_era': base_era,
        'adjusted_a': adj_a, 'adjusted_b': adj_b,
        'z_a': z_a, 'z_b': z_b,
        'edges': edges,
        'composite_a': composite_a, 'composite_b': composite_b,
        'overall_winner': 'A' if composite_a > composite_b else 'B',
    }


def zscore(series):
    """Standardize a series; returns 0s if the series has no variance."""
    std = series.std()
    if std == 0 or pd.isna(std):
        return series * 0
    return (series - series.mean()) / std


def shrink_toward_average(wins, games, pseudo_games, league_avg=0.5):
    """Bayesian-style credibility shrinkage: blends observed win% with
    league-average, weighted by sample size vs. the pseudo_games constant."""
    if games is None or games == 0 or pd.isna(games):
        return league_avg
    observed = wins / games
    weight = games / (games + pseudo_games)
    return weight * observed + (1 - weight) * league_avg


def build_team_season_table(team_stats, playoff_results, league):
    """Joins team season stats with playoff round reached, adds ordinal playoff score.

    Distinguishes three cases, since collapsing them was a real scoring bug:
      - team not in playoff_results at all -> genuinely missed the playoffs -> ordinal 0
      - team in playoff_results with a resolved round -> real ordinal value
      - team in playoff_results but round is NaN (1980-83 bye-era ambiguity) ->
        genuinely unknown, NOT the same as missing the playoffs - flagged separately
        and given a neutral (population mean) ordinal for scoring purposes rather
        than a falsely low one, so it doesn't silently punish a team that we know
        made the playoffs but don't yet know how far they went.
    """
    df = team_stats.copy()

    playoff_lookup = playoff_results.set_index(['season', 'team_code'])['playoff_round_reached'].to_dict()
    made_playoffs_lookup = set(playoff_results.set_index(['season', 'team_code']).index)

    def get_round(row):
        key = (row['season'], row['team_code'])
        if key not in made_playoffs_lookup:
            return ('missed', None)
        result = playoff_lookup.get(key)
        if pd.isna(result):
            return ('ambiguous', None)
        return ('resolved', result)

    results = df.apply(get_round, axis=1)
    df['playoff_status'] = [r[0] for r in results]
    df['playoff_round_reached'] = [r[1] for r in results]

    df['playoff_ordinal_raw'] = df['playoff_round_reached'].map(PLAYOFF_ORDINAL)
    df.loc[df['playoff_status'] == 'missed', 'playoff_ordinal_raw'] = 0
    # ambiguous rows stay NaN here on purpose - filled with a neutral value below,
    # not zero, so the composite score doesn't punish a team for a data gap
    neutral_ordinal = df.loc[df['playoff_status'] == 'resolved', 'playoff_ordinal_raw'].mean()
    df['playoff_ordinal'] = df['playoff_ordinal_raw'].fillna(neutral_ordinal)
    df['era'] = df['season'].apply(lambda s: get_era(s, league))

    # Quality axis input: Massey srs where present, else net_rating (0.992
    # correlated, and net_rating covers every team-season) so a season without
    # game-log-derived srs still scores instead of dropping to NaN.
    if 'srs' in df.columns:
        df['quality_metric'] = df['srs'].where(df['srs'].notna(), df['net_rating'])
    else:
        df['quality_metric'] = df['net_rating']

    return df


def build_roster_strength_table(player_stats):
    """Roster-strength raw metric = sum of (PIE x minutes) over a team's
    single-team player rows.

    PIE is a per-game SHARE (intensive), so summing raw PIE across a roster is
    noisy and not additive the way Win Shares was. Weighting by minutes makes it
    extensive -- sum(pie x min) rewards both quality and playing time and behaves
    like a personnel-strength total. Multi-team combined rows (team_count > 1) are
    excluded, since a blended multi-team line attributed to one code would
    over-credit that team."""
    single = player_stats[player_stats['team_count'] == 1].copy()
    single['pie_min'] = single['pie'] * single['min']
    team = single.groupby(['season', 'team_code'])['pie_min'].sum().reset_index()
    return team.rename(columns={'pie_min': 'roster_strength'})


def compute_player_roster_shares(player_stats, team_stats=None):
    """Per-player roster impact and each player's share of their team's roster.

    Impact metric is player_impact = PIE x minutes (extensive; see
    build_roster_strength_table for why minutes-weighting rather than raw PIE).
    Share (pct_of_team) is player_impact as a percent of the team's POSITIVE
    impact pool per season -- the positive-only denominator keeps a few
    negative-impact bench players from shrinking the pool and inflating
    everyone else's share, exactly as the old positive-WS-pool did. Negative
    contributors still receive a (negative) share against that stable pool.

    Multi-team combined rows (team_count > 1 -- a player traded mid-season,
    from a season-wide non-team-filtered pull) are EXCLUDED from the share:
    a blended multi-team line isn't a coherent 'share of one team's roster'.
    (team_stats is accepted for signature compatibility but no longer needed,
    since the share is roster-internal rather than renormalized to team wins.)"""
    df = player_stats.copy()
    df['player_impact'] = df['pie'] * df['min']
    is_multi_team = df['team_count'] > 1

    positive_impact = df['player_impact'].where(df['player_impact'] > 0, 0)
    positive_pool = positive_impact[~is_multi_team].groupby(
        [df['season'], df['team_code']]).transform('sum')
    positive_pool = positive_pool.reindex(df.index)
    safe_pool = positive_pool.replace(0, np.nan)

    df['pct_of_team'] = (df['player_impact'] / safe_pool * 100).round(1)
    df.loc[is_multi_team, 'pct_of_team'] = None

    # Second, independent contribution share from Win Shares (win-translation)
    # to sit alongside the PIE-based footprint share, when WS has been built.
    # Same positive-pool convention. The gap between a player's WS share and PIE
    # share is the signal (footprint vs. how much it converted to winning).
    if 'ws' in df.columns:
        positive_ws = df['ws'].where(df['ws'] > 0, 0)
        ws_pool = positive_ws[~is_multi_team].groupby(
            [df['season'], df['team_code']]).transform('sum')
        ws_pool = ws_pool.reindex(df.index).replace(0, np.nan)
        df['pct_of_team_ws'] = (df['ws'] / ws_pool * 100).round(1)
        df.loc[is_multi_team, 'pct_of_team_ws'] = None
    return df


# ----------------------------------------------------------------------
# COACHING: hybrid "at the time" prior record, blending season-level data
# (1979-80+) with pre-cutoff tenure aggregates where available
# ----------------------------------------------------------------------

def compute_coach_prior_hybrid(coach_name, evaluated_season, coach_season_wins, coach_tenures,
                                 league, cutoff_season=None):
    """
    Builds a coach's full 'at the time' prior record by blending two sources:
      - season-level records (coach_season_wins) for any prior season within
        our team_season_stats coverage (DATA_CUTOFF_SEASON[league]+), used
        only where unambiguous (single coach that season)
      - tenure-level aggregates (coach_tenures) for any COMPLETE tenure that
        ended entirely before the cutoff season - our only source of
        pre-cutoff history

    A tenure that straddles the cutoff is deliberately excluded from the
    tenure-aggregate fallback, since the aggregate can't be split and using
    it whole would double-count games already captured at the season level.
    Conservative by design: some pre-cutoff portion of a straddling tenure
    stays invisible, but nothing gets double-counted.

    If coach_season_wins is empty for this league (no coach data sourced
    yet), every filter below simply returns zero rows and this function
    falls through to all-zero totals - compute_coaching_score's caller-side
    "no coach data found" fallback is what actually kicks in for scoring,
    this function itself doesn't need a special empty-data branch.
    """
    if cutoff_season is None:
        cutoff_season = DATA_CUTOFF_SEASON[league]
    cutoff_year = int(cutoff_season.split('-')[0]) + 1 if '-' in cutoff_season else int(cutoff_season)

    season_level = coach_season_wins[
        (coach_season_wins['Coach'] == coach_name) &
        (coach_season_wins['Season'] < evaluated_season) &
        (coach_season_wins['num_coaches'] == 1) &
        (coach_season_wins['W'].notna())  # exclude placeholder rows with no real record (pre-1980 gaps)
    ]
    season_wins = season_level['W'].sum()
    season_losses = season_level['L'].sum()
    season_count = len(season_level)

    tenure_wins = tenure_losses = tenure_seasons_added = 0
    if coach_tenures is not None:
        coach_rows = coach_tenures[coach_tenures['Coach'] == coach_name]
        pre_cutoff = coach_rows[coach_rows['To'] < cutoff_year]
        tenure_wins = pre_cutoff['Regular Season: W'].sum()
        tenure_losses = pre_cutoff['Regular Season: L'].sum()
        tenure_seasons_added = pre_cutoff['Yrs'].sum()

    total_wins = season_wins + tenure_wins
    total_losses = season_losses + tenure_losses

    return {
        'total_wins': total_wins,
        'total_losses': total_losses,
        'total_games': total_wins + total_losses,
        'seasons_counted': season_count + tenure_seasons_added,
        'pre_cutoff_tenure_games_included': tenure_wins + tenure_losses,
    }


def compute_coaching_score(team_season_key, coach_season_wins, league, coach_tenures=None, coach_awards=None):
    """
    For a given (Season, Team), finds the coach(es) that season, then computes
    an 'at the time' coaching score using the hybrid prior record above.

    If a season has multiple coaches (flagged in coach_season_wins), this
    scores using whichever coach's row appears first, and flags it - the
    actual games-based split between co-coaches isn't implemented.

    If coach_season_wins has zero rows for this league entirely (data not
    sourced yet), season_coaches is empty for every team-season, so this
    ALWAYS hits the 'No coach data found' branch below and returns the
    neutral 0.5 score league-wide - see the note on zscore() in
    build_full_scoring_table for why that's the correct degrade-gracefully
    behavior rather than a special case that needs handling here.
    """
    season, team = team_season_key
    # Degrade-gracefully guard: when a league has no coach data yet (WNBA), the
    # frame may be empty or even columnless (pd.DataFrame()). Short-circuit to the
    # neutral score rather than indexing a 'Season' column that isn't there. The
    # coach internals below stay on the legacy column contract on purpose -- they
    # get migrated when real WNBA coach data (and its actual schema) exists.
    if coach_season_wins is None or len(coach_season_wins) == 0 \
            or 'Season' not in coach_season_wins.columns:
        return {'coach_name': None, 'coaching_score': 0.5, 'note': 'No coach data found'}

    season_coaches = coach_season_wins[
        (coach_season_wins['Season'] == season) & (coach_season_wins['Team'] == team)
    ]

    if len(season_coaches) == 0:
        return {'coach_name': None, 'coaching_score': 0.5, 'note': 'No coach data found'}

    is_ambiguous = (season_coaches['num_coaches'] > 1).any()
    coach_name = season_coaches.iloc[0]['Coach']

    prior = compute_coach_prior_hybrid(coach_name, season, coach_season_wins, coach_tenures, league)
    prior_win_pct_shrunk = shrink_toward_average(prior['total_wins'], prior['total_games'], COACH_WIN_PSEUDO_GAMES)

    experience_seasons = prior['seasons_counted']
    experience_normalized = min(experience_seasons / 15.0, 1.0)

    accolade_score = 0.0
    if coach_awards is not None:
        prior_awards = coach_awards[
            (coach_awards['Coach'] == coach_name) & (coach_awards['Season'] < season)
        ]
        accolade_score = min(len(prior_awards) / 2.0, 1.0)

    won_award_this_season_coach = False
    if coach_awards is not None:
        won_award_this_season_coach = len(coach_awards[(coach_awards['Coach'] == coach_name) & (coach_awards['Season'] == season)]) > 0

    # NOTE: playoff coaching record is not yet folded into this score -
    # coach_season_wins currently tracks regular-season W/L only. Flagged
    # as a documented gap for a future iteration, not silently ignored.
    coaching_score = (
        0.55 * prior_win_pct_shrunk +
        0.25 * experience_normalized +
        0.20 * accolade_score
    )

    return {
        'coach_name': coach_name,
        'coaching_score': coaching_score,
        'prior_win_pct_shrunk': round(prior_win_pct_shrunk, 3),
        'prior_seasons_experience': experience_seasons,
        'is_rookie_coach': experience_seasons == 0,
        'has_win_loss_history': prior['total_games'] > 0,
        'used_pre_cutoff_tenure_data': prior['pre_cutoff_tenure_games_included'] > 0,
        'multi_coach_season': is_ambiguous,
        'won_award_this_season_coach': won_award_this_season_coach,
        'note': 'Multiple coaches this season - showing first, split not available' if is_ambiguous
                else ('Has prior coaching experience but no W/L record available at all (not even pre-cutoff tenure data) - win% defaulted to league average'
                      if experience_seasons > 0 and prior['total_games'] == 0 else None),
    }


# ----------------------------------------------------------------------
# EXECUTIVES: date-based season mapping + real team record proxy
# (Note: executives are NOT part of the scored composite - display/
# leaderboard use only, per project scope decision)
# ----------------------------------------------------------------------

def date_to_season(date_str):
    """Converts a calendar date (or bare year) to an NBA season label.
    NBA seasons run roughly Aug-June, so a date in Aug-Dec belongs to the
    season starting that year; a date in Jan-Jul belongs to the season
    that started the PREVIOUS year."""
    if pd.isna(date_str):
        return None
    s = str(date_str)
    try:
        if len(s) == 4:
            year, month = int(s), 10  # bare year - assume mid-season
        else:
            parts = s.split('-')
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 else 10
    except (ValueError, IndexError):
        return None
    start_year = year if month >= 8 else year - 1
    return f"{start_year}-{str(start_year+1)[-2:]}"


def build_exec_season_wins(exec_tenures, team_stats):
    """
    Mirrors build_coach_season_wins for executives: expands each exec's
    Start/End date range into per-season rows, then joins each season's
    REAL team win-loss record from File 1. This is a proxy, not a direct
    record - it assumes the exec held the role for the entire season
    whenever their tenure covers it, with the same multi-exec-per-season
    ambiguity flag the coach version uses.
    """
    expanded_rows = []
    for _, row in exec_tenures.iterrows():
        start_season = date_to_season(row.get('Start'))
        if start_season is None:
            continue
        end_season = date_to_season(row.get('End')) if pd.notna(row.get('End')) else None
        start_year = int(start_season.split('-')[0])
        end_year = int(end_season.split('-')[0]) if end_season else start_year

        for yr in range(start_year, end_year + 1):
            expanded_rows.append({
                'Team': row['Team'],
                'Executive': row['Executive'],
                'Season': f"{yr}-{str(yr+1)[-2:]}"
            })

    expanded = pd.DataFrame(expanded_rows).drop_duplicates()
    if len(expanded) == 0:
        return expanded

    overlap_counts = expanded.groupby(['Team', 'Season']).size().reset_index(name='num_execs')
    merged = expanded.merge(team_stats[['Season', 'Team', 'W', 'L', 'W/L%']], on=['Season', 'Team'], how='left')
    merged = merged.merge(overlap_counts, on=['Team', 'Season'])
    merged['attribution'] = merged['num_execs'].apply(
        lambda n: 'Full season record (single exec)' if n == 1
        else 'AMBIGUOUS - exec changed mid-season, split not available'
    )
    return merged


def get_coaches_for_season(coach_tenures, team, season):
    """
    Returns every coach whose tenure at `team` overlaps `season`, with a
    games-coached split where it's actually derivable from the data.

    Split IS derivable when a coach's tenure at this team is a single-season
    tenure (From == To == this year) - their tenure-aggregate G is then
    exactly that season's games, nothing else to disentangle.

    Split is NOT derivable when a coach's tenure at this team spans MULTIPLE
    years including this season as an endpoint (e.g. a coach's 5-year tenure
    that happens to end the same year a successor's tenure begins) - the
    tenure-aggregate G covers their whole multi-year run, not just this one
    season, and there's no way to isolate this season's games from that
    aggregate alone. Flagged explicitly rather than guessed at.
    """
    year = int(season.split('-')[0]) + 1
    overlapping = coach_tenures[
        (coach_tenures['Team'] == team) & (coach_tenures['From'] <= year) & (coach_tenures['To'] >= year)
    ]
    results = []
    for _, row in overlapping.iterrows():
        is_single_season_tenure = (row['From'] == row['To'] == year)
        results.append({
            'coach': row['Coach'],
            'games_this_season': int(row['Regular Season: G']) if is_single_season_tenure else None,
            'split_derivable': is_single_season_tenure,
            'tenure_span': f"{row['From']}-{row['To']}" if row['From'] != row['To'] else str(row['From']),
        })
    return results


def build_multi_coach_season_detail(scored_table, coach_tenures, coach_season_wins, coach_awards, season, team, league):
    """
    For a season with more than one coach, returns full detail on each:
    games/% split where derivable, and each individual coach's OWN
    'at the time' derived coaching score (computed independently of the
    games-split question, since prior-season history doesn't depend on
    how this particular season's games were divided).
    """
    coaches = get_coaches_for_season(coach_tenures, team, season)
    total_known_games = sum(c['games_this_season'] for c in coaches if c['split_derivable'])

    for c in coaches:
        if c['split_derivable'] and total_known_games:
            c['pct_of_season'] = round(c['games_this_season'] / total_known_games * 100, 1)
        else:
            c['pct_of_season'] = None

        prior = compute_coach_prior_hybrid(c['coach'], season, coach_season_wins, coach_tenures, league)
        prior_win_pct_shrunk = shrink_toward_average(prior['total_wins'], prior['total_games'], COACH_WIN_PSEUDO_GAMES)
        experience_normalized = min(prior['seasons_counted'] / 15.0, 1.0)
        accolade_score = 0.0
        if coach_awards is not None:
            prior_awards = coach_awards[(coach_awards['Coach'] == c['coach']) & (coach_awards['Season'] < season)]
            accolade_score = min(len(prior_awards) / 2.0, 1.0)
        c['coaching_score'] = 0.55 * prior_win_pct_shrunk + 0.25 * experience_normalized + 0.20 * accolade_score
        c['prior_seasons_experience'] = prior['seasons_counted']

        won_award_this_season = False
        if coach_awards is not None:
            won_award_this_season = len(coach_awards[(coach_awards['Coach'] == c['coach']) & (coach_awards['Season'] == season)]) > 0
        c['won_award_this_season'] = won_award_this_season

    return coaches


# ----------------------------------------------------------------------
# EXECUTIVE SCORING (display/leaderboard only - NOT part of the composite,
# per project scope decision, to avoid the circularity of scoring a team's
# own win total as if it were an independent signal about the exec)
# ----------------------------------------------------------------------

EXEC_WIN_PSEUDO_GAMES = 41  # same shrinkage constant as coaching, for consistency


def compute_exec_prior(exec_name, evaluated_season, exec_season_wins):
    """
    Exec analog of compute_coach_prior_hybrid - but execs have no tenure-
    aggregate fallback (exec_tenures has no W/L data at all, only dates),
    so pre-1980 win/loss history is simply unavailable for execs, full stop.
    Pre-1980 tenure YEARS still count toward experience (dates exist even
    when outcomes don't), just not toward the win% component.
    """
    prior = exec_season_wins[
        (exec_season_wins['Executive'] == exec_name) &
        (exec_season_wins['Season'] < evaluated_season) &
        (exec_season_wins['num_execs'] == 1) &
        (exec_season_wins['W'].notna())
    ]
    wins, losses = prior['W'].sum(), prior['L'].sum()
    return {'total_wins': wins, 'total_losses': losses, 'total_games': wins + losses, 'seasons_counted': len(prior)}


def compute_exec_score(team_season_key, exec_season_wins, exec_awards=None):
    """
    Parallel construction to compute_coaching_score: 'at the time' (prior
    seasons only, same anti-circularity logic that makes coaching_score
    defensible) win% (shrunk), experience, and prior awards.
    """
    season, team = team_season_key
    season_execs = exec_season_wins[(exec_season_wins['Season'] == season) & (exec_season_wins['Team'] == team)]
    if len(season_execs) == 0:
        return {'exec_name': None, 'exec_score': None, 'note': 'No exec data found'}

    is_ambiguous = (season_execs['num_execs'] > 1).any()
    exec_name = season_execs.iloc[0]['Executive']

    prior = compute_exec_prior(exec_name, season, exec_season_wins)
    prior_win_pct_shrunk = shrink_toward_average(prior['total_wins'], prior['total_games'], EXEC_WIN_PSEUDO_GAMES)
    experience_normalized = min(prior['seasons_counted'] / 15.0, 1.0)

    accolade_score = 0.0
    if exec_awards is not None:
        prior_awards = exec_awards[(exec_awards['Executive'] == exec_name) & (exec_awards['Season'] < season)]
        accolade_score = min(len(prior_awards) / 2.0, 1.0)

    exec_score = 0.55 * prior_win_pct_shrunk + 0.25 * experience_normalized + 0.20 * accolade_score

    won_award_this_season = False
    if exec_awards is not None:
        won_award_this_season = len(exec_awards[(exec_awards['Executive'] == exec_name) & (exec_awards['Season'] == season)]) > 0

    return {
        'exec_name': exec_name, 'exec_score': exec_score,
        'prior_seasons_experience': prior['seasons_counted'],
        'has_win_loss_history': prior['total_games'] > 0,
        'multi_exec_season': is_ambiguous,
        'won_award_this_season': won_award_this_season,
    }


def percentile_of(value, population):
    """Percentile rank of `value` within `population` (0-100), tie-aware."""
    result = _tie_aware_percentile(value, population)
    return round(result, 1) if result is not None else None


# ----------------------------------------------------------------------
# TEAM DISPLAY NAMES - for cascading team-then-season dropdowns
# ----------------------------------------------------------------------

TEAM_CODE_TO_NAME = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets",
    "NJN": "New Jersey Nets", "CHA": "Charlotte Bobcats", "CHH": "Charlotte Hornets",
    "CHO": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "SDC": "San Diego Clippers", "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies", "VAN": "Vancouver Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NOH": "New Orleans Hornets", "NOK": "New Orleans/Oklahoma City Hornets", "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "SEA": "Seattle SuperSonics", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "KCK": "Kansas City Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "NOJ": "New Orleans Jazz",
    "WAS": "Washington Wizards", "WSB": "Washington Bullets",
}


def get_team_display_options(team_stats):
    """
    Returns a sorted list of (display_label, team_code) tuples for a team
    dropdown, using full franchise names. Where two codes share the same
    name (e.g. CHH and CHO both being "Charlotte Hornets" - two different
    franchises 12 years apart), auto-disambiguates with each code's actual
    season range rather than requiring that to be hardcoded per-team.
    """
    # Prefer the franchise name carried in the canonical data (correct per
    # league -- WNBA 'WAS' is the Mystics, NBA 'WAS' the Wizards), falling back
    # to the NBA name map, then the raw code. This is why no hardcoded WNBA name
    # map is needed: the names ride along in team_name.
    if 'team_name' in team_stats.columns:
        name_by_code = team_stats.groupby('team_code')['team_name'].last().to_dict()
    else:
        name_by_code = {}

    def name_for(code):
        return name_by_code.get(code) or TEAM_CODE_TO_NAME.get(code, code)

    codes_present = team_stats['team_code'].unique()
    name_counts = {}
    for code in codes_present:
        name = name_for(code)
        name_counts[name] = name_counts.get(name, 0) + 1

    options = []
    for code in codes_present:
        name = name_for(code)
        if name_counts[name] > 1:
            seasons = team_stats[team_stats['team_code'] == code]['season']
            label = f"{name} ({seasons.min()} to {seasons.max()})"
        else:
            label = name
        options.append((label, code))
    return sorted(options)


def get_season_options_for_team(scored_table, team_code):
    """
    Returns a sorted (descending) list of (display_label, season) tuples
    for the season dropdown, prefixing a trophy emoji on seasons that
    team actually won the championship.
    """
    team_rows = scored_table[scored_table['team_code'] == team_code].sort_values('season', ascending=False)
    options = []
    for _, row in team_rows.iterrows():
        is_champion = row.get('playoff_round_reached') == 'Champion'
        label = f"🏆 {row['season']}" if is_champion else row['season']
        options.append((label, row['season']))
    return options


# ----------------------------------------------------------------------
# LEADERBOARDS
# ----------------------------------------------------------------------

def build_coach_leaderboard(coach_season_wins, coach_tenures, league, coach_awards=None):
    """One row per coach: explicit cutoff+ record AND all-time record (where
    pre-cutoff tenure data is available) shown as SEPARATE columns, rather
    than one ambiguous 'Career W' total - a single blended number was
    misleading users into thinking pre-cutoff legends (e.g. Red Auerbach)
    weren't being counted, when the underlying math was already correct.

    Returns an empty DataFrame (not an error) if coach_season_wins has no
    rows for this league - same "not sourced yet" degrade-gracefully
    pattern build_exec_leaderboard already used, so a league missing this
    data just shows an empty leaderboard rather than crashing on the
    sort_values() call below (an all-columns-missing empty DataFrame has
    no 'W (all-time...)' column to sort by)."""
    if len(coach_season_wins) == 0:
        return pd.DataFrame()

    cutoff_year = _cutoff_year(league)
    cutoff_label = str(cutoff_year)

    rows = []
    for coach in coach_season_wins['Coach'].unique():
        career_post_cutoff = coach_season_wins[
            (coach_season_wins['Coach'] == coach) &
            (coach_season_wins['num_coaches'] == 1) &
            (coach_season_wins['W'].notna())
        ]
        wins_post_cutoff, losses_post_cutoff = career_post_cutoff['W'].sum(), career_post_cutoff['L'].sum()
        seasons_post_cutoff = career_post_cutoff['Season'].nunique()

        pre_cutoff_wins = pre_cutoff_losses = pre_cutoff_seasons = 0
        teams = []
        if coach_tenures is not None:
            coach_rows = coach_tenures[coach_tenures['Coach'] == coach]
            teams = coach_rows['Team'].unique()
            pre_cutoff = coach_rows[coach_rows['To'] < cutoff_year]
            pre_cutoff_wins = pre_cutoff['Regular Season: W'].sum()
            pre_cutoff_losses = pre_cutoff['Regular Season: L'].sum()
            pre_cutoff_seasons = pre_cutoff['Yrs'].sum()

        total_wins = wins_post_cutoff + pre_cutoff_wins
        total_losses = losses_post_cutoff + pre_cutoff_losses
        award_count = 0
        if coach_awards is not None:
            award_count = len(coach_awards[coach_awards['Coach'] == coach])

        rows.append({
            'Coach': coach,
            'Teams': ', '.join(sorted(teams)),
            f'W ({cutoff_label}+)': int(wins_post_cutoff),
            f'L ({cutoff_label}+)': int(losses_post_cutoff),
            f'W/L% ({cutoff_label}+)': round(wins_post_cutoff / (wins_post_cutoff + losses_post_cutoff), 3) if (wins_post_cutoff + losses_post_cutoff) > 0 else None,
            f'W (all-time, incl. pre-{cutoff_label})': int(total_wins) if pd.notna(total_wins) else None,
            f'L (all-time, incl. pre-{cutoff_label})': int(total_losses) if pd.notna(total_losses) else None,
            'W/L% (all-time)': round(total_wins / (total_wins + total_losses), 3) if (total_wins + total_losses) > 0 else None,
            'Seasons (all-time)': seasons_post_cutoff + pre_cutoff_seasons,
            'Career Awards': award_count,
        })
    return pd.DataFrame(rows).sort_values(f'W (all-time, incl. pre-{cutoff_label})', ascending=False, na_position='last').reset_index(drop=True)


def build_exec_leaderboard(exec_season_wins, exec_awards=None):
    """One row per executive: aggregated team performance during their
    tenure(s) (1980+ only - no pre-1980 fallback exists for execs since
    exec_tenures has no win/loss data at all), plus career award count."""
    if len(exec_season_wins) == 0:
        return pd.DataFrame()
    unambiguous = exec_season_wins[exec_season_wins['num_execs'] == 1]
    grouped = unambiguous.groupby('Executive').agg(
        Teams=('Team', lambda x: ', '.join(sorted(x.unique()))),
        Seasons=('Season', 'nunique'),
        Career_W=('W', 'sum'),
        Career_L=('L', 'sum'),
    ).reset_index()
    grouped['Career W/L%'] = (grouped['Career_W'] / (grouped['Career_W'] + grouped['Career_L'])).round(3)
    grouped = grouped.rename(
        columns={'Career_W': 'W (1980+ only)', 'Career_L': 'L (1980+ only)', 'Seasons': 'Seasons (as exec)'}
    )
    if exec_awards is not None:
        grouped['Career Awards'] = grouped['Executive'].apply(lambda e: len(exec_awards[exec_awards['Executive'] == e]))
    else:
        grouped['Career Awards'] = None
    return grouped.sort_values('W (1980+ only)', ascending=False).reset_index(drop=True)


def build_year_over_year_leaderboard(scored_table):
    """
    One row per team-season that HAS a valid prior season in scored_table
    (bridging relocations/renames via the standard prior-season lookup),
    with the year-over-year change in composite score and several 'juicy'
    underlying metrics - surfaces the biggest jumps AND biggest collapses
    for further digging. This is the direct analytical payoff of the
    whole project's original 'which teams made a big jump' premise.
    """
    rows = []
    for _, row in scored_table.iterrows():
        prior = get_prior_season_row(scored_table, row['season'], row['team_code'])
        if prior is None:
            continue
        rows.append({
            'season': row['season'], 'team_code': row['team_code'],
            'prior_season': f"{prior['team_code']} {prior['season']}",
            'composite_change': round(row['composite_score'] - prior['composite_score'], 3),
            'win_change': int(row['w']) - int(prior['w']),
            'quality_change': round(row['quality_metric'] - prior['quality_metric'], 2),
            'net_rating_change': round(row['net_rating'] - prior['net_rating'], 2),
            'pace_change': round(row['pace'] - prior['pace'], 2),
            'ts_pct_change': round(row['ts_pct'] - prior['ts_pct'], 3),
            'roster_change': round(row['roster_strength'] - prior['roster_strength'], 1),
        })
    return pd.DataFrame(rows).sort_values('composite_change', ascending=False).reset_index(drop=True)


def build_player_leaderboard(player_table, season_filter=None):
    """Player-season leaderboard, optionally filtered to a single season.
    Sorts by Win Shares when present (the extensive win-attribution metric),
    else by player_impact (PIE x minutes). Both PIE and WS surface so the
    footprint vs. win-translation views sit side by side."""
    df = player_table.copy()
    if season_filter and season_filter != "All seasons":
        df = df[df['season'] == season_filter]
    cols = ['player_name', 'season', 'team_code', 'age', 'min',
            'ws', 'ows', 'dws', 'pie', 'player_impact', 'pct_of_team', 'pct_of_team_ws']
    cols = [c for c in cols if c in df.columns]
    sort_col = 'ws' if 'ws' in df.columns else 'player_impact'
    return df[cols].sort_values(sort_col, ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# TRANSPARENCY PANEL: what changed for a team vs. its immediately prior
# season - roster additions/losses, coach change, exec change.
# ----------------------------------------------------------------------

# Franchise lineage groups, so "prior season" bridges relocations/renames
# rather than treating a relabeled franchise as having no history. Each group
# lists every code the SAME continuous business entity has used. KEYED BY
# LEAGUE, because codes collide across leagues (WNBA 'DAL' is the Wings lineage;
# NBA 'DAL' is the Mavericks, its own single code) -- a flat combined list would
# cross-bridge them. Codes not listed under a league are their own single-code
# lineage.
FRANCHISE_LINEAGES = {
    'NBA': [
        {'SEA', 'OKC'},
        {'NJN', 'BRK'},
        {'CHH', 'NOH', 'NOK', 'NOP'},   # original Hornets -> New Orleans -> Katrina relabel -> Pelicans
        {'CHA', 'CHO'},                  # Bobcats -> renamed Hornets (current Charlotte entity)
        {'KCK', 'SAC'},
        {'SDC', 'LAC'},
        {'WSB', 'WAS'},
        {'NOJ', 'UTA'},
    ],
    'WNBA': [
        {'UTA', 'SAS', 'LVA'},   # Utah Starzz -> San Antonio Stars -> Las Vegas Aces
        {'DET', 'TUL', 'DAL'},   # Detroit Shock -> Tulsa Shock -> Dallas Wings
        {'ORL', 'CON'},          # Orlando Miracle -> Connecticut Sun
    ],
}


def _lineages_for(scored_table):
    """Franchise-lineage sets for whichever league the scored table holds,
    read from its own `league` column so no league argument has to be threaded
    through the display callers."""
    league = scored_table['league'].iloc[0] if ('league' in scored_table.columns and len(scored_table)) else None
    return FRANCHISE_LINEAGES.get(league, [])


def get_prior_season_row(scored_table, season, team):
    """Finds the immediately prior season for a team, bridging franchise
    relocations/renames via FRANCHISE_LINEAGES. Returns None for a team's
    first-ever season (expansion team) or the first season after a
    relocation if the predecessor isn't in our data range."""
    lineage = next((group for group in _lineages_for(scored_table) if team in group), {team})
    candidates = scored_table[(scored_table['team_code'].isin(lineage)) & (scored_table['season'] < season)]
    if len(candidates) == 0:
        return None
    return candidates.sort_values('season').iloc[-1]


def get_prior_season_label(season, team):
    """
    Pure calendar-math prior season label, bridging franchise relocations
    via FRANCHISE_LINEAGES where the lineage's code changed AT that exact
    boundary. Unlike get_prior_season_row, this does NOT depend on the
    scored table having a matching row - it always returns a label, so it
    can be used to look up coach/exec/player data directly from their own
    source tables even for seasons outside the main scored population
    (e.g. 1978-79, one year before File 1's coverage begins).
    """
    year = int(season.split('-')[0])
    prior_season = f"{year-1}-{str(year)[-2:]}"
    return prior_season, team


def get_coach_name_for_season(coach_tenures, team, season):
    """Direct coach_tenures lookup for a specific team-season, independent
    of the scored table - works for any season coach_tenures covers,
    including seasons before File 1's 1979-80 floor."""
    year = int(season.split('-')[0]) + 1
    matches = coach_tenures[(coach_tenures['Team'] == team) & (coach_tenures['From'] <= year) & (coach_tenures['To'] >= year)]
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        return ' / '.join(matches['Coach'].tolist())
    return matches.iloc[0]['Coach']


def get_exec_name_for_season(exec_tenures, team, season):
    """Direct exec_tenures lookup for a specific team-season, using the
    same date-to-season mapping as build_exec_season_wins, independent of
    the scored table."""
    year = int(season.split('-')[0]) + 1
    matches = []
    for _, row in exec_tenures[exec_tenures['Team'] == team].iterrows():
        start_season = date_to_season(row.get('Start'))
        if start_season is None:
            continue
        end_season = date_to_season(row.get('End')) if pd.notna(row.get('End')) else None
        start_year = int(start_season.split('-')[0]) + 1
        end_year = int(end_season.split('-')[0]) + 1 if end_season else 9999
        if start_year <= year <= end_year:
            matches.append(row['Executive'])
    if len(matches) == 0:
        return None
    return ' / '.join(matches) if len(matches) > 1 else matches[0]


def build_transparency_panel(scored_table, player_stats, exec_season_wins, season, team,
                               coach_tenures=None, exec_tenures=None, min_minutes=300):
    """
    Returns a dict describing what changed for (season, team) vs. its
    immediately prior season: roster additions/losses (sorted by minutes,
    so the most notable changes surface first), and whether the coach
    or executive changed.

    Prior-season lookup has two tiers:
      1. Normal case: prior season/team found as a real row in the scored
         table (the usual path for any season with a prior year also
         covered by File 1).
      2. Fallback (e.g. a 1979-80 selection, one year before File 1's
         1979-80 floor): the scored table has no prior row, but coach and
         exec data (coach_tenures/exec_tenures) already extends much
         further back, so those checks still work via direct lookup.
         Player roster diff in this fallback tier depends entirely on
         whether player_stats happens to include that prior season -
         it degrades gracefully (empty added/lost lists) if not, and
         starts working automatically the moment that data exists, with
         no further code changes needed.

    `min_minutes` filters BOTH added and lost lists to players who logged at
    least that many minutes on the relevant side of the comparison (current
    season for additions, prior season for losses) - a end-of-bench player
    who appeared in a handful of garbage-time minutes isn't a "key" roster
    change worth surfacing. Default 300 (roughly a rotation player over a
    WNBA season). Pass 0 to disable filtering entirely.
    """
    prior_row = get_prior_season_row(scored_table, season, team)

    if prior_row is not None:
        prior_season, prior_team = prior_row['season'], prior_row['team_code']
        current_row = scored_table[(scored_table['season'] == season) & (scored_table['team_code'] == team)].iloc[0]
        # coach_name is a display-only field attached from the (dormant for WNBA)
        # coaching pass; .get keeps this working whether or not it's present.
        coach_current_name = current_row.get('coach_name')
        coach_prior_name = prior_row.get('coach_name')
        fallback_mode = False
    elif season == '1979-80':
        # ONLY the 1979-80 boundary is safe to bridge past File 1's floor -
        # we know from actual NBA history that no team began fresh at this
        # exact point, so a lookup miss here is a genuine data-coverage gap,
        # not a real "team didn't exist" case. Any OTHER season's lookup
        # miss (e.g. a real expansion team's actual first year) should NOT
        # attempt this fallback - coach_tenures.csv has a known mislabeling
        # issue for at least one team (CHA rows containing the original CHH
        # franchise's history), which makes trusting it blindly for
        # arbitrary "first season" lookups unsafe beyond this one verified
        # boundary.
        prior_season, prior_team = get_prior_season_label(season, team)
        prior_year = int(prior_season.split('-')[0]) + 1
        has_any_source_data = (
            (coach_tenures is not None and len(coach_tenures[
                (coach_tenures['Team'] == prior_team) & (coach_tenures['From'] <= prior_year) & (coach_tenures['To'] >= prior_year)
            ]) > 0) or
            (len(player_stats[(player_stats['season'] == prior_season) & (player_stats['team_code'] == prior_team)]) > 0)
        )
        if not has_any_source_data:
            return {'has_prior_data': False}
        coach_current_name = get_coach_name_for_season(coach_tenures, team, season) if coach_tenures is not None else None
        coach_prior_name = get_coach_name_for_season(coach_tenures, prior_team, prior_season) if coach_tenures is not None else None
        fallback_mode = True
    else:
        # any season other than the verified 1979-80 boundary with a
        # scored-table lookup miss is a genuine "no prior team" case
        # (expansion franchise) - not attempted via fallback
        return {'has_prior_data': False}

    current_roster = set(player_stats[(player_stats['season'] == season) & (player_stats['team_code'] == team)]['player_name'])
    prior_roster_df = player_stats[(player_stats['season'] == prior_season) & (player_stats['team_code'] == prior_team)]
    prior_roster = set(prior_roster_df['player_name'])

    # if there's genuinely zero player data for the prior season (not just
    # some players missing), every current player would falsely look like
    # a "new addition" - that's a missing-data artifact, not a real signal,
    # so suppress the roster diff in that case rather than show it as fact
    player_data_available_for_prior = len(prior_roster_df) > 0

    if player_data_available_for_prior:
        added_names = current_roster - prior_roster
        lost_names = prior_roster - current_roster
        current_players = player_stats[(player_stats['season'] == season) & (player_stats['team_code'] == team)]
        # min + ws + pie: playing time, win value, and footprint. ws leads when
        # built (it's the headline player-value metric); pie stays as the
        # outcome-agnostic footprint; usg_pct adds role. efg_pct/tov_pct omitted
        # to keep the diff scannable (ws already carries efficiency-of-value).
        base_cols = ['player_name', 'min', 'ws', 'pie', 'usg_pct']
        diff_cols = [c for c in base_cols if c in current_players.columns]
        added = current_players[current_players['player_name'].isin(added_names)][diff_cols].sort_values('min', ascending=False)
        lost = prior_roster_df[prior_roster_df['player_name'].isin(lost_names)][diff_cols].sort_values('min', ascending=False)
        if min_minutes:
            added = added[added['min'] >= min_minutes]
            lost = lost[lost['min'] >= min_minutes]
    else:
        diff_cols = [c for c in ['player_name', 'min', 'ws', 'pie', 'usg_pct'] if c in player_stats.columns]
        added = pd.DataFrame(columns=diff_cols)
        lost = pd.DataFrame(columns=diff_cols)

    coach_changed = coach_current_name != coach_prior_name if (coach_current_name and coach_prior_name) else None

    exec_current_name = exec_prior_name = None
    exec_changed = None
    if fallback_mode and exec_tenures is not None:
        exec_current_name = get_exec_name_for_season(exec_tenures, team, season)
        exec_prior_name = get_exec_name_for_season(exec_tenures, prior_team, prior_season)
        if exec_current_name is not None and exec_prior_name is not None:
            exec_changed = exec_current_name != exec_prior_name
    elif exec_season_wins is not None and len(exec_season_wins):
        cur_exec_rows = exec_season_wins[(exec_season_wins['Season'] == season) & (exec_season_wins['Team'] == team)]
        prior_exec_rows = exec_season_wins[(exec_season_wins['Season'] == prior_season) & (exec_season_wins['Team'] == prior_team)]
        exec_current_name = cur_exec_rows.iloc[0]['Executive'] if len(cur_exec_rows) else None
        exec_prior_name = prior_exec_rows.iloc[0]['Executive'] if len(prior_exec_rows) else None
        if exec_current_name is not None and exec_prior_name is not None:
            exec_changed = exec_current_name != exec_prior_name

    return {
        'has_prior_data': True,
        'prior_season': prior_season, 'prior_team': prior_team,
        'is_relocation_year': prior_team != team,
        'fallback_mode': fallback_mode,
        'player_data_available_for_prior': player_data_available_for_prior,
        'players_added': added, 'players_lost': lost,
        'coach_changed': coach_changed,
        'coach_current': coach_current_name, 'coach_prior': coach_prior_name,
        'exec_changed': exec_changed,
        'exec_current': exec_current_name, 'exec_prior': exec_prior_name,
    }


# ----------------------------------------------------------------------
# TWO-TEAM COMPARISON
# ----------------------------------------------------------------------

def compare_two_teams(scored_table, player_ws_table, team_season_a, team_season_b):
    """
    team_season_a / b: tuples of (Season, Team)
    Returns a dict with each team's row from the scored table, a per-category
    edge (which team wins each of the 5 scored categories), the overall
    composite winner, and each team's top roster contributors.
    """
    def get_row(season, team):
        match = scored_table[(scored_table['season'] == season) & (scored_table['team_code'] == team)]
        return match.iloc[0] if len(match) else None

    a = get_row(*team_season_a)
    b = get_row(*team_season_b)
    if a is None or b is None:
        missing = team_season_a if a is None else team_season_b
        raise ValueError(f"No data found for {missing}")

    # the three scored categories, in composite order
    edges = {
        'Quality': 'A' if a['quality_metric'] > b['quality_metric'] else ('B' if b['quality_metric'] > a['quality_metric'] else 'Tie'),
        'Playoffs': 'A' if a['playoff_ordinal'] > b['playoff_ordinal'] else ('B' if b['playoff_ordinal'] > a['playoff_ordinal'] else 'Tie'),
        'Roster': 'A' if a['roster_strength'] > b['roster_strength'] else ('B' if b['roster_strength'] > a['roster_strength'] else 'Tie'),
    }
    overall_winner = 'A' if a['composite_score'] > b['composite_score'] else 'B'

    def top_roster(season, team, min_minutes=300, n=25):
        """Rotation-level roster, sorted by value (WS, falling back to
        player_impact/PIE). Filtered by `min_minutes` rather than a hard
        top-N-by-value cutoff -- a heavy-minutes player who had a rough,
        low-efficiency season (e.g. negative Win Shares from high-volume,
        low-efficiency scoring) still played real minutes and belongs on
        this list; a fixed top-8-by-WS cutoff would silently drop them.
        `n` is a generous safety cap (WNBA rosters top out around 20), not
        the intended filter."""
        roster = player_ws_table[(player_ws_table['season'] == season) & (player_ws_table['team_code'] == team)]
        if min_minutes and 'min' in roster.columns:
            roster = roster[roster['min'] >= min_minutes]
        for sort_col in ('ws', 'player_impact', 'pie'):
            if sort_col in roster.columns:
                return roster.sort_values(sort_col, ascending=False).head(n)
        return roster.head(n)

    return {
        'team_a': a, 'team_b': b,
        'quality_a': a['quality_metric'], 'quality_b': b['quality_metric'],
        'net_rating_a': a['net_rating'], 'net_rating_b': b['net_rating'],
        'edges': edges,
        'overall_winner': overall_winner,
        'roster_a': top_roster(*team_season_a),
        'roster_b': top_roster(*team_season_b),
    }


# ----------------------------------------------------------------------
# FULL PIPELINE
# ----------------------------------------------------------------------

def rescore(scored_table, weights=None):
    """Recompute composite_score and every rank column from the already-computed
    z-columns (z_quality, z_playoffs, z_roster) under a new weight set.

    This is the cheap operation the app calls on each weight-slider change: the
    expensive table build (playoff joins, roster aggregation, z-scoring) is done
    once by build_full_scoring_table; only the weighted sum and ranks move here.
    `weights` is any dict over the three categories (auto-normalized); None uses
    the module default."""
    w = normalize_weights(weights) if weights is not None else dict(WEIGHTS)
    df = scored_table.copy()
    df['composite_score'] = (
        w['quality'] * df['z_quality'] +
        w['playoffs'] * df['z_playoffs'] +
        w['roster'] * df['z_roster']
    )
    df['season_rank'] = df.groupby('season')['composite_score'].rank(ascending=False, method='min').astype(int)
    df['season_total'] = df.groupby('season')['composite_score'].transform('count').astype(int)
    df['era_rank'] = df.groupby('era')['composite_score'].rank(ascending=False, method='min').astype(int)
    df['era_total'] = df.groupby('era')['composite_score'].transform('count').astype(int)
    df['all_time_rank'] = df['composite_score'].rank(ascending=False, method='min').astype(int)
    df['all_time_total'] = len(df)
    return df.sort_values('composite_score', ascending=False).reset_index(drop=True)


def build_full_scoring_table(team_stats, playoff_results, player_stats, coach_season_wins, league,
                               coach_tenures=None, coach_awards=None, exec_season_wins=None, exec_awards=None,
                               weights=None):
    """
    Runs every team-season through the full scoring pipeline and returns
    one row per team-season with all raw metrics, sub-scores, and the
    final composite score + z-scores used to rank it. Exec scoring (if
    exec_season_wins is provided) is computed and attached but NOT part
    of the composite - display/leaderboard use only.

    If coach_season_wins is empty for this league (data not sourced yet),
    every row's coaching_score comes back as the same neutral 0.5 (see
    compute_coaching_score), so z_coaching below is zscore() of a constant
    series - which zscore() maps to all zeros rather than dividing by a
    zero std. The coaching category still gets its full WEIGHTS['coaching']
    slice of the composite, it just contributes exactly 0 to every team
    equally, so RELATIVE ranking within that league is unaffected by the
    missing category; only the composite's absolute scale shifts slightly
    versus a league where coaching does differentiate teams. Re-weighting
    to redistribute that 15% elsewhere isn't done automatically, since a
    league can partially source coach data mid-project and that would make
    the weights themselves data-dependent - flagged here for visibility,
    not treated as a bug to silently patch around.
    """
    df = build_team_season_table(team_stats, playoff_results, league)
    roster = build_roster_strength_table(player_stats)
    df = df.merge(roster, on=['season', 'team_code'], how='left')
    df['roster_strength'] = df['roster_strength'].fillna(0)

    # Coaching is display-only now (removed from the composite), but the pass
    # still runs so the transparency panel's coach_name field is populated. For
    # the WNBA it degrades to neutral/None across the board (no coach data).
    coaching_results = [
        compute_coaching_score((row['season'], row['team_code']), coach_season_wins, league, coach_tenures, coach_awards)
        for _, row in df.iterrows()
    ]
    coaching_df = pd.DataFrame(coaching_results)
    df = pd.concat([df.reset_index(drop=True), coaching_df.reset_index(drop=True)], axis=1)

    if exec_season_wins is not None:
        exec_results = [
            compute_exec_score((row['season'], row['team_code']), exec_season_wins, exec_awards)
            for _, row in df.iterrows()
        ]
        exec_df = pd.DataFrame(exec_results)
        # rename to avoid column collisions with the coaching fields (e.g. both
        # dicts use 'prior_seasons_experience') - concat would otherwise silently
        # create two same-named columns rather than erroring
        exec_df = exec_df.rename(columns={
            'prior_seasons_experience': 'exec_prior_seasons_experience',
            'has_win_loss_history': 'exec_has_win_loss_history',
            'note': 'exec_note',
        })
        df = pd.concat([df.reset_index(drop=True), exec_df.reset_index(drop=True)], axis=1)

    # Three z-scored categories -> composite. quality_metric is the Massey srs
    # (net_rating fallback) set in build_team_season_table; roster_strength is
    # sum(pie x min). Coaching and net-rating-as-its-own-slot are intentionally
    # gone (see module docstring / WEIGHTS).
    # Three z-scored categories -> composite. quality_metric is the Massey srs
    # (net_rating fallback) set in build_team_season_table; roster_strength is
    # sum(pie x min). Coaching and net-rating-as-its-own-slot are intentionally
    # gone (see module docstring / WEIGHTS).
    df['z_quality'] = zscore(df['quality_metric'])
    df['z_playoffs'] = zscore(df['playoff_ordinal'])
    df['z_roster'] = zscore(df['roster_strength'])

    # composite + ranks under the requested weights (default WEIGHTS). The
    # z-columns above are weight-independent, so re-weighting is delegated to the
    # cheap rescore() below -- which is exactly what the app calls on each slider
    # change without paying for the joins/aggregation again.
    return rescore(df, weights)
