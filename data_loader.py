"""
data_loader.py

Loads and caches all data files the app needs, per league. Expects the
files to live in a ./data/processed/{league}/ folder next to this script,
using the same filenames the project's combiner scripts / nba_api pipeline
produce:

  team_season_stats.csv     (File 1)
  playoff_results.csv       (File 2, API-sourced)
  playoff_results_manual.csv(File 2 supplement, OPTIONAL - manually-sourced
                              seasons the API doesn't cover, e.g. the
                              confirmed WNBA 1997-2000 playoff gap)
  player_season_stats.csv   (File 3)
  coach_season_wins.csv     (File 4, OPTIONAL - app degrades gracefully
                              without it, see below)
  coach_tenures.csv         (File 4, full tenure detail for display, OPTIONAL)
  exec_tenures.csv          (File 4, execs - display only, OPTIONAL)
  coach_awards.csv          (File 5, OPTIONAL)
  exec_awards.csv           (File 5, OPTIONAL)

LEAGUE FOLDERS
--------------
A league whose folder is missing entirely, or missing team_season_stats.csv
(the one truly required file), returns {'league_available': False} instead
of raising - the app is responsible for showing a coming-soon state rather
than crashing. This is the expected state for a league whose pipeline pull
hasn't been run yet.

COACH DATA IS OPTIONAL, NOT REQUIRED
-------------------------------------
Unlike the original single-league version, coach_season_wins.csv is now
OPTIONAL. If it's missing, we return an empty-but-correctly-shaped
DataFrame rather than None, so downstream code (compute_coaching_score,
build_coach_leaderboard, etc.) can keep filtering/grouping on it without
special-casing None vs. an empty frame. compute_coaching_score already
falls back to a neutral 0.5 score when it finds zero rows for a team-season
- with an empty table that fallback fires for EVERY team-season in that
league, which is the correct "we don't have this yet" behavior: the
coaching category becomes a constant, and zscore() maps a constant series
to all zeros, so it stops differentiating teams without crashing or
silently distorting the composite score.
"""

import pandas as pd
import streamlit as st
import os

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')

COACH_SEASON_WINS_COLUMNS = ['Season', 'Team', 'Coach', 'num_coaches', 'W', 'L']


def _empty_coach_season_wins():
    return pd.DataFrame(columns=COACH_SEASON_WINS_COLUMNS)


@st.cache_data
def load_all_data(league):
    """
    league: 'NBA' or 'WNBA'. See module docstring for folder layout and the
    coach-data-optional behavior.
    """
    data_dir = os.path.join(BASE_DATA_DIR, league)
    team_stats_path = os.path.join(data_dir, 'team_season_stats.csv')

    if not os.path.exists(team_stats_path):
        return {'league_available': False}

    team_stats = pd.read_csv(team_stats_path)

    # Playoff results: merge the API-sourced pull with a manual supplement,
    # if one exists. Kept as two files on disk rather than combined into
    # one, so the split between "reproducible from the API" and "manually
    # sourced" stays visible in the repo structure - same provenance
    # principle the original BR/Stathead runbook used.
    playoff_api_path = os.path.join(data_dir, 'playoff_results.csv')
    playoff_results = pd.read_csv(playoff_api_path) if os.path.exists(playoff_api_path) else pd.DataFrame()

    playoff_manual_path = os.path.join(data_dir, 'playoff_results_manual.csv')
    if os.path.exists(playoff_manual_path):
        playoff_manual = pd.read_csv(playoff_manual_path)
        playoff_results = pd.concat([playoff_results, playoff_manual], ignore_index=True)
        # A Season/team_code should only ever come from ONE source. If the
        # API ever backfills a season the manual file also covers, fail
        # loudly rather than silently duplicating rows or picking one
        # arbitrarily - this is exactly the kind of gap that should get
        # caught in a verification pass, not discovered post-deployment.
        dupes = playoff_results.duplicated(subset=['Season', 'team_code'], keep=False)
        if dupes.any():
            overlap = playoff_results.loc[dupes, ['Season', 'team_code']].drop_duplicates().to_dict('records')
            raise ValueError(
                f"[{league}] playoff_results.csv and playoff_results_manual.csv both "
                f"cover the same Season/team_code: {overlap}. Remove the overlap from "
                f"one of the two files before loading."
            )

    player_stats = pd.read_csv(os.path.join(data_dir, 'player_season_stats.csv'))

    coach_season_wins_path = os.path.join(data_dir, 'coach_season_wins.csv')
    coach_season_wins = (
        pd.read_csv(coach_season_wins_path) if os.path.exists(coach_season_wins_path)
        else _empty_coach_season_wins()
    )

    coach_tenures_path = os.path.join(data_dir, 'coach_tenures.csv')
    coach_tenures = pd.read_csv(coach_tenures_path) if os.path.exists(coach_tenures_path) else None

    exec_tenures_path = os.path.join(data_dir, 'exec_tenures.csv')
    exec_tenures = pd.read_csv(exec_tenures_path) if os.path.exists(exec_tenures_path) else None

    coach_awards_path = os.path.join(data_dir, 'coach_awards.csv')
    coach_awards = pd.read_csv(coach_awards_path) if os.path.exists(coach_awards_path) else None

    exec_awards_path = os.path.join(data_dir, 'exec_awards.csv')
    exec_awards = pd.read_csv(exec_awards_path) if os.path.exists(exec_awards_path) else None

    return {
        'league_available': True,
        'team_stats': team_stats,
        'playoff_results': playoff_results,
        'player_stats': player_stats,
        'coach_season_wins': coach_season_wins,
        'coach_data_available': len(coach_season_wins) > 0,
        'coach_tenures': coach_tenures,
        'exec_tenures': exec_tenures,
        'coach_awards': coach_awards,
        'exec_awards': exec_awards,
    }


def get_team_season_options(team_stats):
    """Returns a sorted list of 'TEAM Season' strings for the dropdowns."""
    options = (team_stats['Team'] + ' ' + team_stats['Season']).sort_values().tolist()
    return options


def parse_team_season_option(option_str):
    """Reverses get_team_season_options formatting back into (Season, Team)."""
    team, season = option_str.split(' ', 1)
    return (season, team)
