"""
app.py

WNBA Team Comparison Tool - Streamlit app.
Data sourced exclusively from nba_api (stats.wnba.com) via the canonical
data pipeline in data/processed/WNBA/. No Basketball-Reference/Stathead
dependency (ToS-restricted; that data stays local-only for the NBA version).

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import load_league_data
from scoring import (
    build_full_scoring_table, compute_player_roster_shares, compare_two_teams,
    build_player_leaderboard, build_transparency_panel, apply_era_adjustment,
    build_year_over_year_leaderboard, percentile_of,
    get_team_display_options, get_season_options_for_team, get_prior_season_row,
    normalize_weights, rescore,
    WEIGHTS, ERA_BOUNDARIES, STAT_DEFINITIONS
)

LEAGUE = 'WNBA'

st.set_page_config(page_title="WNBA Team Jump Comparison", layout="wide")

COLOR_A = '#1f77b4'
COLOR_B = '#ff7f0e'

# Quartile color scale: green=top quartile, yellow=3rd, orange=2nd, red=bottom.
# Every color pairs background+text together explicitly (not just background),
# so contrast stays correct regardless of Streamlit's light/dark theme.
QUARTILE_COLORS = {
    'green': ('#2ecc71', 'white'),
    'yellow': ('#f1c40f', 'black'),
    'orange': ('#e67e22', 'white'),
    'red': ('#e74c3c', 'white'),
    'gray': ('#888888', 'white'),
}


def get_quartile_bucket(percentile):
    if percentile is None:
        return QUARTILE_COLORS['gray']  # missing data should look neutral, not falsely "top quartile"
    if percentile >= 75:
        return QUARTILE_COLORS['green']
    elif percentile >= 50:
        return QUARTILE_COLORS['yellow']
    elif percentile >= 25:
        return QUARTILE_COLORS['orange']
    else:
        return QUARTILE_COLORS['red']


def render_badge(text, bg_color, text_color):
    return f"<span style='background-color:{bg_color}; color:{text_color}; padding:2px 10px; border-radius:10px; font-weight:600; font-size:0.95em;'>{text}</span>"


def render_quartile_badge(value, percentile, fmt="{:.3f}"):
    bg, txt = get_quartile_bucket(percentile)
    return render_badge(fmt.format(value), bg, txt)


def format_pie_as_pct(df, col='pie'):
    """Returns a copy with the PIE column converted from a raw share
    (e.g. 0.082) to a display-ready percentage string (e.g. '8.2%'). PIE is
    a share of league-wide box-score production, so a percentage reads far
    more naturally than the raw decimal."""
    if col not in df.columns:
        return df
    df = df.copy()
    df[col] = df[col].apply(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "")
    return df


def render_delta_badge(current, prior, fmt="{:+.3f}", tolerance=1e-9):
    """Directional badge: green+up-arrow if higher, red+down-arrow if lower,
    yellow+no-arrow if unchanged. Returns a plain string (not HTML) if no
    prior value exists, so callers can display that as plain text."""
    if prior is None or pd.isna(prior):
        return None
    diff = current - prior
    if abs(diff) < tolerance:
        bg, txt, arrow = QUARTILE_COLORS['yellow'][0], QUARTILE_COLORS['yellow'][1], ''
    elif diff > 0:
        bg, txt, arrow = QUARTILE_COLORS['green'][0], QUARTILE_COLORS['green'][1], '▲ '
    else:
        bg, txt, arrow = QUARTILE_COLORS['red'][0], QUARTILE_COLORS['red'][1], '▼ '
    return render_badge(f"{arrow}{fmt.format(diff)}", bg, txt)


# ------------------------------------------------------------------
# CACHED LOAD + BASE SCORING TABLE (built once; weight changes are cheap)
# ------------------------------------------------------------------
@st.cache_data
def get_scored_data():
    team_stats, player_stats, playoff_results = load_league_data(LEAGUE)

    # MOV isn't a stored column - derive it once here (see handoff note).
    team_stats = team_stats.copy()
    team_stats['mov'] = (team_stats['pts'] - team_stats['opp_pts']) / team_stats['g']

    # Coaching is dormant for WNBA (no coach data sourced yet) - pass an
    # empty frame so compute_coaching_score degrades gracefully to a
    # neutral, non-differentiating value across the board.
    base_table = build_full_scoring_table(
        team_stats, playoff_results, player_stats,
        pd.DataFrame(), LEAGUE
    )
    player_shares = compute_player_roster_shares(player_stats)
    return team_stats, player_stats, playoff_results, base_table, player_shares


team_stats, player_stats, playoff_results, base_table, player_shares = get_scored_data()

# WNBA's first season (1997) is real and fully selectable - unlike the NBA
# app, there is no hidden "background" season to filter out here.
team_stats_selectable = team_stats
ws_selectable_base = player_shares

# Apply any pending team selection staged by a leaderboard "Set as Team A/B"
# button click, BEFORE the Compare tab's widgets instantiate this run. This
# must happen here, not inside the widget's own on-click handler - Streamlit
# refuses to modify a widget's session_state key after that widget has
# already rendered in the current script execution, even across a rerun.
for slot in ['a', 'b']:
    pending_key = f'pending_team_{slot}'
    if pending_key in st.session_state:
        pending = st.session_state.pop(pending_key)
        st.session_state[f'team_{slot}_select'] = pending['team']
        st.session_state[f'season_{slot}_select'] = pending['season']

# ------------------------------------------------------------------
# SIDEBAR: user-adjustable composite weights
# ------------------------------------------------------------------
DEFAULT_WEIGHTS_PCT = {'quality': 45, 'playoffs': 40, 'roster': 15}

st.sidebar.header("⚖️ Composite Weights")
st.sidebar.caption("Quality = Massey SRS  •  Playoffs = postseason depth  •  Roster = Σ(PIE × min)")
for k, v in DEFAULT_WEIGHTS_PCT.items():
    st.session_state.setdefault(f'w_{k}', v)
if st.sidebar.button('Reset to default (45 / 40 / 15)'):
    for k, v in DEFAULT_WEIGHTS_PCT.items():
        st.session_state[f'w_{k}'] = v
raw_weights = {
    k: st.sidebar.slider(k.capitalize(), 0, 100, step=5, key=f'w_{k}')
    for k in DEFAULT_WEIGHTS_PCT
}
norm_weights = normalize_weights(raw_weights)
st.sidebar.caption(
    f"In effect: Quality **{norm_weights['quality']:.0%}** / "
    f"Playoffs **{norm_weights['playoffs']:.0%}** / "
    f"Roster **{norm_weights['roster']:.0%}**"
)

# The base table (with weight-independent z-columns) is built once and
# cached above; only this cheap rescore runs per slider interaction, and it
# drives every view for the rest of the script.
scored = rescore(base_table, raw_weights)
scored_selectable = scored

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("🏀 WNBA Team Jump Comparison Tool")
st.caption(
    "Compare any two team-seasons (1997 to present) across quality, roster strength, "
    "and playoff performance.  \n"
    "**Data sourced exclusively from [nba_api](https://github.com/swar/nba_api) "
    "(stats.wnba.com) — no Basketball-Reference/Stathead dependency.**"
)

tab_compare, tab_leaderboards, tab_methodology = st.tabs(["⚖️ Compare Teams", "🏆 Leaderboards", "📖 Methodology"])


# ------------------------------------------------------------------
# SHARED HELPER: box-and-whisker subplot grid
# ------------------------------------------------------------------
def get_landmark_points(df, col):
    """
    Identifies which real team-season achieves each key statistic (min,
    25th/50th/75th percentile, max) for a population - so a box plot's
    hover can answer 'which team-year IS that?' instead of just showing a
    bare number. Min/Max are exact matches; the quantile points use the
    closest actual team-season to that statistic (quantiles are often
    interpolated and may not land exactly on any single real observation),
    labeled as such rather than implied to be an exact match.
    """
    valid = df.assign(_val=df[col]).dropna(subset=['_val'])
    if len(valid) == 0:
        return []
    stats = [
        ('Min', valid['_val'].min(), True),
        ('25th %ile', valid['_val'].quantile(0.25), False),
        ('Median', valid['_val'].quantile(0.5), False),
        ('75th %ile', valid['_val'].quantile(0.75), False),
        ('Max', valid['_val'].max(), True),
    ]
    points = []
    for stat_label, stat_val, is_exact in stats:
        closest_idx = (valid['_val'] - stat_val).abs().idxmin()
        closest_row = valid.loc[closest_idx]
        team_year = f"{closest_row['team_code']} {closest_row['season']}"
        hover = f"{stat_label}: {stat_val:.2f}<br>{team_year}" if is_exact else f"{stat_label}: {stat_val:.2f}<br>Closest: {team_year}"
        points.append({'value': stat_val, 'hover': hover})
    return points


def render_box_whisker_section(metrics, row_a, row_b, label_a, label_b, population_mode, key_prefix):
    """
    metrics: list of (column_name, display_label) tuples - every column is a
             real column on `scored` (no computed-on-the-fly special case;
             net_rating is now a direct column like everything else).
    population_mode: "All-time (pooled)" or "Each team's own era"
    Renders an N-column subplot grid (up to 4 per row) with population box
    plots, each team's value marked as a distinct point, and invisible
    landmark markers at the min/quartile/median/max positions so hovering
    over those points on the box identifies which real team-season they are.
    """
    n = len(metrics)
    cols_per_row = min(n, 4)
    rows_needed = -(-n // cols_per_row)  # ceil

    fig = make_subplots(rows=rows_needed, cols=cols_per_row, subplot_titles=[label for _, label in metrics])
    same_era = row_a['era'] == row_b['era']

    def add_landmark_markers(fig, pop_df, col, pop_label, r, c):
        points = get_landmark_points(pop_df, col)
        if points:
            fig.add_trace(go.Scatter(
                x=[pop_label] * len(points), y=[p['value'] for p in points],
                mode='markers', marker=dict(size=6, color='gray', opacity=0.35),
                hovertext=[p['hover'] for p in points], hoverinfo='text',
                showlegend=False
            ), row=r, col=c)

    for idx, (col, label) in enumerate(metrics):
        r, c = (idx // cols_per_row) + 1, (idx % cols_per_row) + 1
        val_a, val_b = row_a[col], row_b[col]

        if population_mode == "All-time (pooled)" or same_era:
            pop_label = "All-time" if population_mode == "All-time (pooled)" else row_a['era']
            pop_df = scored if population_mode == "All-time (pooled)" else scored[scored['era'] == row_a['era']]
            population = pop_df[col]
            fig.add_trace(go.Box(y=population, name=pop_label, boxpoints=False, marker_color='lightgray',
                                   showlegend=False, hoverinfo='skip'), row=r, col=c)
            add_landmark_markers(fig, pop_df, col, pop_label, r, c)
            fig.add_trace(go.Scatter(x=[pop_label], y=[val_a], mode='markers', name=label_a,
                                       marker=dict(size=12, symbol='circle', color=COLOR_A),
                                       hovertext=f"{label_a}: {val_a:.2f}", hoverinfo='text',
                                       showlegend=(idx == 0), legendgroup='a'), row=r, col=c)
            fig.add_trace(go.Scatter(x=[pop_label], y=[val_b], mode='markers', name=label_b,
                                       marker=dict(size=12, symbol='diamond', color=COLOR_B),
                                       hovertext=f"{label_b}: {val_b:.2f}", hoverinfo='text',
                                       showlegend=(idx == 0), legendgroup='b'), row=r, col=c)
        else:
            pop_a_df = scored[scored['era'] == row_a['era']]
            pop_b_df = scored[scored['era'] == row_b['era']]
            fig.add_trace(go.Box(y=pop_a_df[col], name=row_a['era'], boxpoints=False, marker_color='lightgray', showlegend=False, hoverinfo='skip'), row=r, col=c)
            fig.add_trace(go.Box(y=pop_b_df[col], name=row_b['era'], boxpoints=False, marker_color='lightgray', showlegend=False, hoverinfo='skip'), row=r, col=c)
            add_landmark_markers(fig, pop_a_df, col, row_a['era'], r, c)
            add_landmark_markers(fig, pop_b_df, col, row_b['era'], r, c)
            fig.add_trace(go.Scatter(x=[row_a['era']], y=[val_a], mode='markers', name=label_a,
                                       marker=dict(size=12, symbol='circle', color=COLOR_A),
                                       hovertext=f"{label_a}: {val_a:.2f}", hoverinfo='text',
                                       showlegend=(idx == 0), legendgroup='a'), row=r, col=c)
            fig.add_trace(go.Scatter(x=[row_b['era']], y=[val_b], mode='markers', name=label_b,
                                       marker=dict(size=12, symbol='diamond', color=COLOR_B),
                                       hovertext=f"{label_b}: {val_b:.2f}", hoverinfo='text',
                                       showlegend=(idx == 0), legendgroup='b'), row=r, col=c)

    fig.update_layout(height=340 * rows_needed, showlegend=True, margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")
    st.caption("Hover over the faint gray dots on each box (min/25th/median/75th/max) to see which team-season that landmark value belongs to.")
    if population_mode == "Each team's own era" and not same_era:
        st.caption(f"{label_a} plotted against {row_a['era']}; {label_b} plotted against {row_b['era']} - different eras shown side by side rather than pooled.")

    with st.expander("What do these mean?"):
        for col, label in metrics:
            definition = STAT_DEFINITIONS.get(label, "No definition available.")
            st.markdown(f"**{label}**: {definition}")


def get_exact_season_label(scored_table, team, season):
    """Returns the properly formatted season dropdown label (including the
    trophy prefix if that team won the championship that year), so a
    programmatic selection matches an option that actually exists in the
    dropdown's list - Streamlit errors if a widget's session_state value
    isn't a member of its options."""
    for label, s in get_season_options_for_team(scored_table, team):
        if s == season:
            return label
    return season


def render_send_to_compare_table(df, key_prefix):
    """Renders 'Set as Team A/B' buttons ABOVE a row-selectable dataframe
    (using a placeholder reserved before the dataframe renders, then filled
    in afterward once we know the selection - Streamlit's selection event
    only exists after the dataframe itself has rendered, but we still want
    the buttons visually on top for clarity about what the row-selector
    checkbox is for).

    Button clicks stage the selection into a 'pending_team_a/b' key (NOT
    the widget's own key directly - see the pending-selection handling near
    the top of the script for why) and rerun; this does NOT switch the
    active tab automatically (Streamlit has no API for that), so the user
    needs to manually click back to 'Compare Teams' afterward."""
    button_area = st.empty()
    event = st.dataframe(df, hide_index=True, use_container_width=True, height=550,
                          on_select="rerun", selection_mode="single-row", key=f"{key_prefix}_select_df")
    selected_rows = event.selection.rows if event and event.selection else []

    with button_area.container():
        if selected_rows:
            selected = df.iloc[selected_rows[0]]
            sel_season, sel_team = selected['Season'], selected['Team']
            team_code_to_label = {code: label for label, code in get_team_display_options(team_stats_selectable)}
            team_label = team_code_to_label.get(sel_team, sel_team)
            season_label = get_exact_season_label(scored_selectable, sel_team, sel_season)
            btn_cols = st.columns(2)
            with btn_cols[0]:
                if st.button("Set as Team A", key=f"{key_prefix}_set_a"):
                    st.session_state['pending_team_a'] = {'team': team_label, 'season': season_label}
                    st.rerun()
            with btn_cols[1]:
                if st.button("Set as Team B", key=f"{key_prefix}_set_b"):
                    st.session_state['pending_team_b'] = {'team': team_label, 'season': season_label}
                    st.rerun()
        else:
            st.caption("☑️ Select a row below (checkbox on the left of each row), then use the buttons that appear here to send it to Team A or Team B.")


# ====================================================================
# TAB 1: HEAD-TO-HEAD COMPARISON
# ====================================================================
with tab_compare:
    team_display_options = get_team_display_options(team_stats_selectable)
    team_labels = [label for label, code in team_display_options]
    team_code_lookup = dict(team_display_options)

    col_a, col_b = st.columns(2)
    with col_a:
        default_team_a = next((label for label, code in team_display_options if code == 'LVA'), team_labels[0])
        team_label_a = st.selectbox("Team A", team_labels, index=team_labels.index(default_team_a), key="team_a_select")
        team_a = team_code_lookup[team_label_a]
        season_options_a = get_season_options_for_team(scored_selectable, team_a)
        season_labels_a = [label for label, season in season_options_a]
        season_lookup_a = dict(season_options_a)
        season_label_a = st.selectbox("Season A", season_labels_a, index=0, key="season_a_select")
        season_a = season_lookup_a[season_label_a]
    with col_b:
        default_team_b = next((label for label, code in team_display_options if code == 'NYL'), team_labels[1] if len(team_labels) > 1 else team_labels[0])
        team_label_b = st.selectbox("Team B", team_labels, index=team_labels.index(default_team_b), key="team_b_select")
        team_b = team_code_lookup[team_label_b]
        season_options_b = get_season_options_for_team(scored_selectable, team_b)
        season_labels_b = [label for label, season in season_options_b]
        season_lookup_b = dict(season_options_b)
        season_label_b = st.selectbox("Season B", season_labels_b, index=0, key="season_b_select")
        season_b = season_lookup_b[season_label_b]

    try:
        comparison = compare_two_teams(scored, player_shares, (season_a, team_a), (season_b, team_b))
    except ValueError as e:
        st.error(str(e))
        st.stop()

    row_a, row_b = comparison['team_a'], comparison['team_b']
    label_a, label_b = f"{team_a} {season_a}", f"{team_b} {season_b}"

    # --- Era adjustment: HIDDEN for WNBA (single placeholder era; quantile
    # mapping is an identity transform until the PELT changepoint study is
    # done). The call-site path is kept intact but permanently unreachable
    # here, so real era boundaries can be switched on later with no rework.
    era_adjust_on = False
    base_era = ERA_BOUNDARIES[LEAGUE][0][2]
    era_result = apply_era_adjustment(scored, row_a, row_b, base_era) if era_adjust_on else None

    st.divider()

    # --- Team headers, symmetric layout, with quartile-colored win% and prior-season delta ---
    win_pct_population = scored['w_pct']

    def render_record_block(row, season, team):
        current_pctile = percentile_of(row['w_pct'], win_pct_population)
        st.metric("Record", f"{int(row['w'])}-{int(row['l'])}", help="Wins-Losses")
        st.markdown(f"Win% {render_quartile_badge(row['w_pct'], current_pctile)}", unsafe_allow_html=True)

        prior_row = get_prior_season_row(scored, season, team)
        if prior_row is not None:
            prior_pctile = percentile_of(prior_row['w_pct'], win_pct_population)
            st.caption(f"Prior season ({prior_row['team_code']} {prior_row['season']}): {int(prior_row['w'])}-{int(prior_row['l'])}")
            st.markdown(f"<span style='font-size:0.85em;'>Prior Win% {render_quartile_badge(prior_row['w_pct'], prior_pctile)}</span>", unsafe_allow_html=True)
            delta_badge = render_delta_badge(row['w_pct'], prior_row['w_pct'])
            st.markdown(f"Change: {delta_badge}", unsafe_allow_html=True)
        else:
            st.caption("No prior-season record available for comparison.")

    head_a, head_vs, head_b = st.columns([5, 1, 5])
    with head_a:
        st.subheader(label_a)
        st.caption(f"Era: {row_a['era']}")
        render_record_block(row_a, season_a, team_a)
    with head_vs:
        st.markdown("<h2 style='text-align:center; margin-top:2em;'>VS</h2>", unsafe_allow_html=True)
    with head_b:
        st.subheader(label_b)
        st.caption(f"Era: {row_b['era']}")
        render_record_block(row_b, season_b, team_b)

    st.divider()

    # --- COMPOSITE SCORE: prominent, quartile-colored percentile, prior-score comparison ---
    st.subheader("Composite Score & Rankings")
    winner_is_a = row_a['composite_score'] > row_b['composite_score']
    score_a, score_b = row_a['composite_score'], row_b['composite_score']

    winner_label = label_a if winner_is_a else label_b
    st.success(f"🏆 **Overall Composite Winner: {winner_label}**")

    score_cols = st.columns(2)
    for col, (label, row, score, season, team) in zip(score_cols, [(label_a, row_a, score_a, season_a, team_a), (label_b, row_b, score_b, season_b, team_b)]):
        with col:
            st.markdown(f"**{label} — Composite Score**")
            st.markdown(f"<span style='font-size:1.6em; font-weight:bold;'>{score:.3f}</span>", unsafe_allow_html=True)
            pctile = percentile_of(score, scored['composite_score'])
            st.markdown(render_quartile_badge(score, pctile), unsafe_allow_html=True)

            prior_row = get_prior_season_row(scored, season, team)
            if prior_row is not None:
                prior_pctile = percentile_of(prior_row['composite_score'], scored['composite_score'])
                st.markdown(
                    f"Prior ({prior_row['season']}): <span style='font-size:0.85em;'>{prior_row['composite_score']:.3f} "
                    f"{render_quartile_badge(prior_row['composite_score'], prior_pctile)}</span>",
                    unsafe_allow_html=True
                )
                delta_badge = render_delta_badge(score, prior_row['composite_score'])
                st.markdown(f"Change: {delta_badge}", unsafe_allow_html=True)
            else:
                st.caption("No prior-season composite score available (first season in our data, or an expansion team).")

            rank_cols = st.columns(3)
            with rank_cols[0]:
                st.metric("All-Time Rank", f"{row['all_time_rank']}/{row['all_time_total']}")
            with rank_cols[1]:
                st.metric("Era Rank", f"{row['era_rank']}/{row['era_total']}")
            with rank_cols[2]:
                st.metric("Season Rank", f"{row['season_rank']}/{row['season_total']}")

    st.divider()

    # --- SCORED CATEGORIES: box-and-whisker (Quality + Roster; Playoffs shown separately) ---
    st.subheader("Scored Categories")
    scored_pop_mode = st.radio("Population", ["All-time (pooled)", "Each team's own era"], horizontal=True, key="scored_pop_mode")

    scored_metrics = [('srs', 'SRS'), ('roster_strength', 'Roster')]
    render_box_whisker_section(scored_metrics, row_a, row_b, label_a, label_b, scored_pop_mode, key_prefix="scored")

    # Playoffs shown separately - a 6-point ordinal doesn't box-plot meaningfully
    st.markdown("**Playoffs**")
    pcol1, pcol2 = st.columns(2)

    def playoff_display(row):
        if row['playoff_status'] == 'missed':
            return 'Missed'
        elif row['playoff_status'] == 'ambiguous':
            return 'Unresolved (data gap)'
        return row['playoff_round_reached']

    with pcol1:
        marker = "🏆 " if comparison['edges']['Playoffs'] == 'A' else ""
        st.write(f"{marker}{label_a}: **{playoff_display(row_a)}**")
    with pcol2:
        marker = "🏆 " if comparison['edges']['Playoffs'] == 'B' else ""
        st.write(f"{marker}{label_b}: **{playoff_display(row_b)}**")
    st.caption(STAT_DEFINITIONS['Playoffs'])

    st.divider()

    # --- FOUR FACTORS: box-and-whisker (display-only, not part of the composite) ---
    st.subheader("Four Factors")
    st.caption("Click a legend entry to show/hide that trace - this is Plotly's built-in interactive legend, not a separate app control.")
    ff_pop_mode = st.radio("Population for Four Factors", ["All-time (pooled)", "Each team's own era"], horizontal=True, key="ff_pop_mode")
    four_factors = [('efg_pct', 'eFG%'), ('tov_pct', 'TOV%'), ('orb_pct', 'ORB%'), ('ft_rate', 'FTr')]
    render_box_whisker_section(four_factors, row_a, row_b, label_a, label_b, ff_pop_mode, key_prefix="fourfactors")

    st.divider()

    # --- ADDITIONAL ADVANCED METRICS: box-and-whisker (context, not scored) ---
    st.subheader("Additional Advanced Metrics")
    st.caption("Context beyond the scored categories - not part of the composite score.")
    adv_pop_mode = st.radio("Population for Additional Metrics", ["All-time (pooled)", "Each team's own era"], horizontal=True, key="adv_pop_mode")
    additional_metrics = [
        ('pace', 'Pace'), ('off_rating', 'ORtg'), ('def_rating', 'DRtg'),
        ('ts_pct', 'TS%'), ('net_rating', 'Net Rating'), ('mov', 'MOV'), ('sos', 'SOS'),
    ]
    render_box_whisker_section(additional_metrics, row_a, row_b, label_a, label_b, adv_pop_mode, key_prefix="advanced")

    st.divider()

    # --- What Changed ---
    WHATS_CHANGED_MIN_MINUTES = 300
    st.subheader("What Changed (vs. the prior season)")
    st.caption(
        f"Additions/losses (by minutes), filtered to players with {WHATS_CHANGED_MIN_MINUTES}+ minutes "
        "on the relevant side of the comparison, so end-of-bench/garbage-time roster churn doesn't crowd "
        "out the moves that actually mattered. WS/PIE/USG% shown alongside to help identify not just WHO "
        "changed but WHY it mattered - WS for win value, PIE for outcome-agnostic footprint, USG% for "
        "role/impact size."
    )
    change_cols = st.columns(2)
    for col, (season, team, label) in zip(change_cols, [(season_a, team_a, label_a), (season_b, team_b, label_b)]):
        with col:
            st.markdown(f"**{label}**")
            panel = build_transparency_panel(scored, player_stats, pd.DataFrame(), season, team,
                                               coach_tenures=None, exec_tenures=None,
                                               min_minutes=WHATS_CHANGED_MIN_MINUTES)
            if not panel['has_prior_data']:
                st.info("No prior-season data available - likely an expansion franchise's first season.")
                continue

            prior_label = f"{panel['prior_team']} {panel['prior_season']}"
            st.caption(f"Compared to {prior_label}" + (" (bridged relocation/rename)" if panel['is_relocation_year'] else ""))

            if not panel.get('player_data_available_for_prior', True):
                st.caption("ℹ️ No player-level data exists yet for the prior season.")
            else:
                display_cols = {'player_name': 'Player', 'min': 'Min', 'ws': 'WS', 'pie': 'PIE', 'usg_pct': 'USG%'}
                st.markdown("**Key additions**")
                if len(panel['players_added']):
                    st.dataframe(format_pie_as_pct(panel['players_added']).rename(columns=display_cols), hide_index=True, use_container_width=True)
                else:
                    st.write("_None_")
                st.markdown("**Key losses**")
                if len(panel['players_lost']):
                    st.dataframe(format_pie_as_pct(panel['players_lost']).rename(columns=display_cols), hide_index=True, use_container_width=True)
                else:
                    st.write("_None_")

    with st.expander("What do these mean?"):
        for stat in ['MP', 'WS', 'PIE', 'USG%']:
            st.markdown(f"**{stat}**: {STAT_DEFINITIONS.get(stat, 'No definition available.')}")

    st.divider()

    # --- Roster Contribution ---
    st.subheader("Roster Contribution (PIE × minutes, and Win Shares → % of team wins)")
    st.caption(
        "Shows every rotation player (300+ minutes) on the roster, sorted by Win Shares. "
        "PIE and WS surface side by side - footprint (PIE) vs. win-translation (WS); the gap between "
        "the two is itself informative. A high-minutes, high-usage player can still land near the bottom "
        "of this list with negative WS - that's Win Shares correctly penalizing low-efficiency volume, "
        "not a data gap, and it's exactly the kind of disagreement PIE vs. WS is meant to surface. "
        "Blank '% of team wins' means that row is a multi-team combined season (a player traded mid-year) "
        "- the metric isn't meaningful for a blended team label, so it's left blank rather than shown as "
        "a misleading number."
    )
    roster_cols_list = ['player_name', 'min', 'ws', 'pie', 'pct_of_team_ws']
    roster_display_names = {'player_name': 'Player', 'min': 'Min', 'ws': 'WS', 'pie': 'PIE', 'pct_of_team_ws': '% of team wins'}
    roster_cols = st.columns(2)
    with roster_cols[0]:
        st.markdown(f"**{label_a}**")
        cols_present_a = [c for c in roster_cols_list if c in comparison['roster_a'].columns]
        st.dataframe(format_pie_as_pct(comparison['roster_a'][cols_present_a]).rename(columns=roster_display_names), hide_index=True, use_container_width=True)
    with roster_cols[1]:
        st.markdown(f"**{label_b}**")
        cols_present_b = [c for c in roster_cols_list if c in comparison['roster_b'].columns]
        st.dataframe(format_pie_as_pct(comparison['roster_b'][cols_present_b]).rename(columns=roster_display_names), hide_index=True, use_container_width=True)
    with st.expander("What do these mean?"):
        for stat in ['WS', 'PIE', '% of team wins']:
            st.markdown(f"**{stat}**: {STAT_DEFINITIONS.get(stat, 'No definition available.')}")

    # NOTE: Coach & Executive panel is DORMANT for WNBA (no coach/exec data
    # sourced yet) - intentionally not rendered. Flagged as separate future
    # work (see Methodology tab).


# ====================================================================
# TAB 2: LEADERBOARDS
# ====================================================================
with tab_leaderboards:
    lb_team, lb_yoy, lb_player, lb_coach_exec = st.tabs(["Teams", "Year-over-Year Changes", "Players", "Coaches & Executives"])

    with lb_team:
        st.subheader("All-Time Team Leaderboard")
        season_filter = st.selectbox("Filter to a single season (optional)", ["All seasons"] + sorted(team_stats_selectable['season'].unique(), reverse=True), key="team_lb_filter")

        display_df = scored_selectable[['all_time_rank', 'season_rank', 'season', 'era', 'team_code', 'w', 'l', 'composite_score',
                              'srs', 'playoff_status', 'playoff_round_reached']].copy()
        display_df['Playoff Result'] = display_df.apply(
            lambda r: 'Missed' if r['playoff_status'] == 'missed'
            else ('Unresolved (data gap)' if r['playoff_status'] == 'ambiguous' else r['playoff_round_reached']),
            axis=1
        )
        display_df = display_df.drop(columns=['playoff_status', 'playoff_round_reached'])
        display_df = display_df.rename(columns={
            'srs': 'SRS', 'composite_score': 'Composite Score',
            'all_time_rank': 'All-Time Rank', 'season_rank': 'Season Rank', 'era': 'Era',
            'season': 'Season', 'team_code': 'Team', 'w': 'W', 'l': 'L',
        })
        display_df['Composite Score'] = display_df['Composite Score'].round(3)
        if season_filter != "All seasons":
            display_df = display_df[display_df['Season'] == season_filter]
        display_df = display_df.sort_values('All-Time Rank').reset_index(drop=True)
        render_send_to_compare_table(display_df, key_prefix="team_lb")

    with lb_yoy:
        st.subheader("Year-over-Year Changes")
        st.caption(
            "Every team-season compared to its immediately prior season (bridging relocations/renames). "
            "Sort by any column to find the biggest jumps or collapses - this is the direct analytical "
            "payoff of the project's original 'which teams made a big jump' premise."
        )
        yoy_lb = build_year_over_year_leaderboard(scored)
        yoy_lb = yoy_lb.rename(columns={
            'season': 'Season', 'team_code': 'Team', 'prior_season': 'Prior Season',
            'composite_change': 'Composite Score Change', 'win_change': 'Win Change',
            'quality_change': 'Quality (SRS) Change', 'net_rating_change': 'Net Rating Change',
            'pace_change': 'Pace Change', 'ts_pct_change': 'TS% Change', 'roster_change': 'Roster Change',
        })
        direction = st.radio("Show", ["Biggest jumps first", "Biggest collapses first"], horizontal=True, key="yoy_direction")
        yoy_display = yoy_lb.sort_values('Composite Score Change', ascending=(direction == "Biggest collapses first")).reset_index(drop=True)
        render_send_to_compare_table(yoy_display, key_prefix="yoy_lb")

    with lb_player:
        st.subheader("Player-Season Leaderboard")
        st.caption("Ranked by Win Shares. '% of team' columns are each player's renormalized share of their team's positive-impact pool (PIE-based footprint and WS-based win-translation, shown side by side).")
        player_season_filter = st.selectbox("Filter to a single season", ["All seasons"] + sorted(scored_selectable['season'].unique(), reverse=True), key="player_lb_filter")
        player_lb = build_player_leaderboard(player_shares, player_season_filter)
        player_lb = format_pie_as_pct(player_lb)
        player_lb = player_lb.rename(columns={
            'player_name': 'Player', 'season': 'Season', 'team_code': 'Team', 'age': 'Age', 'min': 'Min',
            'ws': 'WS', 'ows': 'OWS', 'dws': 'DWS', 'pie': 'PIE', 'player_impact': 'Impact (PIE×Min)',
            'pct_of_team': '% of team (PIE)', 'pct_of_team_ws': '% of team (WS)',
        })
        st.dataframe(player_lb, hide_index=True, use_container_width=True, height=550)

    with lb_coach_exec:
        st.subheader("Coaches & Executives")
        st.info(
            "🚧 Not yet sourced for the WNBA. Coach and executive data is flagged as a "
            "separate future work chunk (a distinct mode of manual lookup and entry, "
            "unlike the automated nba_api pipeline that powers everything else in this "
            "app) - this tab will populate once that data exists, with no other changes "
            "needed elsewhere in the app."
        )


# ====================================================================
# TAB 3: METHODOLOGY
# ====================================================================
with tab_methodology:
    st.header("How the scoring works")

    st.markdown(f"""
    The composite score blends **three categories**, each standardized (z-scored) across all
    **{len(scored)} team-seasons** (1997 through present) before weighting, so a team's performance
    is measured relative to the full WNBA population rather than on raw units. Every season is
    user-selectable - there's no hidden "background" season the way the NBA version needs one for
    prior-season bridging at its 1979-80 data floor, since 1997 is genuinely the WNBA's first season.
    """)

    st.subheader("Category weights")
    st.caption("Defaults shown below. Every value is user-adjustable via the sidebar sliders, which recompute the composite live.")
    weight_df = pd.DataFrame([
        {"Category": "Quality (Massey SRS)", "Default Weight": f"{WEIGHTS['quality']:.0%}",
         "Why": "Schedule-adjusted point differential, solved simultaneously over the whole season's game graph. Best single 'how good were they' number."},
        {"Category": "Playoffs", "Default Weight": f"{WEIGHTS['playoffs']:.0%}",
         "Why": "Depth-honest ordinal scale (0=missed, 5=champion). The one genuinely semi-independent axis (~0.79 correlated with Quality), so it carries real weight."},
        {"Category": "Roster (Σ PIE × min)", "Default Weight": f"{WEIGHTS['roster']:.0%}",
         "Why": "Personnel-strength signal. Kept light (~0.85 correlated with Quality) since it's largely re-stating the same underlying team strength."},
    ])
    st.table(weight_df)

    st.subheader("Why only three axes (signal independence, not coverage)")
    st.markdown("""
    An earlier draft carried SRS and Net Rating as two separate weighted categories. Empirically,
    `corr(srs, net_rating) = 0.992` - scoring both was counting the same signal twice under different
    names, at half the composite. They're collapsed into one **Quality** axis (Massey SRS, with a
    `net_rating` fallback for the rare team-season without a game-log-derived rating).

    **Coaching was removed from the composite entirely** - two independent reasons, not one:
    a prior-record coaching score is itself correlated with team quality (re-importing the same
    signal a third time), and an award-based component necessarily scores a team the season
    *after* the recognition was earned (a lag artifact). There's also no WNBA coach data sourced
    yet regardless (see the Leaderboards tab), so coaching is fully dormant rather than
    partially-scored.

    **Roster strength and win percentage were both evaluated and ruled out** as separate scored
    categories - both cross-correlate 0.83-1.00 with `net_rating`, meaning they re-import the
    Quality signal under a different name rather than adding independent information. Roster
    survives as a *light-weight* category specifically because, even at 0.85 correlated, it's
    still meaningfully less redundant than the other candidates were.
    """)

    st.subheader("SRS (Massey rating)")
    st.markdown("""
    Solved via points-based Massey least-squares over every regular-season team-game: each team's
    offense and defense ratings are fit simultaneously across the whole schedule graph, so the
    result is schedule-adjusted by construction (no separate SOS bolt-on needed for the rating
    itself). Blowout margins are capped at the 95th percentile (26 points) before fitting, so a
    handful of lopsided games don't dominate the L2 loss - `srs`, `srs_off`, and `srs_def` all live
    on this capped-margin scale.

    `sos` is the games-weighted mean of a team's opponents' own SRS ratings - deliberately **not**
    `srs - mov`, which would mix capped-rating units against raw-margin units (a unit error, even
    though the two are numerically close).

    Validated: `corr(srs, net_rating) = 0.992` league-wide, and margin-treatment choice (raw vs.
    cap vs. tanh) barely moves team rankings (Spearman ρ = 0.999 between raw and capped).
    """)

    st.subheader("Win Shares")
    st.markdown("""
    Rebuilt in-house via Dean Oliver's method, straight from the box score - no
    Basketball-Reference dependency (`nba_api` doesn't carry pre-built Win Shares). Validated to
    sum to roughly the team's actual win total (~1.008× across all team-seasons). PIE is also
    surfaced as a second, outcome-agnostic footprint metric; the two contribution shares
    (footprint via PIE, win-translation via WS) are shown side by side wherever a roster breaks
    down - the *gap* between them is itself a signal worth reading, not noise to average away.
    """)

    st.subheader("Era adjustment")
    st.markdown("""
    The mechanism exists (quantile-mapping a team's Quality/Roster values onto an equivalent
    percentile in a different era's distribution) but the toggle is **hidden** for the WNBA: real
    era boundaries haven't been validated yet via the offline PELT changepoint study the NBA
    version uses, so every WNBA season currently buckets into a single placeholder era
    (`{}`). Turning quantile-mapping on against a single-era population would be
    an identity transform anyway - nothing to show. The underlying code path is intact and
    unreachable rather than removed, so it activates with no rework once that analysis is done
    (flagged as a separate future session).
    """.format(ERA_BOUNDARIES[LEAGUE][0][2]))

    st.subheader("Box-and-whisker population toggle")
    st.markdown("""
    Every box-and-whisker section (Scored Categories, Four Factors, Additional Advanced Metrics)
    offers a choice between plotting both teams' dots against the **all-time pooled** distribution
    or **each team's own era**. With only one placeholder era currently defined, these two modes
    are equivalent for now - the toggle is kept live so it starts doing real work the moment
    genuine era boundaries exist.
    """)

    st.subheader("Quartile color coding")
    st.markdown("""
    Win% and composite-score badges are colored by their **all-time percentile**: green = top quartile
    (75th+), yellow = 3rd quartile (50th-75th), orange = 2nd quartile (25th-50th), red = bottom quartile
    (below 25th). Change-from-prior-season badges use a different, directional scheme: green + ▲ for an
    improvement, red + ▼ for a decline, yellow with no arrow for no change.
    """)

    st.subheader("What's NOT scored (display-only)")
    st.markdown("""
    - **Shooting splits, Four Factors, Pace, and other advanced metrics** - Quality already captures
      the efficiency signal these feed into; scoring them separately would double-count.
    - **Net Rating** - shown in Additional Advanced Metrics as context; it's the fallback input to
      Quality when SRS is unavailable, not a separate scored slot.
    - **Coaching / Executives** - not sourced for the WNBA yet; see the Leaderboards tab.
    """)

    st.subheader("Known limitations, stated plainly")
    st.markdown("""
    - **WNBA eras are pending** - the PELT changepoint study that produced the NBA's six eras
      hasn't been run for the WNBA yet. Until it is, era-based rankings and the era-adjustment
      toggle are effectively no-ops.
    - **PER / BPM / VORP are unavailable** - `nba_api` doesn't carry these for the WNBA, so they're
      not referenced anywhere in this app (they were part of the old Basketball-Reference-sourced
      NBA model).
    - **Multi-team (traded) players' Win Shares are approximate** - computed against their listed
      team's context only; a blended multi-team box-score line isn't a fully coherent
      single-team season.
    - **1997-2000 WNBA playoff results have a confirmed data gap** in the underlying API (not a
      pipeline bug) - a manual backfill from an independent source is planned but not yet merged.
    - **Coach and executive data is not yet sourced** for the WNBA at all - a distinct, manual data
      mode from the rest of the automated pipeline, tracked as separate future work.
    """)

    st.divider()
    st.header("Future Enhancements")
    st.markdown("""
    This is a working v1 of the WNBA migration, not a finished product. Concrete next steps:

    - **WNBA era boundaries**: run the offline PELT changepoint study (as already done for the NBA)
      and swap real boundaries into `ERA_BOUNDARIES['WNBA']` - nothing else in the scoring or app
      code needs to change once that lookup table is populated.
    - **WNBA-specific Bayesian shrinkage calibration**: the coaching pseudo-games constant (and any
      future shrinkage constants) should be recalibrated for the WNBA's much shorter ~29-season
      history rather than reusing NBA-tuned defaults.
    - **1997-2000 playoff backfill**: a manually sourced, documented supplement CSV, merged at load
      time with a loud error on any season/team overlap against the primary pipeline.
    - **Coach and executive data sourcing**: a genuinely separate work chunk (manual lookup/entry,
      not an `nba_api` pull) - once it exists, the dormant coaching/exec code paths in `scoring.py`
      activate with minimal rework, exactly as designed.
    - **Fitting the composite weights rather than hand-setting them**: the current Quality/Playoffs/
      Roster split (45/40/15) is a reasonable, empirically-motivated starting point (see the
      signal-independence analysis above), not the output of a rigorous fitting process against an
      external target.
    """)
