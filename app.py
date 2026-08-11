"""
app.py

NBA Team Comparison Tool - Streamlit prototype.
Data sourced from Basketball-Reference.com and Stathead.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import load_all_data
from scoring import (
    build_full_scoring_table, compute_player_win_shares, compare_two_teams,
    build_exec_season_wins, build_coach_leaderboard, build_exec_leaderboard,
    build_player_leaderboard, build_transparency_panel, apply_era_adjustment,
    build_multi_coach_season_detail, build_year_over_year_leaderboard, percentile_of,
    get_team_display_options, get_season_options_for_team, get_prior_season_row,
    WEIGHTS, ERA_BOUNDARIES, DATA_CUTOFF_SEASON, STAT_DEFINITIONS
)

st.set_page_config(page_title="NBA Team Jump Comparison", layout="wide")

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


@st.cache_data
def get_scored_data(league):
    data = load_all_data(league)
    if not data['league_available']:
        return data, None, None, None
    exec_season_wins = build_exec_season_wins(data['exec_tenures'], data['team_stats']) if data['exec_tenures'] is not None else pd.DataFrame()
    scored = build_full_scoring_table(
        data['team_stats'], data['playoff_results'], data['player_stats'],
        data['coach_season_wins'], league, data['coach_tenures'], data['coach_awards'],
        exec_season_wins, data['exec_awards']
    )
    ws = compute_player_win_shares(data['player_stats'], data['team_stats'])
    return data, scored, ws, exec_season_wins


# ------------------------------------------------------------------
# LEAGUE SELECTION
# A league whose data/processed/{league}/ folder hasn't been populated yet
# (pipeline not run for it) shows a coming-soon state rather than crashing -
# see data_loader.load_all_data's 'league_available' flag.
# ------------------------------------------------------------------
st.sidebar.header("League")
league = st.sidebar.radio("League", ["WNBA", "NBA"], index=0, key="league_select")

data, scored, ws, exec_season_wins = get_scored_data(league)

if not data['league_available']:
    st.title("🏀 Team Jump Comparison Tool")
    st.warning(
        f"**{league} data isn't loaded yet.** This app expects "
        f"`data/processed/{league}/team_season_stats.csv` and hasn't found it. "
        f"Run the data pipeline for {league} (or, for NBA, move the existing "
        f"Basketball-Reference/Stathead CSVs into that folder) and reload."
    )
    st.stop()

if not data.get('coach_data_available', True):
    st.sidebar.info(
        f"Coach/exec data isn't sourced yet for {league}. Coaching scores show "
        "as a neutral placeholder (0.5) for every team - it doesn't affect "
        "relative ranking within this league, just narrows what the "
        "composite score can differentiate on."
    )

# Selectable-population views: a synthetic prior-cutoff-season row may exist
# in `scored`/`data['team_stats']` (added specifically to power prior-season
# lookups for the first real season), but must NEVER appear as a pickable
# option anywhere - dropdowns, leaderboards, season filters all use these
# filtered versions instead. Prior-season lookups (get_prior_season_row,
# build_year_over_year_leaderboard) and z-scoring/percentile populations
# intentionally keep using the FULL `scored`/`ws` so that backing row can
# still feed comparisons.
cutoff_season = DATA_CUTOFF_SEASON[league]
team_stats_selectable = data['team_stats'][data['team_stats']['Season'] >= cutoff_season]
scored_selectable = scored[scored['Season'] >= cutoff_season]
ws_selectable_base = ws[ws['Season'] >= cutoff_season]

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
# HEADER
# ------------------------------------------------------------------
earliest_selectable_season = team_stats_selectable['Season'].min()
st.title(f"🏀 {league} Team Jump Comparison Tool")
st.caption(
    f"Compare any two team-seasons ({earliest_selectable_season} to present) across coaching, "
    "efficiency, roster strength, and playoff performance.  \n"
    f"**Data sourced via the nba_api pipeline (`league_id`-aware) for {league}.**"
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
    values = (df['Miscellaneous: ORtg'] - df['Miscellaneous: DRtg']) if col is None else df[col]
    valid = df.assign(_val=values).dropna(subset=['_val'])
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
        team_year = f"{closest_row['Team']} {closest_row['Season']}"
        hover = f"{stat_label}: {stat_val:.2f}<br>{team_year}" if is_exact else f"{stat_label}: {stat_val:.2f}<br>Closest: {team_year}"
        points.append({'value': stat_val, 'hover': hover})
    return points


def render_box_whisker_section(metrics, row_a, row_b, label_a, label_b, population_mode, key_prefix):
    """
    metrics: list of (column_name_or_None, display_label) tuples.
             column_name=None is handled as Net Rating (computed, not a raw column).
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

    def get_val(row, col):
        return (row['Miscellaneous: ORtg'] - row['Miscellaneous: DRtg']) if col is None else row[col]

    def get_pop_series(df, col):
        return (df['Miscellaneous: ORtg'] - df['Miscellaneous: DRtg']) if col is None else df[col]

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
        val_a, val_b = get_val(row_a, col), get_val(row_b, col)

        if population_mode == "All-time (pooled)" or same_era:
            pop_label = "All-time" if population_mode == "All-time (pooled)" else row_a['era']
            pop_df = scored if population_mode == "All-time (pooled)" else scored[scored['era'] == row_a['era']]
            population = get_pop_series(pop_df, col)
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
            fig.add_trace(go.Box(y=get_pop_series(pop_a_df, col), name=row_a['era'], boxpoints=False, marker_color='lightgray', showlegend=False, hoverinfo='skip'), row=r, col=c)
            fig.add_trace(go.Box(y=get_pop_series(pop_b_df, col), name=row_b['era'], boxpoints=False, marker_color='lightgray', showlegend=False, hoverinfo='skip'), row=r, col=c)
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
            season_label = get_exact_season_label(scored, sel_team, sel_season)

            st.info(f"Selected: **{sel_team} {sel_season}** — after clicking below, go to the ⚖️ Compare Teams tab to see it applied (Streamlit can't switch tabs automatically).")
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
        default_team_a = next((label for label, code in team_display_options if code == 'BOS'), team_labels[0])
        team_label_a = st.selectbox("Team A", team_labels, index=team_labels.index(default_team_a), key="team_a_select")
        team_a = team_code_lookup[team_label_a]
        season_options_a = get_season_options_for_team(scored_selectable, team_a)
        season_labels_a = [label for label, season in season_options_a]
        season_lookup_a = dict(season_options_a)
        default_season_label_a = next((l for l, s in season_options_a if s == cutoff_season), season_labels_a[0])
        season_label_a = st.selectbox("Season A", season_labels_a, index=season_labels_a.index(default_season_label_a), key="season_a_select")
        season_a = season_lookup_a[season_label_a]
    with col_b:
        default_team_b = next((label for label, code in team_display_options if code == 'LAL'), team_labels[1] if len(team_labels) > 1 else team_labels[0])
        team_label_b = st.selectbox("Team B", team_labels, index=team_labels.index(default_team_b), key="team_b_select")
        team_b = team_code_lookup[team_label_b]
        season_options_b = get_season_options_for_team(scored_selectable, team_b)
        season_labels_b = [label for label, season in season_options_b]
        season_lookup_b = dict(season_options_b)
        default_season_label_b = next((l for l, s in season_options_b if s == cutoff_season), season_labels_b[0])
        season_label_b = st.selectbox("Season B", season_labels_b, index=season_labels_b.index(default_season_label_b), key="season_b_select")
        season_b = season_lookup_b[season_label_b]

    try:
        comparison = compare_two_teams(scored, ws, (season_a, team_a), (season_b, team_b))
    except ValueError as e:
        st.error(str(e))
        st.stop()

    row_a, row_b = comparison['team_a'], comparison['team_b']
    label_a, label_b = f"{team_a} {season_a}", f"{team_b} {season_b}"

    # --- Era adjustment toggle ---
    era_labels = [label for _, _, label in ERA_BOUNDARIES[league]]
    toggle_cols = st.columns([2, 3])
    with toggle_cols[0]:
        era_adjust_on = st.toggle("Era-adjust this comparison", value=False)
    with toggle_cols[1]:
        default_base_idx = era_labels.index('The Efficiency Explosion')
        base_era = st.selectbox("Base era (restate both teams in these terms)", era_labels, index=default_base_idx, disabled=not era_adjust_on)

    era_result = None
    if era_adjust_on:
        era_result = apply_era_adjustment(scored, row_a, row_b, base_era)
        st.info(
            f"📐 Showing **{label_a}** and **{label_b}** restated as if each played in **{base_era}**. "
            f"SRS, Net Rating, Roster/WS, and Coaching are quantile-mapped from each team's own era "
            f"onto the equivalent percentile in {base_era}'s distribution. Playoffs is left un-transformed "
            f"but re-standardized against {base_era}'s own distribution. See the Methodology tab for details."
        )

    st.divider()

    # --- Team headers, symmetric layout, with quartile-colored win% and prior-season delta ---
    win_pct_population = scored['W/L%']

    def render_record_block(row, season, team):
        current_pctile = percentile_of(row['W/L%'], win_pct_population)
        st.metric("Record", f"{int(row['W'])}-{int(row['L'])}", help="Wins-Losses")
        st.markdown(f"Win% {render_quartile_badge(row['W/L%'], current_pctile)}", unsafe_allow_html=True)

        prior_row = get_prior_season_row(scored, season, team)
        if prior_row is not None:
            prior_pctile = percentile_of(prior_row['W/L%'], win_pct_population)
            st.caption(f"Prior season ({prior_row['Team']} {prior_row['Season']}): {int(prior_row['W'])}-{int(prior_row['L'])}")
            st.markdown(f"<span style='font-size:0.85em;'>Prior Win% {render_quartile_badge(prior_row['W/L%'], prior_pctile)}</span>", unsafe_allow_html=True)
            delta_badge = render_delta_badge(row['W/L%'], prior_row['W/L%'])
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
    if era_adjust_on:
        winner_is_a = era_result['overall_winner'] == 'A'
        score_a, score_b = era_result['composite_a'], era_result['composite_b']
    else:
        winner_is_a = row_a['composite_score'] > row_b['composite_score']
        score_a, score_b = row_a['composite_score'], row_b['composite_score']

    winner_label = label_a if winner_is_a else label_b
    st.success(f"🏆 **Overall Composite Winner: {winner_label}**" + (f" (era-adjusted to {base_era})" if era_adjust_on else ""))

    score_cols = st.columns(2)
    for col, (label, row, score, season, team) in zip(score_cols, [(label_a, row_a, score_a, season_a, team_a), (label_b, row_b, score_b, season_b, team_b)]):
        with col:
            st.markdown(f"**{label} — Composite Score**")
            st.markdown(f"<span style='font-size:1.6em; font-weight:bold;'>{score:.3f}</span>", unsafe_allow_html=True)
            if not era_adjust_on:
                pctile = percentile_of(score, scored['composite_score'])
                st.markdown(render_quartile_badge(score, pctile), unsafe_allow_html=True)

                prior_row = get_prior_season_row(scored, season, team)
                if prior_row is not None:
                    prior_pctile = percentile_of(prior_row['composite_score'], scored['composite_score'])
                    st.markdown(
                        f"Prior ({prior_row['Season']}): <span style='font-size:0.85em;'>{prior_row['composite_score']:.3f} "
                        f"{render_quartile_badge(prior_row['composite_score'], prior_pctile)}</span>",
                        unsafe_allow_html=True
                    )
                    delta_badge = render_delta_badge(score, prior_row['composite_score'])
                    st.markdown(f"Change: {delta_badge}", unsafe_allow_html=True)
                else:
                    st.caption("No prior-season composite score available (predates our scored data).")

                rank_cols = st.columns(3)
                with rank_cols[0]:
                    st.metric("All-Time Rank (1980+)", f"{row['all_time_rank']}/{row['all_time_total']}")
                with rank_cols[1]:
                    st.metric("Era Rank", f"{row['era_rank']}/{row['era_total']}")
                with rank_cols[2]:
                    st.metric("Season Rank", f"{row['season_rank']}/{row['season_total']}")

    st.divider()

    # --- SCORED CATEGORIES: box-and-whisker (replaces old list + redundant bar chart) ---
    st.subheader("Scored Categories")
    scored_pop_mode = st.radio("Population", ["All-time (pooled)", "Each team's own era"], horizontal=True, key="scored_pop_mode")

    if era_adjust_on:
        st.caption("Showing era-adjusted values (quantile-mapped to " + base_era + ")")
        adj_metrics = ['SRS', 'Net Rating', 'Roster (Win Shares)', 'Coaching']
        cols = st.columns(4)
        for i, m in enumerate(adj_metrics):
            with cols[i]:
                edge = era_result['edges'][m]
                marker_a = "🏆 " if edge == 'A' else ""
                marker_b = "🏆 " if edge == 'B' else ""
                st.markdown(f"**{m}**")
                st.write(f"{marker_a}{label_a}: {era_result['adjusted_a'][m]:.2f}")
                st.write(f"{marker_b}{label_b}: {era_result['adjusted_b'][m]:.2f}")
                st.caption(STAT_DEFINITIONS.get(m, ""))
    else:
        scored_metrics = [('Miscellaneous: SRS', 'SRS'), (None, 'Net Rating'), ('team_total_ws', 'Roster (Win Shares)'), ('coaching_score', 'Coaching')]
        render_box_whisker_section(scored_metrics, row_a, row_b, label_a, label_b, scored_pop_mode, key_prefix="scored")

        # Playoffs shown separately - a 6-point ordinal doesn't box-plot meaningfully
        st.markdown("**Playoffs**")
        pcol1, pcol2 = st.columns(2)

        def playoff_display(row):
            if row['playoff_status'] == 'missed':
                return 'Missed'
            elif row['playoff_status'] == 'ambiguous':
                return 'Unresolved (1980-83 bye-era gap)'
            return row['playoff_round_reached']

        with pcol1:
            marker = "🏆 " if comparison['edges']['Playoffs'] == 'A' else ""
            st.write(f"{marker}{label_a}: **{playoff_display(row_a)}**")
        with pcol2:
            marker = "🏆 " if comparison['edges']['Playoffs'] == 'B' else ""
            st.write(f"{marker}{label_b}: **{playoff_display(row_b)}**")
        st.caption(STAT_DEFINITIONS['Playoffs'])

    st.divider()

    # --- FOUR FACTORS: box-and-whisker (existing, kept) ---
    st.subheader("Four Factors")
    st.caption("Click a legend entry to show/hide that trace - this is Plotly's built-in interactive legend, not a separate app control.")
    ff_pop_mode = st.radio("Population for Four Factors", ["All-time (pooled)", "Each team's own era"], horizontal=True, key="ff_pop_mode")
    four_factors = [('Team Shooting: eFG%', 'eFG%'), ('Team: TOV%', 'TOV%'), ('Team: ORB%', 'ORB%'), ('Team: FTr', 'FTr')]
    render_box_whisker_section(four_factors, row_a, row_b, label_a, label_b, ff_pop_mode, key_prefix="fourfactors")

    st.divider()

    # --- ADDITIONAL ADVANCED METRICS: box-and-whisker (replaces "Additional Context") ---
    st.subheader("Additional Advanced Metrics")
    st.caption("Context beyond the scored categories - not part of the composite score.")
    adv_pop_mode = st.radio("Population for Additional Metrics", ["All-time (pooled)", "Each team's own era"], horizontal=True, key="adv_pop_mode")
    additional_metrics = [
        ('Miscellaneous: Pace', 'Pace'), ('Miscellaneous: ORtg', 'ORtg'), ('Miscellaneous: DRtg', 'DRtg'),
        ('Team Shooting: TS%', 'TS%'), ('Miscellaneous: MOV', 'MOV'), ('Miscellaneous: SOS', 'SOS'),
    ]
    render_box_whisker_section(additional_metrics, row_a, row_b, label_a, label_b, adv_pop_mode, key_prefix="advanced")

    st.divider()

    # --- What Changed ---
    st.subheader("What Changed (vs. the prior season)")
    st.caption("Additions/losses (by minutes), plus coach/exec changes. eFG% and USG% shown alongside MP/WS to help identify not just WHO changed but WHY it mattered - eFG% for shooting efficiency, USG% for role/impact size. TS% and TOV% omitted to keep the table scannable (see the Roster Contribution section's rationale below for why eFG%+USG% over eFG%+TS%).")
    change_cols = st.columns(2)
    for col, (season, team, label) in zip(change_cols, [(season_a, team_a, label_a), (season_b, team_b, label_b)]):
        with col:
            st.markdown(f"**{label}**")
            panel = build_transparency_panel(scored, data['player_stats'], exec_season_wins, season, team,
                                               coach_tenures=data['coach_tenures'], exec_tenures=data['exec_tenures'])
            if not panel['has_prior_data']:
                st.info("No prior-season data available - likely an expansion franchise's first season.")
                continue

            prior_label = f"{panel['prior_team']} {panel['prior_season']}"
            st.caption(f"Compared to {prior_label}" + (" (bridged relocation/rename)" if panel['is_relocation_year'] else ""))

            if panel['coach_changed'] is None:
                st.write(f"Coach: {panel.get('coach_current') or 'unknown'} (prior unknown)")
            elif panel['coach_changed']:
                st.warning(f"🔄 Coach changed: {panel['coach_prior']} → {panel['coach_current']}")
            else:
                st.write(f"Coach: {panel['coach_current']} (unchanged)")

            if panel['exec_changed'] is None:
                st.write("Exec: no data available")
            elif panel['exec_changed']:
                st.warning(f"🔄 Exec changed: {panel['exec_prior']} → {panel['exec_current']}")
            else:
                st.write(f"Exec: {panel['exec_current']} (unchanged)")

            if not panel.get('player_data_available_for_prior', True):
                st.caption("ℹ️ No player-level data exists yet for the prior season.")
            else:
                st.markdown("**Key additions**")
                if len(panel['players_added']):
                    st.dataframe(panel['players_added'].rename(columns={'MP': 'Minutes', 'WS': 'Win Shares'}), hide_index=True, use_container_width=True)
                else:
                    st.write("_None_")
                st.markdown("**Key losses**")
                if len(panel['players_lost']):
                    st.dataframe(panel['players_lost'].rename(columns={'MP': 'Minutes', 'WS': 'Win Shares'}), hide_index=True, use_container_width=True)
                else:
                    st.write("_None_")

    with st.expander("What do these mean?"):
        for stat in ['MP', 'WS', 'eFG%', 'USG%']:
            st.markdown(f"**{stat}**: {STAT_DEFINITIONS.get(stat, 'No definition available.')}")

    st.divider()

    # --- Roster Contribution ---
    st.subheader("Roster Contribution (Win Shares → % of team wins)")
    st.caption("eFG% and USG% shown alongside WS - efficiency and role size, the two dimensions behind a player's contribution. Blank % of team wins means that row is a multi-team combined season (a player traded mid-year) - the metric isn't meaningful for a blended team label, so it's left blank rather than shown as a misleading number.")
    roster_cols_list = ['Player', 'MP', 'WS', 'eFG%', 'USG%', 'pct_of_team_wins']
    roster_cols = st.columns(2)
    with roster_cols[0]:
        st.markdown(f"**{label_a}**")
        st.dataframe(comparison['roster_a'][roster_cols_list].rename(columns={'pct_of_team_wins': '% of team wins'}), hide_index=True, use_container_width=True)
    with roster_cols[1]:
        st.markdown(f"**{label_b}**")
        st.dataframe(comparison['roster_b'][roster_cols_list].rename(columns={'pct_of_team_wins': '% of team wins'}), hide_index=True, use_container_width=True)
    with st.expander("What do these mean?"):
        for stat in ['MP', 'WS', 'eFG%', 'USG%', '% of team wins']:
            st.markdown(f"**{stat}**: {STAT_DEFINITIONS.get(stat, 'No definition available.')}")

    st.divider()

    # --- Coach & Executive (moved below What Changed / Roster, per request) ---
    st.subheader("Coach & Executive")
    coach_exec_cols = st.columns(2)
    for col, (season, team, row) in zip(coach_exec_cols, [(season_a, team_a, row_a), (season_b, team_b, row_b)]):
        with col:
            st.markdown(f"**{team} {season}**")
            if row.get('multi_coach_season'):
                st.warning("⚠️ More than one coach this season - showing all")
                detail = build_multi_coach_season_detail(scored, data['coach_tenures'], data['coach_season_wins'], data['coach_awards'], season, team, league)
                for c in detail:
                    pct_str = f"{c['pct_of_season']}% of games" if c['pct_of_season'] is not None else "split not derivable from data"
                    award_badge = " 🏆 COY" if c['won_award_this_season'] else ""
                    pctile = percentile_of(c['coaching_score'], scored['coaching_score'])
                    st.write(f"**{c['coach']}**{award_badge} — {pct_str}")
                    st.caption(f"Score: {round(c['coaching_score'],3)} ({pctile}th percentile of all team-seasons)")
            else:
                pre1980_note = " (incl. pre-1980 tenure data)" if row.get('used_pre_cutoff_tenure_data') else ""
                coach_note = f" ⚠️ {row['note']}" if pd.notna(row.get('note')) else ""
                award_badge = " 🏆 COY" if row.get('won_award_this_season_coach') else ""
                pctile = percentile_of(row['coaching_score'], scored['coaching_score'])
                st.write(f"**Coach: {row['coach_name']}**{award_badge}{pre1980_note}{coach_note}")
                st.caption(f"Score: {round(row['coaching_score'],3)} ({pctile}th percentile) — {STAT_DEFINITIONS['Coaching']}")

            st.write("")
            if pd.notna(row.get('exec_name')):
                exec_award = " 🏆 EOY" if row.get('won_award_this_season') else ""
                exec_pctile = percentile_of(row['exec_score'], scored['exec_score']) if pd.notna(row.get('exec_score')) else None
                st.write(f"**Exec: {row['exec_name']}**{exec_award}")
                if exec_pctile is not None:
                    st.caption(f"Score: {round(row['exec_score'],3)} ({exec_pctile}th percentile) — display-only, not part of the composite")
                else:
                    st.caption("Score not available")
            else:
                st.write("**Exec:** no data available")

    st.caption("⚠️ Coaching and executive scoring formulas are early-stage and flagged internally for further review/tuning - treat scores as directional, not final.")


# ====================================================================
# TAB 2: LEADERBOARDS
# ====================================================================
with tab_leaderboards:
    lb_team, lb_yoy, lb_player, lb_coach, lb_exec = st.tabs(["Teams", "Year-over-Year Changes", "Players", "Coaches", "Executives"])

    with lb_team:
        st.subheader("All-Time Team Leaderboard")
        filter_cols = st.columns(2)
        with filter_cols[0]:
            season_filter = st.selectbox("Filter to a single season (optional)", ["All seasons"] + sorted(team_stats_selectable['Season'].unique(), reverse=True), key="team_lb_filter")
        with filter_cols[1]:
            era_filter = st.selectbox("Filter to an era (optional)", ["All eras"] + [label for _, _, label in ERA_BOUNDARIES[league]], key="team_lb_era_filter")

        display_df = scored_selectable[['all_time_rank', 'season_rank', 'Season', 'era', 'Team', 'W', 'L', 'composite_score',
                              'Miscellaneous: SRS', 'playoff_status', 'playoff_round_reached', 'coach_name']].copy()
        display_df['Playoff Result'] = display_df.apply(
            lambda r: 'Missed' if r['playoff_status'] == 'missed'
            else ('Unresolved (1980-83 gap)' if r['playoff_status'] == 'ambiguous' else r['playoff_round_reached']),
            axis=1
        )
        display_df = display_df.drop(columns=['playoff_status', 'playoff_round_reached'])
        display_df = display_df.rename(columns={
            'Miscellaneous: SRS': 'SRS', 'composite_score': 'Composite Score', 'coach_name': 'Coach',
            'all_time_rank': 'All-Time Rank (1980+)', 'season_rank': 'Season Rank', 'era': 'Era'
        })
        display_df['Composite Score'] = display_df['Composite Score'].round(3)
        if season_filter != "All seasons":
            display_df = display_df[display_df['Season'] == season_filter]
        if era_filter != "All eras":
            display_df = display_df[display_df['Era'] == era_filter]
        display_df = display_df.sort_values('All-Time Rank (1980+)').reset_index(drop=True)
        render_send_to_compare_table(display_df, key_prefix="team_lb")

    with lb_yoy:
        st.subheader("Year-over-Year Changes")
        st.caption(
            "Every team-season compared to its immediately prior season (bridging relocations/renames). "
            "Sort by any column to find the biggest jumps or collapses - this is the direct analytical "
            "payoff of the project's original 'which teams made a big jump' premise. "
            "⚠️ Win Change can be misleading across lockout-shortened seasons (1998-99, 2011-12, 2019-20) "
            "since games played differs - Composite Score Change (built from rate stats) is more reliable there."
        )
        yoy_lb = build_year_over_year_leaderboard(scored)
        direction = st.radio("Show", ["Biggest jumps first", "Biggest collapses first"], horizontal=True, key="yoy_direction")
        yoy_display = yoy_lb.sort_values('Composite Score Change', ascending=(direction == "Biggest collapses first")).reset_index(drop=True)
        render_send_to_compare_table(yoy_display, key_prefix="yoy_lb")

    with lb_player:
        st.subheader("Player-Season Leaderboard")
        st.caption("Ranked by Win Shares. '% of team wins' is that player's renormalized share of their team's actual win total.")
        player_season_filter = st.selectbox("Filter to a single season", ["All seasons"] + sorted(scored_selectable['Season'].unique(), reverse=True), key="player_lb_filter")
        # restrict to the selectable population (1979-80+) - ws itself carries
        # 1978-79 rows too (needed by the transparency panel's roster diff),
        # but that season should never surface on a user-facing leaderboard
        ws_selectable = ws[ws['Season'].isin(scored_selectable['Season'].unique())]
        player_lb = build_player_leaderboard(ws_selectable, player_season_filter)
        st.dataframe(player_lb, hide_index=True, use_container_width=True, height=550)

    with lb_coach:
        st.subheader("Coach Career Leaderboard")
        if data.get('coach_data_available', True):
            cutoff_label = DATA_CUTOFF_SEASON[league].split('-')[0]
            st.caption(f"W/L shown as TWO separate figures: {cutoff_label}-onward-only (what our season-level data directly covers) and all-time (adding pre-{cutoff_label} tenure aggregates where available) - shown explicitly rather than blended, so it's never ambiguous which era's wins are being counted.")
            coach_lb = build_coach_leaderboard(data['coach_season_wins'], data['coach_tenures'], league, data['coach_awards'])
            st.dataframe(coach_lb, hide_index=True, use_container_width=True, height=550)
        else:
            st.info(f"No coach data sourced yet for {league}.")

    with lb_exec:
        st.subheader("Executive Career Leaderboard")
        st.caption("Executives are NOT part of the scored composite. Win totals are a proxy (the real record of whichever team they led), 1980+ only - no pre-1980 fallback exists for execs since source data has no win/loss at all before that.")
        if len(exec_season_wins):
            exec_lb = build_exec_leaderboard(exec_season_wins, data['exec_awards'])
            st.dataframe(exec_lb, hide_index=True, use_container_width=True, height=550)
        else:
            st.info("No executive data loaded.")


# ====================================================================
# TAB 3: METHODOLOGY
# ====================================================================
with tab_methodology:
    st.header("How the scoring works")

    st.markdown("""
    The composite score blends **five categories**, each standardized (z-scored) across **1,336 team-seasons**
    (1978-79 through present - see the note below on why 1978-79 is one season wider than what's actually
    selectable) before weighting, so a team's performance is measured relative to the full population
    rather than on raw, era-dependent units. **1,314 of those (1979-80 onward) are user-selectable** -
    everything you can pick in a dropdown, every leaderboard row, every ranking is drawn from that subset.
    """)

    st.subheader("Category weights")
    weight_df = pd.DataFrame([
        {"Category": "SRS (Simple Rating System)", "Weight": f"{WEIGHTS['srs']:.0%}",
         "Why": "Best single 'how good were they' number - point differential adjusted for strength of schedule."},
        {"Category": "Net Rating", "Weight": f"{WEIGHTS['net_rating']:.0%}",
         "Why": "ORtg - DRtg. Correlated with SRS by design - kept separate so agreement/disagreement is itself informative."},
        {"Category": "Playoffs", "Weight": f"{WEIGHTS['playoffs']:.0%}",
         "Why": "Ordinal scale (0=missed, 5=champion). Weighted lower deliberately since playoff results are noisier/smaller-sample than a full season."},
        {"Category": "Roster / Win Shares", "Weight": f"{WEIGHTS['roster_ws']:.0%}", "Why": "Sum of the roster's Win Shares - the personnel-strength signal."},
        {"Category": "Coaching", "Weight": f"{WEIGHTS['coaching']:.0%}", "Why": "Built from the coach's 'at the time' record - see below."},
    ])
    st.table(weight_df)

    st.subheader("Coaching sub-score formula")
    st.code("coaching_score = 0.55 * prior_win_pct_shrunk + 0.25 * experience_normalized + 0.20 * accolade_score", language="python")
    st.markdown("""
    - **`prior_win_pct_shrunk`**: win% from ALL seasons strictly *before* the evaluated season, shrunk toward
      league-average (0.5) via Bayesian credibility weighting: `weight = games / (games + 41)`.
    - **`experience_normalized`**: prior seasons coached (any team), capped at 15 for full credit.
    - **`accolade_score`**: prior Coach of the Year awards, capped at 2 for full credit.
    - **Known gap**: playoff coaching record isn't yet folded in - regular season only, for now.
    """)

    st.subheader("Executive scoring (display-only, NOT part of the composite)")
    st.code("exec_score = 0.55 * prior_win_pct_shrunk + 0.25 * experience_normalized + 0.20 * accolade_score", language="python")
    st.markdown("""
    Same formula shape as coaching, deliberately kept **out of the composite score**: an executive's
    "win%" here is literally the same team's actual win total during their tenure, so folding it into a
    composite that already scores SRS/Net Rating for the same seasons would double-count the same
    outcome under two different labels. It's shown as a standalone percentile on the comparison view and
    the Executive leaderboard instead - useful context, not treated as independent evidence.

    A real data limitation worth stating plainly: `exec_tenures.csv` has no win/loss data at all (only
    tenure dates), so unlike coaching there's no tenure-aggregate fallback for pre-1980 executive history -
    that data is simply unavailable further back, not just harder to compute.
    """)
    st.warning("⚠️ Both the coaching and executive formulas are early-stage - the weights (0.55/0.25/0.20) were a reasonable starting point, not the product of a rigorous fitting/validation process. Flagged for further back-and-forth analysis and tuning.")

    st.subheader("The pre-1980 coaching data problem, and the hybrid fix")
    st.markdown("""
    File 1 only goes back to **1979-80**. For any coaching tenure that ended *entirely* before 1980,
    this model falls back to that tenure's aggregate win-loss record (from `coach_tenures.csv`). A tenure
    that *straddles* the 1980 cutoff is deliberately **excluded** from this fallback, since the aggregate
    can't be split and using it whole would double-count games already captured at the season level.
    """)

    st.subheader("The 1978-79 'background data' architecture")
    st.markdown("""
    File 1 (team stats) and File 2 (playoffs) were both extended to include 1978-79, giving that season
    a fully real, computed composite score - not just enough to power the roster diff, but enough to
    support genuine Record and Composite Score "vs. prior season" comparisons and Year-over-Year Change
    calculations for every 1979-80 team (e.g. correctly surfacing the 1979-80 Celtics' +32 wins and
    Larry Bird's arrival as the largest single-season jump tied to that boundary in the dataset).

    1978-79 is still deliberately kept **out of every selectable population**: it's not a dropdown option,
    doesn't appear as its own row on any leaderboard, and can never be picked as Team A or B. It exists
    purely as backing data for whatever needs a "prior season" reference to 1979-80. Under the hood, this
    means the app maintains two views of the same underlying table: the full 1,336-row population (used
    for z-scoring, prior-season lookups, and Year-over-Year deltas) and a filtered 1,314-row "selectable"
    view (used for every dropdown, leaderboard, and ranking) - both are computed once, and it's purely a
    display-layer filter that keeps 1978-79 from ever surfacing as a pickable option, not a separate
    scoring pipeline.

    Coach/exec continuity for the 1979-80 boundary specifically also has a second-tier fallback using
    `coach_tenures`/`exec_tenures` directly (which extend back to 1947, further than File 1 ever will),
    for cases where even the extended File 1 data wouldn't help.
    """)

    st.subheader("Player win-share contribution")
    st.code("player_pct_of_wins = (player_WS / team_total_WS) * team_actual_wins", language="python")
    st.markdown("""
    Rows with a combined multi-team label (e.g. `"BOS,GSW"` - a player traded mid-season from a
    season-wide, non-team-filtered pull) are **excluded** from this calculation and show a blank
    percentage rather than a number. Grouping by the literal team-code string means a combined label
    forms a group of exactly one player, which would otherwise make their share trivially (and
    misleadingly) 100% - not a real signal, since a blended multi-team season isn't a coherent "share of
    one team's roster" in the first place.
    """)

    st.subheader("Era adjustment (quantile mapping)")
    st.markdown("""
    The era-adjustment toggle restates a team's SRS, Net Rating, Roster/WS, and Coaching score as if they
    played in a different ("base") era, using **quantile mapping**: find the team's percentile within
    their own era's distribution, then read off the value at that same percentile in the base era's
    distribution. Playoffs is deliberately **not** quantile-mapped (a championship means the same thing
    in any era) but is re-standardized against the base era's own playoff distribution so it still
    combines consistently with the adjusted metrics. When enabled, this fully recomputes the category
    edges, composite score, and winner - not just a display overlay.
    """)

    st.subheader("Box-and-whisker population toggle")
    st.markdown("""
    Every box-and-whisker section (Scored Categories, Four Factors, Additional Advanced Metrics) offers
    the same choice: plot both teams' dots against the **all-time pooled** distribution, or against
    **each team's own era**. When the two teams share an era, that's one box; when they're from
    different eras, each team's dot is shown against its own era's distribution side by side, rather than
    pooling two different populations together.
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
    - **Shooting splits, Four Factors, and other advanced metrics** - Net Rating already captures the
      efficiency signal these feed into; scoring them separately would double-count.
    - **Payroll** - deferred entirely; historical salary data is thin/unreliable pre-1990s.
    - **Executives** - see above; tracked and percentiled, never composited.
    """)

    st.subheader("Known limitations, stated plainly")
    st.markdown("""
    - Seasons with more than one coach score each coach independently using their own prior history;
      the games-based split between them (when derivable) reflects tenure-level data, not exact dates.
    - Executive win totals assume date-range tenure overlap maps cleanly onto season boundaries.
    - `coach_tenures.csv` has a known mislabeling issue for at least one franchise (some Charlotte
      Bobcats-labeled rows actually contain the original Charlotte Hornets' pre-2002 history) - worked
      around via the 1979-80-only fallback restriction above, but the source file itself still needs a
      manual cleanup pass.
    - Coaching and executive scoring formulas are directional first drafts, not finalized.
    - The New Orleans Jazz (Utah's franchise before their 1979 relocation) required adding a new team
      code (`NOJ`) and franchise-lineage entry - a good example of the kind of one-off franchise-history
      gap that surfaces whenever the data range gets extended.
    """)

    st.divider()
    st.header("Era Classification Methodology")
    st.markdown("""
    Six eras are used throughout the app for filtering and comparing teams against contemporaries.
    Boundaries went through a three-step validation process: candidate boundaries from known NBA
    rule-change history, validation against league-wide trend data, and an objective cross-check via
    PELT changepoint detection (run separately on offense/pace and defense signal sets, at multiple
    penalty sensitivities, since offense-driven and defense-driven regime shifts turned out to be
    genuinely different structural forces rather than always coinciding).
    """)

    st.subheader("Changepoint detection results")
    cp_results = pd.DataFrame([
        {"Signal set": "Offense/Pace (pace, ORtg, eFG%, 3PAr)", "Most robust break(s)": "1993-94, 2018-19",
         "Weaker/penalty-dependent": "2003-04, 2013-14"},
        {"Signal set": "Defense (opp. TOV%, DRB%, opp. FTr, STL/BLK rate)", "Most robust break(s)": "2003-04",
         "Weaker/penalty-dependent": "1988-89, 2013-14"},
    ])
    st.table(cp_results)

    st.subheader("Era-by-era: what actually happened, and why")
    st.markdown("""
    **The Pace & Post Era (1979-80 to 1993-94)** — the baseline period. The 3-point line existed
    (introduced 1979-80) but was barely used strategically; offense was built around traditional
    half-court and post play at high pace. No strong break signal needed to define this era's start -
    it's simply everything before the first real regime shift.

    **The Short Arc Era (1994-95 to 1996-97)** — the NBA's shortened 3-point line experiment, and the
    **strongest changepoint in the entire 46-year dataset**, robust at every penalty level tested on the
    offense signal. 3-point attempt rate spikes from 0.12 the year before to ~0.19-0.21 during these three
    seasons, then drops back to 0.16 the moment the line reverted to its original distance in 1997-98 -
    a real, temporary anomaly, not a lasting stylistic shift, which is why it gets its own short era
    rather than being folded into a neighbor.

    **The Dead-Ball Era (1997-98 to 2003-04)** — pace and scoring both bottom out league-wide; this is
    the well-documented low-scoring, physical-defense period. Its end (2003-04) is the **single strongest
    break in the defensive signal set**, more robust there than anywhere on the offense side - makes sense,
    since what ended this era was fundamentally a defensive rule change (see next).

    **The Freedom of Movement Era (2004-05 to 2013-14)** — named after the NBA's own term for the 2004
    hand-checking/illegal-defense reform. ORtg jumps from 102.9 to 106.1 the very next season - a clean,
    immediate break in scoring efficiency. Notably, **pace does NOT recover at this boundary** - it stays
    flat/depressed (~90-92) all the way through 2011-12, only climbing starting ~2012-13. Two different
    metrics, two different break patterns: the 2004 rule change reshaped *efficiency*, not *tempo*.

    **The Pace & Space Build-Up (2014-15 to 2018-19)** — gradual acceleration of both tempo and 3-point
    rate, a moderate offense-side signal (visible at looser penalties, not the most conservative one) -
    the runway before the sharper break that follows.

    **The Efficiency Explosion (2019-20 to present)** — the **second-strongest break in the whole dataset**,
    robust at every penalty level tested, exactly as strong a signal as the Short Arc Era's start. eFG%
    jumps from .502 to .514 to .521 to .524 across 2016-17 through 2018-19, and league-average ORtg breaks
    110 for the first time right at this boundary, climbing to 114-115+ by 2022-25. Unlike the Dead-Ball
    Era's end, this break shows **no corresponding defensive-side signal** - it looks like a pure offensive
    strategy shift (shot selection, spacing) rather than a rule-driven change on defense.
    """)

    st.subheader("What didn't define an era, and why that's informative")
    st.markdown("""
    - **League-wide variance has been shrinking**, not just the mean shifting - std(pace) and std(ORtg)
      are both visibly tighter in 2018-present than across most of NBA history. This is exactly why the
      composite score offers both pooled and within-era standardization: a raw point differential in a
      low-variance recent season represents a more extreme percentile finish than the same number would
      in a high-variance 1980s season.
    - **Offensive and defensive rebounding are smooth secular trends, not regime shifts** - ORB% declines
      continuously from 33.5% to the low-20s over the full 46 years (with a slight recent uptick), DRB%
      is its mirror image. Neither supports a discrete era boundary on its own, which is why rebounding
      wasn't used as an era-defining metric despite the size of the long-run change.
    - **Competitive balance (spread of team SRS within a season) tracks no proposed boundary at all** -
      it fluctuates between roughly 3.1 and 6.0 with no clean pattern tied to any of the six eras,
      suggesting parity/dominance cycles are an independent dimension from *style* eras, not something
      that moves in lockstep with pace, spacing, or rule changes.
    """)

    era_table = pd.DataFrame([
        {"Era": label, "Seasons": f"{start} to {end}"} for start, end, label in ERA_BOUNDARIES[league]
    ])
    st.table(era_table)

    st.divider()
    st.header("Future Enhancements")
    st.markdown("""
    This is a working v1, not a finished product. Concrete directions for further sophistication:

    **Additional data sources**
    - **Deeper playoff data**: currently just an ordinal round-reached scale - series length, margin of
      victory in each round, and opponent quality faced would let playoff performance carry real
      information beyond "how far did they get."
    - **Payroll/salary data**: deferred entirely so far; historical salary data is thin and unreliable
      pre-1990s, but could be layered in for recent decades to explore value-for-money questions
      (wins per payroll dollar) rather than just raw spending.
    - **Additional player/team-level metrics**: shot-location and shot-quality data, on/off-court impact
      metrics, and lineup-level data would all deepen the Roster and Four Factors analysis well beyond
      what season-long box-score aggregates can show.

    **Predictive modeling and optimization**
    - **Fitting the composite weights rather than hand-setting them**: the current SRS/Net Rating/
      Playoffs/Roster/Coaching weights (30/20/15/20/15) and the coaching/exec sub-formula weights
      (0.55/0.25/0.20) were reasonable starting points, not the output of a rigorous fitting process.
      A natural next step: define a target outcome (e.g. predicting next-season win total, or matching
      some independent "greatest teams" consensus ranking) and fit weights via regression or a
      constrained optimization, rather than asserting them.
    - **ML-based feature selection**: techniques like gradient-boosted trees or LASSO regression could
      surface which underlying signals actually carry the most predictive weight, potentially revealing
      that some currently-weighted factors matter less than assumed, or that an unused metric
      (e.g. a Four Factors component, or SOS specifically) deserves a place in the composite.
    - **Tuning the shrinkage constants empirically**: the coaching/exec Bayesian shrinkage pseudo-games
      constant (41, roughly half a season) was a reasonable-sounding default, not cross-validated against
      actual predictive performance - a natural target for the same fitting exercise.

    These are flagged as a deliberate follow-up phase, not gaps in the current build - the present version
    prioritized getting the data pipeline, methodology, and transparency right first, with optimization
    as the next layer once that foundation is trusted.
    """)
