"""
build_win_shares.py  —  Layer 2 (shaper; ZERO API calls)

Derives Basketball-Reference-style Win Shares (Dean Oliver's method) for every
WNBA player-season, straight from the canonical box score -- no Basketball-
Reference dependency. WS is a PLAYER-LEVEL insight metric that surfaces in the
leaderboards/roster views alongside PIE; it is deliberately NOT a composite
input (its team sum is ~team wins, which is near-collinear with the Quality and
Playoffs axes the composite already scores).

Inputs  : data/processed/WNBA/player_season_stats.csv (box score, per player)
          data/processed/WNBA/team_season_stats.csv   (team + opponent totals)
Outputs : merges ows, dws, ws back into player_season_stats.csv
          (on season + player_id + team_code, idempotent)

WHY IT'S TRUSTWORTHY: Win Shares are constructed so a roster's WS sum to about
the team's actual win total. main() prints that check across all team-seasons;
the ratio should sit near 1.0. On this data it lands at ~1.008.

Multi-team rows (team_count > 1, a player traded mid-season) get a WS computed
against their listed team's context -- approximate, since their box totals span
two teams, but fine for a display metric. The sums-to-wins validation uses
single-team rows only.

Constants are Oliver's (0.92 offensive baseline, 1.08 defensive baseline, 0.32
marginal-points-per-win, 1.07 ORB, 0.4 FT-possession, 1.14 assist weight). The
0.32 factor self-scales to the league's scoring level and pace, so the WNBA's
lower-scoring, 40-minute environment is handled without re-tuning; Tm_MP uses
5 x team game-minutes (five players on the floor).
"""

import numpy as np
import pandas as pd

PLAYER_CSV = "data/processed/WNBA/player_season_stats.csv"
TEAM_CSV = "data/processed/WNBA/team_season_stats.csv"


def _safe(numer, denom):
    """Elementwise divide, yielding 0 where the denominator is 0 (matches the
    'no attempts -> no contribution' intent of Oliver's ratios)."""
    denom = np.asarray(denom, dtype=float)
    numer = np.asarray(numer, dtype=float)
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)


def _team_context(teams):
    """Team + opponent totals renamed to the tm_/opp_ names the WS chain uses,
    plus derived Tm_MP (team player-minutes) and Oliver team possessions."""
    tm = teams.rename(columns={
        'fgm': 'tm_fg', 'fga': 'tm_fga', 'fg3m': 'tm_fg3', 'ftm': 'tm_ft', 'fta': 'tm_fta',
        'oreb': 'tm_orb', 'dreb': 'tm_drb', 'ast': 'tm_ast', 'stl': 'tm_stl', 'blk': 'tm_blk',
        'tov': 'tm_tov', 'pf': 'tm_pf', 'pts': 'tm_pts', 'pace': 'tm_pace', 'w': 'tm_w', 'g': 'tm_g',
        'opp_fgm': 'opp_fg', 'opp_fga': 'opp_fga', 'opp_ftm': 'opp_ft', 'opp_fta': 'opp_fta',
        'opp_oreb': 'opp_orb', 'opp_dreb': 'opp_drb', 'opp_tov': 'opp_tov', 'opp_pts': 'opp_pts',
    }).copy()
    tm['tm_mp'] = 5.0 * tm['min']                      # 5 players on the floor x game-minutes
    tm['tm_poss'] = 0.5 * (
        (tm['tm_fga'] + 0.4 * tm['tm_fta']
         - 1.07 * (tm['tm_orb'] / (tm['tm_orb'] + tm['opp_drb'])) * (tm['tm_fga'] - tm['tm_fg'])
         + tm['tm_tov'])
        + (tm['opp_fga'] + 0.4 * tm['opp_fta']
           - 1.07 * (tm['opp_orb'] / (tm['opp_orb'] + tm['tm_drb'])) * (tm['opp_fga'] - tm['opp_fg'])
           + tm['opp_tov']))
    keep = ['season', 'team_code', 'tm_fg', 'tm_fga', 'tm_fg3', 'tm_ft', 'tm_fta', 'tm_orb',
            'tm_drb', 'tm_ast', 'tm_stl', 'tm_blk', 'tm_tov', 'tm_pf', 'tm_pts', 'tm_mp',
            'tm_pace', 'tm_poss', 'tm_w', 'tm_g', 'opp_fg', 'opp_fga', 'opp_ft', 'opp_fta',
            'opp_orb', 'opp_drb', 'opp_tov', 'opp_pts']
    return tm[keep]


def _league_context(tm):
    lg = tm.groupby('season').apply(lambda d: pd.Series({
        'lg_pts': d['tm_pts'].sum(), 'lg_poss': d['tm_poss'].sum(),
        'lg_g': d['tm_g'].sum(), 'lg_pace': d['tm_pace'].mean(),
    })).reset_index()
    lg['lg_pts_per_poss'] = lg['lg_pts'] / lg['lg_poss']
    lg['lg_pts_per_g'] = lg['lg_pts'] / lg['lg_g']
    return lg[['season', 'lg_pts_per_poss', 'lg_pts_per_g', 'lg_pace']]


def compute_win_shares(players, teams):
    """Return one row per input player row with ows, dws, ws attached, all
    intermediates computed as locals (nothing but the final columns is kept)."""
    tm = _team_context(teams)
    lg = _league_context(tm)

    d = players.rename(columns={
        'fgm': 'fg', 'fga': 'fga', 'fg3m': 'fg3', 'ftm': 'ft', 'fta': 'fta',
        'oreb': 'orb', 'dreb': 'drb', 'tov': 'tov', 'pf': 'pf', 'pts': 'pts', 'min': 'mp',
    }).merge(tm, on=['season', 'team_code'], how='left').merge(lg, on='season', how='left')

    g = d  # short alias; pull needed columns as numpy for the arithmetic
    mp, fg, fga, fg3, ft, fta = g['mp'].values, g['fg'].values, g['fga'].values, g['fg3'].values, g['ft'].values, g['fta'].values
    orb, drb, ast, stl, blk, pf, pts, tov = (g['orb'].values, g['drb'].values, g['ast'].values,
                                             g['stl'].values, g['blk'].values, g['pf'].values,
                                             g['pts'].values, g['tov'].values)
    tm_fg, tm_fga, tm_fg3, tm_ft, tm_fta = g['tm_fg'].values, g['tm_fga'].values, g['tm_fg3'].values, g['tm_ft'].values, g['tm_fta'].values
    tm_orb, tm_drb, tm_ast, tm_stl, tm_blk = g['tm_orb'].values, g['tm_drb'].values, g['tm_ast'].values, g['tm_stl'].values, g['tm_blk'].values
    tm_tov, tm_pf, tm_pts, tm_mp, tm_pace, tm_poss = g['tm_tov'].values, g['tm_pf'].values, g['tm_pts'].values, g['tm_mp'].values, g['tm_pace'].values, g['tm_poss'].values
    opp_fg, opp_fga, opp_ft, opp_fta, opp_orb, opp_tov, opp_pts = (g['opp_fg'].values, g['opp_fga'].values,
        g['opp_ft'].values, g['opp_fta'].values, g['opp_orb'].values, g['opp_tov'].values, g['opp_pts'].values)
    lg_ppp, lg_ppg, lg_pace = g['lg_pts_per_poss'].values, g['lg_pts_per_g'].values, g['lg_pace'].values

    # --- Offensive Rating / Points Produced ---
    qAST = ((mp / (tm_mp / 5)) * (1.14 * _safe(tm_ast - ast, tm_fg))
            + (_safe(_safe(tm_ast, tm_mp) * (mp * 5) - ast,
                     _safe(tm_fg, tm_mp) * (mp * 5) - fg)) * (1 - (mp / (tm_mp / 5))))
    FG_Part = fg * (1 - 0.5 * _safe(pts - ft, 2 * fga) * qAST)
    AST_Part = 0.5 * _safe((tm_pts - tm_ft) - (pts - ft), 2 * (tm_fga - fga)) * ast
    FT_Part = np.where(fta > 0, (1 - (1 - _safe(ft, fta)) ** 2) * 0.4 * fta, 0.0)
    Tm_ScPoss = tm_fg + (1 - (1 - _safe(tm_ft, tm_fta)) ** 2) * tm_fta * 0.4
    Tm_ORBpct = tm_orb / (tm_orb + g['opp_drb'].values)
    Tm_Play = _safe(Tm_ScPoss, tm_fga + tm_fta * 0.4 + tm_tov)
    Tm_ORBw = ((1 - Tm_ORBpct) * Tm_Play) / ((1 - Tm_ORBpct) * Tm_Play + Tm_ORBpct * (1 - Tm_Play))
    ORB_Part = orb * Tm_ORBw * Tm_Play
    ScPoss = (FG_Part + AST_Part + FT_Part) * (1 - _safe(tm_orb, Tm_ScPoss) * Tm_ORBw * Tm_Play) + ORB_Part
    FGxPoss = (fga - fg) * (1 - 1.07 * Tm_ORBpct)
    FTxPoss = np.where(fta > 0, ((1 - _safe(ft, fta)) ** 2) * 0.4 * fta, 0.0)
    TotPoss = ScPoss + FGxPoss + FTxPoss + tov

    PProd_FG = 2 * (fg + 0.5 * fg3) * (1 - 0.5 * _safe(pts - ft, 2 * fga) * qAST)
    PProd_AST = (2 * _safe(tm_fg - fg + 0.5 * (tm_fg3 - fg3), tm_fg - fg) * 0.5
                 * _safe((tm_pts - tm_ft) - (pts - ft), 2 * (tm_fga - fga)) * ast)
    PProd_ORB = orb * Tm_ORBw * Tm_Play * _safe(tm_pts, tm_fg + (1 - (1 - _safe(tm_ft, tm_fta)) ** 2) * 0.4 * tm_fta)
    PProd = (PProd_FG + PProd_AST + ft) * (1 - _safe(tm_orb, Tm_ScPoss) * Tm_ORBw * Tm_Play) + PProd_ORB

    marg_ppw = 0.32 * lg_ppg * (tm_pace / lg_pace)
    ows = _safe(PProd - 0.92 * lg_ppp * TotPoss, marg_ppw)

    # --- Defensive Rating / Defensive Win Shares ---
    DORpct = opp_orb / (opp_orb + tm_drb)
    DFGpct = _safe(opp_fg, opp_fga)
    FMwt = _safe(DFGpct * (1 - DORpct), DFGpct * (1 - DORpct) + (1 - DFGpct) * DORpct)
    Stops1 = stl + blk * FMwt * (1 - 1.07 * DORpct) + drb * (1 - FMwt)
    Stops2 = ((_safe(opp_fga - opp_fg - tm_blk, tm_mp) * FMwt * (1 - 1.07 * DORpct)
               + _safe(opp_tov - tm_stl, tm_mp)) * mp
              + _safe(pf, tm_pf) * 0.4 * opp_fta * (1 - _safe(opp_ft, opp_fta)) ** 2)
    Stops = Stops1 + Stops2
    Stoppct = _safe(Stops * tm_mp, tm_poss * mp)
    Tm_DRtg = 100 * _safe(opp_pts, tm_poss)
    D_Pts_ScPoss = _safe(opp_pts, opp_fg + (1 - (1 - _safe(opp_ft, opp_fta)) ** 2) * opp_fta * 0.4)
    DRtg = Tm_DRtg + 0.2 * (100 * D_Pts_ScPoss * (1 - Stoppct) - Tm_DRtg)
    dws = _safe(_safe(mp, tm_mp) * tm_poss * (1.08 * lg_ppp - DRtg / 100), marg_ppw)

    out = players[['season', 'player_id', 'team_code', 'team_count']].copy()
    out['ows'] = np.round(ows, 3)
    out['dws'] = np.round(dws, 3)
    out['ws'] = np.round(ows + dws, 3)
    return out


def merge_into_player_stats(ws, player_csv=PLAYER_CSV):
    ps = pd.read_csv(player_csv)
    new = ['ows', 'dws', 'ws']
    ps = ps.drop(columns=[c for c in new if c in ps.columns])             # re-run safe
    ps = ps.merge(ws[['season', 'player_id', 'team_code'] + new],
                  on=['season', 'player_id', 'team_code'], how='left')
    ps.to_csv(player_csv, index=False)
    print(f"[write] merged {new} into {player_csv}  ({len(ps)} rows)")


def main():
    players = pd.read_csv(PLAYER_CSV)
    teams = pd.read_csv(TEAM_CSV)
    ws = compute_win_shares(players, teams)

    # VALIDATION: roster WS should sum to ~team wins (single-team rows)
    single = ws[ws['team_count'] == 1]
    tm = teams[['season', 'team_code', 'w']]
    chk = single.groupby(['season', 'team_code'])['ws'].sum().reset_index().merge(
        tm, on=['season', 'team_code'])
    ratio = chk['ws'].sum() / chk['w'].sum()
    print("=" * 60)
    print("WIN SHARES  —  sums-to-wins validation")
    print("=" * 60)
    print(f"  corr(team WS sum, actual wins) = {chk['ws'].corr(chk['w']):.4f}")
    print(f"  total WS / total wins          = {ratio:.4f}   (want ~1.00)")
    print(f"  mean WS sum {chk['ws'].mean():.2f}  vs  mean wins {chk['w'].mean():.2f}")
    if not 0.95 <= ratio <= 1.05:
        print("  [warn] ratio outside 0.95-1.05 -- check team-minute / possession scaling")
    print("=" * 60)

    merge_into_player_stats(ws)


if __name__ == "__main__":
    main()
