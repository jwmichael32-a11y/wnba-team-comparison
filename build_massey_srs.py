"""
build_massey_srs.py  —  Layer 2 (shaper; ZERO API calls)

Reads cached regular-season game logs from
    data/raw/WNBA/regular_games/{season}.json
(one row per team per game: SEASON_ID, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME,
 GAME_ID, GAME_DATE, MATCHUP, WL, PTS, PLUS_MINUS, + box cols)

and produces a per-team-season Massey rating that decomposes into
offense / defense / strength-of-schedule — the true SRS replacement for the
v1 `net_rating` stand-in.

Outputs:
  1. data/processed/WNBA/team_massey_ratings.csv   (full auditable decomposition)
  2. merges srs, srs_off, srs_def, sos into
     data/processed/WNBA/team_season_stats.csv      (on season + team_id, idempotent)

METHOD (points-based Massey offense/defense least squares, solved per season):
  For each team-game observation we model the points that team scored:
        pts_scored = mu + offense_i - defense_j + home*beta + eps
  offense_i is team i's scoring rating, defense_j suppresses the opponent's
  points (higher defense = better), beta is home-court advantage. Fit by least
  squares over every team-game row in the season, with offense/defense centered
  to mean zero. The overall rating is
        srs_i = offense_i + defense_i
  and srs_i - srs_j is the expected neutral-court margin. Because the fit is over
  the whole schedule graph simultaneously, srs is schedule-adjusted by
  construction, so it IS the SRS analog:
        sos_i = srs_i - mov_i          (SRS identity: SRS = MOV + SOS)

MARGIN TREATMENT (the knob): raw margin overrates blowouts and, under L2 loss,
  gives them squared leverage. The treatment is applied to the game MARGIN while
  holding the game TOTAL fixed, so offense and defense compress symmetrically in
  a blowout (the winner's offense looks slightly less great AND the loser's
  defense looks slightly less bad, in equal measure):
        total S = pts_i + pts_j              (preserved)
        margin m = pts_i - pts_j             (transformed -> m')
        pts_i' = (S + m')/2 ,  pts_j' = (S - m')/2
  treatments:  'raw'  -> m' = m
               'cap'  -> m' = clip(m, -CAP, +CAP)         (Winsorize; point units)
               'tanh' -> m' = CAP * tanh(m / CAP)         (smooth, no cliff)
  NOTE: under 'cap'/'tanh' the ratings live in transformed-margin units, which is
  the intended consequence of bounding blowout influence.
"""

import json
import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RAW_DIR      = "data/raw/WNBA/regular_games"
TEAM_CSV     = "data/processed/WNBA/team_season_stats.csv"
RATINGS_CSV  = "data/processed/WNBA/team_massey_ratings.csv"
SEASONS      = list(range(1997, 2026))   # 1997..2025 inclusive

# --- margin knob (PROVISIONAL default; confirm after first diagnostic run) ---
MARGIN_TREATMENT = "cap"     # 'raw' | 'cap' | 'tanh'
MARGIN_CAP       = 26.0      # point cap / tanh scale; see diagnostic recommendation
INCLUDE_HCA      = True      # estimate a home-court term (2020 was neutral-site)

# franchise-id join key is stable across abbreviation changes, so we key the
# whole pipeline on TEAM_ID and only carry TEAM_ABBREVIATION for readability.
TEAM_ID_COL, TEAM_ABBR_COL, GAME_ID_COL = "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID"


# ----------------------------------------------------------------------
# LOAD (robust to whichever shape the raw cache stored)
# ----------------------------------------------------------------------
def load_raw_games(path):
    """Load a cached season into a DataFrame, tolerating three plausible shapes:
    a records list, a {columns:...,data:...} dict, or the raw nba_api
    {resultSets:[{headers,rowSet}]} envelope."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        if "resultSets" in obj:
            rs = obj["resultSets"][0]
            return pd.DataFrame(rs["rowSet"], columns=rs["headers"])
        if "data" in obj and "columns" in obj:
            return pd.DataFrame(obj["data"], columns=obj["columns"])
        # dict-of-columns fallback
        return pd.DataFrame(obj)
    raise ValueError(f"Unrecognized JSON shape in {path}")


# ----------------------------------------------------------------------
# BUILD per-team-game frame (with verification)
# ----------------------------------------------------------------------
def build_team_game_frame(raw, season):
    """Pair rows on GAME_ID to attach opponent id + opponent points, derive the
    home flag from MATCHUP, and VERIFY the pairing before trusting it."""
    need = {TEAM_ID_COL, GAME_ID_COL, "PTS", "MATCHUP"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"[{season}] raw log missing columns: {missing}")

    g = raw[[GAME_ID_COL, TEAM_ID_COL, TEAM_ABBR_COL, "PTS", "MATCHUP"]].copy()
    g["PTS"] = pd.to_numeric(g["PTS"], errors="coerce")

    # every game must be exactly two rows
    counts = g.groupby(GAME_ID_COL).size()
    bad = counts[counts != 2]
    if len(bad):
        raise ValueError(f"[{season}] {len(bad)} GAME_IDs without exactly 2 rows "
                         f"(e.g. {list(bad.index[:3])}) — coverage/pull problem")

    # self-join to attach the opponent row
    merged = g.merge(g, on=GAME_ID_COL, suffixes=("", "_opp"))
    merged = merged[merged[TEAM_ID_COL] != merged[f"{TEAM_ID_COL}_opp"]]

    # home flag: MATCHUP is from the row team's perspective, "X @ Y" => away
    merged["is_home"] = (~merged["MATCHUP"].astype(str).str.contains("@")).astype(float)

    frame = pd.DataFrame({
        "season":   season,
        "team_id":  merged[TEAM_ID_COL].astype("int64"),
        "opp_id":   merged[f"{TEAM_ID_COL}_opp"].astype("int64"),
        "team_abbr": merged[TEAM_ABBR_COL],
        "pts":      merged["PTS"].astype(float),
        "opp_pts":  merged["PTS_opp"].astype(float),
        "is_home":  merged["is_home"],
    })
    if frame["pts"].isna().any() or frame["opp_pts"].isna().any():
        raise ValueError(f"[{season}] non-numeric PTS after pairing")
    return frame.reset_index(drop=True)


# ----------------------------------------------------------------------
# SOLVE (points-based offense/defense least squares, one season)
# ----------------------------------------------------------------------
def _transform_margin(m, treatment, cap):
    if treatment == "raw":
        return m
    if treatment == "cap":
        return np.clip(m, -cap, cap)
    if treatment == "tanh":
        return cap * np.tanh(m / cap)
    raise ValueError(f"unknown treatment {treatment!r}")


def solve_massey_season(frame, treatment=MARGIN_TREATMENT, cap=MARGIN_CAP,
                        include_hca=INCLUDE_HCA):
    teams = sorted(set(frame["team_id"]) | set(frame["opp_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    S = frame["pts"].values + frame["opp_pts"].values
    m = frame["pts"].values - frame["opp_pts"].values
    y = (S + _transform_margin(m, treatment, cap)) / 2.0   # treated points-scored

    ncol = 1 + 2 * n + (1 if include_hca else 0)
    X = np.zeros((len(frame), ncol))
    X[:, 0] = 1.0
    ti = frame["team_id"].map(idx).values
    oi = frame["opp_id"].map(idx).values
    X[np.arange(len(frame)), 1 + ti] = 1.0            # own offense
    X[np.arange(len(frame)), 1 + n + oi] = -1.0       # opponent defense suppresses pts
    if include_hca:
        X[:, -1] = frame["is_home"].values

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    o = beta[1:1 + n]
    d = beta[1 + n:1 + 2 * n]
    o = o - o.mean()                                  # canonical mean-zero centering
    d = d - d.mean()
    R = o + d                                         # overall rating (SRS analog)
    hca = float(beta[-1]) if include_hca else np.nan

    mov = (frame.groupby("team_id")
                .apply(lambda x: (x["pts"] - x["opp_pts"]).mean()))
    out = pd.DataFrame({
        "team_id": teams,
        "srs_off": o, "srs_def": d, "srs": R,
    })
    out["mov"] = out["team_id"].map(mov)
    out["sos"] = out["srs"] - out["mov"]              # SRS identity
    out["games"] = out["team_id"].map(frame.groupby("team_id").size())
    out["home_court_adv"] = hca
    return out


# ----------------------------------------------------------------------
# DIAGNOSTICS (verify before trusting)
# ----------------------------------------------------------------------
def run_diagnostics(all_frames, team_stats, treatment, cap):
    print("\n" + "=" * 68)
    print("DIAGNOSTICS")
    print("=" * 68)

    # 1) margin distribution -> informs the cap / tanh scale
    allm = np.abs(np.concatenate([f["pts"].values - f["opp_pts"].values
                                  for f in all_frames.values()]))
    print("\n[margin environment]  |margin| across all games:")
    print("   sd=%.1f  p90=%.0f  p95=%.0f  p99=%.0f  max=%.0f"
          % (allm.std(), np.percentile(allm, 90), np.percentile(allm, 95),
             np.percentile(allm, 99), allm.max()))
    print("   share of games above ±%.0f: %.1f%%"
          % (cap, 100 * (allm > cap).mean()))
    print("   -> data-driven cap ~ p95 (=%.0f); tanh scale ~ p85 (=%.0f) so only"
          % (np.percentile(allm, 95), np.percentile(allm, 85)))
    print("      blowouts saturate while competitive games stay near-linear")

    # 2) sanity: srs should track the existing per-100 net_rating closely
    ratings = pd.concat([solve_massey_season(f, treatment, cap).assign(season=s)
                         for s, f in all_frames.items()], ignore_index=True)
    chk = ratings.merge(team_stats[["season", "team_id", "net_rating"]],
                        on=["season", "team_id"], how="left")
    r = chk[["srs", "net_rating"]].corr().iloc[0, 1]
    print("\n[sanity]  corr(srs, existing net_rating) = %.3f  (expect ~0.9+)" % r)
    print("[sos]     sd(sos)=%.2f   |sos|>2 in %.1f%% of team-seasons"
          % (ratings["sos"].std(), 100 * (ratings["sos"].abs() > 2).mean()))

    # 3) treatment sensitivity: how much do rankings actually move?
    print("\n[treatment sensitivity]  Spearman rank-corr of srs vs 'raw', by treatment:")
    base = pd.concat([solve_massey_season(f, "raw").assign(season=s)
                      for s, f in all_frames.items()], ignore_index=True)
    for tr in ["cap", "tanh"]:
        alt = pd.concat([solve_massey_season(f, tr, cap).assign(season=s)
                         for s, f in all_frames.items()], ignore_index=True)
        j = base.merge(alt, on=["season", "team_id"], suffixes=("_raw", f"_{tr}"))
        # Spearman == Pearson of ranks; compute via ranks so this module stays
        # scipy-free (pandas' method="spearman" imports scipy), matching the
        # rest of the pipeline's no-scipy convention.
        rho = j["srs_raw"].rank().corr(j[f"srs_{tr}"].rank())
        moved = (j["srs_raw"].rank() - j[f"srs_{tr}"].rank()).abs()
        print("   %-5s rho=%.4f   mean|rank shift|=%.2f" % (tr, rho, moved.mean()))
    print("=" * 68 + "\n")
    return ratings


# ----------------------------------------------------------------------
# MERGE into canonical team CSV (idempotent on season + team_id)
# ----------------------------------------------------------------------
def merge_into_team_stats(ratings, team_csv=TEAM_CSV):
    ts = pd.read_csv(team_csv)
    new_cols = ["srs", "srs_off", "srs_def", "sos"]
    ts = ts.drop(columns=[c for c in new_cols if c in ts.columns])  # re-run safe
    ts = ts.merge(ratings[["season", "team_id"] + new_cols],
                  on=["season", "team_id"], how="left")
    if ts["srs"].isna().any():
        n = int(ts["srs"].isna().sum())
        print(f"[warn] {n} team-seasons in {team_csv} have no srs "
              f"(missing game logs?) — left as NaN, not zero")
    ts.to_csv(team_csv, index=False)
    print(f"[write] merged {new_cols} into {team_csv}  ({len(ts)} rows)")


# ----------------------------------------------------------------------
def main():
    all_frames = {}
    for s in SEASONS:
        path = os.path.join(RAW_DIR, f"{s}.json")
        if not os.path.exists(path):
            print(f"[skip] no cache for {s}: {path}")
            continue
        all_frames[s] = build_team_game_frame(load_raw_games(path), s)
        print(f"[ok] {s}: {all_frames[s]['team_id'].nunique():>2} teams, "
              f"{len(all_frames[s])//2:>3} games")

    if not all_frames:
        raise SystemExit("No game logs found — check RAW_DIR.")

    team_stats = pd.read_csv(TEAM_CSV)
    ratings = run_diagnostics(all_frames, team_stats, MARGIN_TREATMENT, MARGIN_CAP)

    ratings["treatment"] = MARGIN_TREATMENT
    ratings["cap"] = MARGIN_CAP
    # attach a readable code from the team file for the audit artifact
    ratings = ratings.merge(team_stats[["season", "team_id", "team_code"]],
                            on=["season", "team_id"], how="left")
    os.makedirs(os.path.dirname(RATINGS_CSV), exist_ok=True)
    cols = ["season", "team_id", "team_code", "games", "mov",
            "srs", "srs_off", "srs_def", "sos", "home_court_adv", "treatment", "cap"]
    ratings[cols].sort_values(["season", "srs"], ascending=[True, False]) \
                 .to_csv(RATINGS_CSV, index=False)
    print(f"[write] {RATINGS_CSV}  ({len(ratings)} team-seasons)")

    merge_into_team_stats(ratings)


if __name__ == "__main__":
    main()
