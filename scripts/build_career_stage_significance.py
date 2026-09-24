"""Build significance layer for career regular-vs-playoff change analysis.

This computes a per-player stage-difference significance screen on the
single-game relative Elo change metric:

    rel_delta = delta_adj / rating_before

Effect:
    playoff_mean_rel_delta - regular_mean_rel_delta

Significance:
    normal-approximation two-sided test on the mean difference with unequal
    variances (Welch-style standard error). This is intended as a scalable
    screening layer for all players, not as a final inferential claim.

Outputs:
    results/analysis/4.2_regular_vs_playoff_shape/career_stage_change_significance.csv
    results/analysis/4.2_regular_vs_playoff_shape/career_stage_change_significance_summary.txt
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
DB_PATH = RESULTS / "nba_elo.db"
OUT_DIR = RESULTS / "analysis" / "4.2_regular_vs_playoff_shape"


def normal_p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=OUT_DIR / "career_stage_change.csv")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    base = pd.read_csv(args.input)
    keep_ids = tuple(int(x) for x in base["player_id"].tolist())

    con = sqlite3.connect(DB_PATH)
    try:
        q = f"""
            SELECT p.full_name,
                   vt.player_id,
                   vt.game_track,
                   vt.delta_adj,
                   vt.rating_before
            FROM variant_timeline vt
            JOIN players p
              ON p.player_id = vt.player_id
            WHERE vt.variant_id = 'canonical_pk2'
              AND vt.player_id IN ({",".join("?" for _ in keep_ids)})
        """
        df = pd.read_sql_query(q, con, params=list(keep_ids))
    finally:
        con.close()

    df["rel_delta"] = df["delta_adj"] / df["rating_before"]

    rows = []
    for (player_id, full_name), g in df.groupby(["player_id", "full_name"], sort=False):
        reg = g.loc[g["game_track"].eq("regular"), "rel_delta"]
        po = g.loc[g["game_track"].eq("playoffs"), "rel_delta"]
        if reg.empty or po.empty:
            continue
        m_reg = float(reg.mean())
        m_po = float(po.mean())
        v_reg = float(reg.var(ddof=1)) if len(reg) > 1 else 0.0
        v_po = float(po.var(ddof=1)) if len(po) > 1 else 0.0
        se = math.sqrt(v_reg / len(reg) + v_po / len(po))
        effect = m_po - m_reg
        if se == 0:
            z = 0.0
            pval = 1.0
        else:
            z = effect / se
            pval = normal_p_two_sided(z)
        rows.append(
            {
                "player_id": int(player_id),
                "full_name": full_name,
                "regular_rate_pct": m_reg * 100.0,
                "playoff_rate_pct": m_po * 100.0,
                "effect_pct": effect * 100.0,
                "z_score": z,
                "p_value": pval,
                "significant_5pct": bool(pval < 0.05),
            }
        )

    sig = pd.DataFrame(rows)
    out = base.merge(sig, on=["player_id", "full_name"], how="left")
    out["significance_label"] = out["significant_5pct"].map(
        lambda v: "significant" if bool(v) else "not_significant"
    )
    out["distance_from_origin"] = (out["regular_change_rate"] ** 2 + out["playoff_change_rate"] ** 2) ** 0.5

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = OUT_DIR / f"career_stage_change_significance{suffix}.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_lines = [
        f"players={len(out)}",
        f"significant_5pct={(out['significant_5pct'] == True).sum()}",
        f"not_significant={(out['significant_5pct'] != True).sum()}",
        "",
        "quadrant x significance:",
        out.groupby(["quadrant", "significance_label"]).size().to_string(),
        "",
        "furthest_from_origin:",
        out.nlargest(20, "distance_from_origin")[
            [
                "full_name",
                "regular_change_rate",
                "playoff_change_rate",
                "playoff_minus_regular",
                "quadrant",
                "significance_label",
                "distance_from_origin",
            ]
        ].to_string(index=False),
    ]
    summary_path = OUT_DIR / f"career_stage_change_significance_summary{suffix}.txt"
    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print(out_csv)
    print(summary_path)
    print(f"significant={(out['significant_5pct'] == True).sum()} / {len(out)}")


if __name__ == "__main__":
    main()
