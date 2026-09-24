"""Render presentation-first quadrant-only scatter plots.

Outputs:
- career_stage_change_scatter_quadrant_labeled.png
- career_stage_change_scatter_quadrant_kd.png
- career_stage_change_scatter_quadrant_extremes.png
- career_stage_change_scatter_quadrant_plain.png
- career_stage_change_scatter_quadrant_set_a.png
- career_stage_change_scatter_quadrant_set_b.png
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = PROJECT_ROOT / "results" / "analysis" / "4.2_regular_vs_playoff_shape" / "career_stage_change_significance_r200_p15.csv"
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "4.2_regular_vs_playoff_shape"
OUT_LABELED = OUT_DIR / "career_stage_change_scatter_quadrant_labeled.png"
OUT_KD = OUT_DIR / "career_stage_change_scatter_quadrant_kd.png"
OUT_EXTREMES = OUT_DIR / "career_stage_change_scatter_quadrant_extremes.png"
OUT_PLAIN = OUT_DIR / "career_stage_change_scatter_quadrant_plain.png"
OUT_SET_A = OUT_DIR / "career_stage_change_scatter_quadrant_set_a.png"
OUT_SET_B = OUT_DIR / "career_stage_change_scatter_quadrant_set_b.png"

COLORS = {
    "all_phase_riser": "#2E8B57",
    "playoff_reversal": "#D97706",
    "all_phase_decliner": "#6B7280",
    "playoff_penalty": "#C2410C",
}

LABELS = {
    "all_phase_riser": "All-Phase Riser",
    "playoff_reversal": "Playoff Reversal",
    "all_phase_decliner": "All-Phase Decliner",
    "playoff_penalty": "Playoff Penalty",
}

SET_A = {
    "Joel Embiid": {"dx": -225, "dy": -22},
    "David Robinson": {"dx": 18, "dy": 42},
    "Michael Jordan": {"dx": 18, "dy": -14},
    "LeBron James": {"dx": -205, "dy": -22},
    "Tim Duncan": {"dx": 18, "dy": 16},
    "Shaquille O'Neal": {"dx": 18, "dy": -38},
}

SET_B = {
    "James Harden": {"dx": 18, "dy": -34},
    "Reggie Miller": {"dx": 18, "dy": -42},
    "Draymond Green": {"dx": 18, "dy": -6},
}

HIGHLIGHTS = {
    "Joel Embiid": {"dx": -225, "dy": -22},
    "Tim Duncan": {"dx": 18, "dy": 16},
    "David Robinson": {"dx": 18, "dy": 42},
    "Michael Jordan": {"dx": 18, "dy": -14},
    "James Harden": {"dx": 18, "dy": -34},
    "Baron Davis": {"dx": 18, "dy": -78},
    "Reggie Miller": {"dx": 18, "dy": -42},
    "Draymond Green": {"dx": 18, "dy": -6},
}

KD_ONLY = {
    "Kevin Durant": {"dx": 18, "dy": -24},
}

EXTREMES_ONLY = {
    "Jalen Johnson": {"dx": -230, "dy": -8},
    "Anthony Roberts": {"dx": 18, "dy": -24},
    "Mickey Johnson": {"dx": 18, "dy": -34},
    "Phil Ford": {"dx": 18, "dy": 10},
}

SHORT = {
    "Joel Embiid": "Embiid",
    "Tim Duncan": "Duncan",
    "David Robinson": "Robinson",
    "Michael Jordan": "Jordan",
    "Shai Gilgeous-Alexander": "SGA",
    "LeBron James": "LeBron",
    "Shaquille O'Neal": "Shaq",
    "James Harden": "Harden",
    "Baron Davis": "Baron Davis",
    "Reggie Miller": "Reggie",
    "Draymond Green": "Draymond",
    "Rudy Gay": "Rudy Gay",
    "Kevin Durant": "KD",
    "Jalen Johnson": "Reg+ Jalen Johnson",
    "Anthony Roberts": "Reg- Anthony Roberts",
    "Mickey Johnson": "PO+ Mickey Johnson",
    "Phil Ford": "PO- Phil Ford",
}


def classify(row: pd.Series) -> str:
    x = float(row["regular_change_rate"])
    y = float(row["playoff_change_rate"])
    if x >= 0 and y >= 0:
        return "all_phase_riser"
    if x < 0 and y >= 0:
        return "playoff_reversal"
    if x < 0 and y < 0:
        return "all_phase_decliner"
    return "playoff_penalty"


def draw_plot(
    df: pd.DataFrame,
    highlights: dict[str, dict[str, int]],
    out_path: Path,
    subtitle: str,
) -> None:
    width, height = 1800, 1400
    margin_left, margin_right = 160, 280
    margin_top, margin_bottom = 160, 150
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    bg = "#F7F3EA"
    panel = "#FBFAF6"
    ink = "#111827"
    muted = "#4B5563"
    grid = "#D1D5DB"

    font_dir = Path("C:/Windows/Fonts")
    title_font = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 42)
    sub_font = ImageFont.truetype(str(font_dir / "arial.ttf"), 22)
    label_font = ImageFont.truetype(str(font_dir / "arial.ttf"), 24)
    small_font = ImageFont.truetype(str(font_dir / "arial.ttf"), 18)
    bold_font = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 24)
    name_font = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 20)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
        fill=panel,
        outline="#C7C2B8",
        width=2,
    )

    lim_x = float(df["regular_change_rate"].abs().quantile(0.99) * 1.15)
    lim_y = float(df["playoff_change_rate"].abs().quantile(0.995) * 1.15)
    highlight_df = df[df["full_name"].isin(highlights)]
    if not highlight_df.empty:
        lim_x = max(lim_x, float(highlight_df["regular_change_rate"].abs().max() * 1.05))
        lim_y = max(lim_y, float(highlight_df["playoff_change_rate"].abs().max() * 1.05))

    def xmap(v: float) -> float:
        return margin_left + (v + lim_x) / (2 * lim_x) * plot_w

    def ymap(v: float) -> float:
        return margin_top + plot_h - ((v + lim_y) / (2 * lim_y) * plot_h)

    for frac in [0.25, 0.5, 0.75]:
        x = margin_left + plot_w * frac
        y = margin_top + plot_h * frac
        draw.line([x, margin_top, x, margin_top + plot_h], fill=grid, width=1)
        draw.line([margin_left, y, margin_left + plot_w, y], fill=grid, width=1)

    x0 = xmap(0.0)
    y0 = ymap(0.0)
    draw.line([margin_left, y0, margin_left + plot_w, y0], fill=ink, width=3)
    draw.line([x0, margin_top, x0, margin_top + plot_h], fill=ink, width=3)

    pts = []
    for x in (-lim_x, lim_x):
        y = x
        if -lim_y <= y <= lim_y:
            pts.append((x, y))
    for y in (-lim_y, lim_y):
        x = y
        if -lim_x <= x <= lim_x:
            pts.append((x, y))
    uniq = []
    for p in pts:
        if p not in uniq:
            uniq.append(p)
    if len(uniq) >= 2:
        a, b = uniq[0], uniq[1]
        draw.line([xmap(a[0]), ymap(a[1]), xmap(b[0]), ymap(b[1])], fill="#64748B", width=2)

    for klass in ["all_phase_riser", "playoff_reversal", "all_phase_decliner", "playoff_penalty"]:
        sub = df[df["class4"] == klass]
        fill = COLORS[klass]
        for row in sub.itertuples(index=False):
            xv = float(row.regular_change_rate)
            yv = float(row.playoff_change_rate)
            if xv < -lim_x or xv > lim_x or yv < -lim_y or yv > lim_y:
                continue
            x = xmap(xv)
            y = ymap(yv)
            r = 5
            draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=None)

    draw.text((margin_left, 34), "Career Elo Change Shape: Regular vs Playoffs", font=title_font, fill=ink)
    draw.text((margin_left, 86), subtitle, font=sub_font, fill=muted)
    draw.text((margin_left + plot_w / 2 - 150, height - 74), "Regular Change Rate (%)", font=label_font, fill=ink)
    draw.text((30, margin_top + plot_h / 2 - 20), "Playoff Change Rate (%)", font=label_font, fill=ink)

    draw.text((xmap(lim_x * 0.70), ymap(lim_y * 0.92)), "Q1", font=bold_font, fill=muted)
    draw.text((xmap(-lim_x * 0.88), ymap(lim_y * 0.92)), "Q2", font=bold_font, fill=muted)
    draw.text((xmap(-lim_x * 0.88), ymap(-lim_y * 0.90)), "Q3", font=bold_font, fill=muted)
    draw.text((xmap(lim_x * 0.70), ymap(-lim_y * 0.90)), "Q4", font=bold_font, fill=muted)
    draw.text((xmap(min(lim_x, lim_y) * 0.55), ymap(min(lim_x, lim_y) * 0.58)), "y = x", font=small_font, fill="#64748B")
    draw.text((x0 + 8, y0 + 8), "origin (0,0)", font=small_font, fill=ink)

    tick_x = [-lim_x, -lim_x / 2, 0.0, lim_x / 2, lim_x]
    tick_y = [-lim_y, -lim_y / 2, 0.0, lim_y / 2, lim_y]
    for v in tick_x:
        xv = xmap(v)
        draw.line([xv, y0 - 8, xv, y0 + 8], fill=ink, width=2)
        draw.text((xv - 34, y0 + 14), f"{v*100:.2f}%", font=small_font, fill=muted)
    for v in tick_y:
        yv = ymap(v)
        draw.line([x0 - 8, yv, x0 + 8, yv], fill=ink, width=2)
        draw.text((x0 + 14, yv - 10), f"{v*100:.2f}%", font=small_font, fill=muted)

    legend_x = width - margin_right + 28
    legend_y = margin_top + 30
    draw.text((legend_x, legend_y), "Classes", font=bold_font, fill=ink)
    y = legend_y + 48
    for klass in ["all_phase_riser", "playoff_reversal", "all_phase_decliner", "playoff_penalty"]:
        draw.ellipse([legend_x, y + 4, legend_x + 16, y + 20], fill=COLORS[klass])
        label = f"{LABELS[klass]} ({(df['class4'] == klass).sum()})"
        draw.text((legend_x + 28, y), label, font=small_font, fill=ink)
        y += 34

    for name, cfg in highlights.items():
        sub = df[df["full_name"].eq(name)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        xv = float(row["regular_change_rate"])
        yv = float(row["playoff_change_rate"])
        if xv < -lim_x or xv > lim_x or yv < -lim_y or yv > lim_y:
            continue
        x = xmap(xv)
        y = ymap(yv)
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], outline="#111827", width=2)
        dx, dy = cfg["dx"], cfg["dy"]
        bx1 = x + dx
        by1 = y + dy
        bx2 = bx1 + 190
        by2 = by1 + 50
        bx1 = max(margin_left + 6, min(bx1, width - margin_right - 210))
        by1 = max(margin_top + 6, min(by1, margin_top + plot_h - 60))
        bx2 = bx1 + 190
        by2 = by1 + 50
        draw.line([x, y, bx1, by1 + 20], fill="#111827", width=2)
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=10, fill="#FFF7ED", outline=COLORS[row["class4"]], width=2)
        draw.text((bx1 + 10, by1 + 7), SHORT.get(name, name), font=name_font, fill=ink)
        draw.text((bx1 + 10, by1 + 28), row["quadrant"], font=small_font, fill=muted)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df["class4"] = df.apply(classify, axis=1)

    draw_plot(
        df,
        {},
        OUT_PLAIN,
        "Canonical pk2, stricter gate: regular >= 200 games, playoffs >= 15 games; pure quadrant map",
    )
    draw_plot(
        df,
        SET_A,
        OUT_SET_A,
        "Star anchors: Embiid, Robinson, Jordan, LeBron, Duncan, Shaq",
    )
    draw_plot(
        df,
        SET_B,
        OUT_SET_B,
        "Contrast cases: Harden, Reggie, and Draymond",
    )
    draw_plot(
        df,
        HIGHLIGHTS,
        OUT_LABELED,
        "Canonical pk2, stricter gate: regular >= 200 games, playoffs >= 15 games; pure quadrant map",
    )
    draw_plot(
        df,
        KD_ONLY,
        OUT_KD,
        "KD sits on the Q4 side here, but close to the axis; this motivates a future neutral / 9-class extension",
    )
    draw_plot(
        df,
        EXTREMES_ONLY,
        OUT_EXTREMES,
        "Current video gate full-sample axis extremes: biggest regular-season rise/drop and playoff rise/drop",
    )
    print(OUT_LABELED)
    print(OUT_KD)
    print(OUT_EXTREMES)
    print(OUT_PLAIN)
    print(OUT_SET_A)
    print(OUT_SET_B)


if __name__ == "__main__":
    main()
