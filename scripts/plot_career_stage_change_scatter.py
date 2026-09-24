"""Plot career-level regular vs playoff Elo change-rate scatter.

Pure-PIL renderer to avoid optional plotting dependencies.

Outputs:
- career_stage_change_scatter.png        : full-range plot
- career_stage_change_scatter_zoom.png   : zoomed plot using central quantiles
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = PROJECT_ROOT / "results" / "analysis" / "4.2_regular_vs_playoff_shape" / "career_stage_change_significance.csv"
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "4.2_regular_vs_playoff_shape"
OUT_PATH = OUT_DIR / "career_stage_change_scatter.png"
OUT_ZOOM_PATH = OUT_DIR / "career_stage_change_scatter_zoom.png"

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


def draw_plot(df: pd.DataFrame, out_path: Path, lim_x: float, lim_y: float, subtitle: str) -> None:
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

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
        fill=panel,
        outline="#C7C2B8",
        width=2,
    )

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
    # y = x reference inside the visible rectangle
    x1, x2 = -lim_x, lim_x
    y1, y2 = -lim_y, lim_y
    pts = []
    for x in (x1, x2):
        y = x
        if y1 <= y <= y2:
            pts.append((x, y))
    for y in (y1, y2):
        x = y
        if x1 <= x <= x2:
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
            is_sig = bool(getattr(row, "significant_5pct", False))
            if is_sig:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=None)
            else:
                draw.ellipse([x - r, y - r, x + r, y + r], outline=fill, width=2, fill="#FBFAF6")

    draw.text((margin_left, 34), "Career Elo Change Shape: Regular vs Playoffs", font=title_font, fill=ink)
    draw.text(
        (margin_left, 86),
        subtitle,
        font=sub_font,
        fill=muted,
    )

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
    y += 10
    draw.text((legend_x, y), "Significance", font=bold_font, fill=ink)
    y += 38
    draw.ellipse([legend_x, y + 4, legend_x + 16, y + 20], fill="#111827")
    draw.text((legend_x + 28, y), "Significant (filled)", font=small_font, fill=ink)
    y += 30
    draw.ellipse([legend_x, y + 4, legend_x + 16, y + 20], outline="#111827", width=2, fill="#FBFAF6")
    draw.text((legend_x + 28, y), "Not significant (hollow)", font=small_font, fill=ink)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df["class4"] = df.apply(classify, axis=1)

    lim_x = float(df["regular_change_rate"].abs().max() * 1.08)
    lim_y = float(df["playoff_change_rate"].abs().max() * 1.08)
    draw_plot(
        df,
        OUT_PATH,
        lim_x,
        lim_y,
        "Canonical pk2, gates: regular >= 100 games, playoffs >= 5 games (full range)",
    )

    qx = float(df["regular_change_rate"].abs().quantile(0.98) * 1.1)
    qy = float(df["playoff_change_rate"].abs().quantile(0.98) * 1.1)
    draw_plot(
        df,
        OUT_ZOOM_PATH,
        qx,
        qy,
        "Canonical pk2, percentage axes, zoomed to the central 98% range",
    )
    print(OUT_PATH)
    print(OUT_ZOOM_PATH)


if __name__ == "__main__":
    main()
