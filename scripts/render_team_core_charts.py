"""Render season-average team core charts for presentation.

Outputs to results/analysis/4.3_team_strength_ideas:
- seasonavg_core_rankings_pk1.png
- seasonavg_core_player_frequency_pk1.png
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "4.3_team_strength_ideas"


def fmt_entry(season_label: str, team_name: str, value: float) -> str:
    return f"{season_label} {team_name} ({value:.1f})"


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def render_rankings_table() -> Path:
    top2 = pd.read_csv(OUT_DIR / "top_seasonavg_top2_seasons_pk1.csv").head(15)
    top3 = pd.read_csv(OUT_DIR / "top_seasonavg_top3_seasons_pk1.csv").head(15)
    top5 = pd.read_csv(OUT_DIR / "top_seasonavg_top5_seasons_pk1.csv").head(15)

    rankings = pd.DataFrame(
        {
            "rank": range(1, 16),
            "top2_entry": [fmt_entry(r.season_label, r.team_name, r.seasonavg_top2_elo) for r in top2.itertuples(index=False)],
            "top3_entry": [fmt_entry(r.season_label, r.team_name, r.seasonavg_top3_elo) for r in top3.itertuples(index=False)],
            "top5_entry": [fmt_entry(r.season_label, r.team_name, r.seasonavg_top5_elo) for r in top5.itertuples(index=False)],
        }
    )

    width, height = 2250, 1700
    margin = 70
    col_w = 650
    row_h = 86
    header_h = 108
    bg = "#F7F3EA"
    panel = "#FBFAF6"
    ink = "#111827"
    muted = "#4B5563"
    line = "#D1D5DB"
    accent = "#8B1E3F"

    title_font = load_font("arialbd.ttf", 50)
    head_font = load_font("arialbd.ttf", 34)
    body_font = load_font("arial.ttf", 27)
    rank_font = load_font("arialbd.ttf", 30)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    draw.text((margin, 28), "Season-Average Core Elo: Canonical pk1", font=title_font, fill=ink)
    draw.text(
        (margin, 82),
        "Combined regular season + playoffs; stable core-configuration view",
        font=body_font,
        fill=muted,
    )
    top = 140
    left = margin
    table_w = width - margin * 2
    table_h = height - top - margin
    draw.rounded_rectangle([left, top, left + table_w, top + table_h], radius=18, fill=panel, outline="#C7C2B8", width=2)

    columns = [("Rank", 120), ("Top-2", col_w), ("Top-3", col_w), ("Top-5", col_w)]
    x = left + 20
    y = top + 18
    for label, w in columns:
        fill = accent if label != "Rank" else ink
        draw.text((x + 8, y + 16), label, font=head_font, fill=fill)
        x += w
    draw.line([left + 18, y + header_h, left + table_w - 18, y + header_h], fill=line, width=2)

    for i, row in rankings.iterrows():
        row_y = y + header_h + i * row_h
        if i > 0:
            draw.line([left + 18, row_y, left + table_w - 18, row_y], fill=line, width=1)
        x = left + 20
        draw.text((x + 12, row_y + 20), str(int(row["rank"])), font=rank_font, fill=ink)
        x += 120
        for key in ["top2_entry", "top3_entry", "top5_entry"]:
            draw.text((x + 8, row_y + 20), str(row[key]), font=body_font, fill=ink)
            x += col_w

    out_path = OUT_DIR / "seasonavg_core_rankings_pk1.png"
    img.save(out_path, quality=95)
    return out_path


def render_player_frequency_chart() -> Path:
    frames = []
    for n in [2, 3, 5]:
        df = pd.read_csv(OUT_DIR / f"top_seasonavg_top{n}_seasons_pk1.csv").head(20).copy()
        df["slot"] = f"Top-{n}"
        df["players"] = df[f"top{n}_players"].str.split(" / ")
        frames.append(df[["season_label", "team_name", "slot", "players"]])
    combined = pd.concat(frames, ignore_index=True)

    counts: Counter[str] = Counter()
    slot_counts: Counter[tuple[str, str]] = Counter()
    for row in combined.itertuples(index=False):
        seen = set(row.players)
        for player in seen:
            counts[player] += 1
            slot_counts[(player, row.slot)] += 1

    top_players = counts.most_common(12)

    width, height = 1800, 1180
    margin = 90
    bg = "#F7F3EA"
    panel = "#FBFAF6"
    ink = "#111827"
    muted = "#4B5563"
    line = "#D1D5DB"
    bar = "#C76B50"
    bar2 = "#D9A441"
    bar3 = "#5C8D89"
    accent = "#8B1E3F"

    title_font = load_font("arialbd.ttf", 48)
    sub_font = load_font("arial.ttf", 28)
    label_font = load_font("arialbd.ttf", 30)
    body_font = load_font("arial.ttf", 26)
    small_font = load_font("arial.ttf", 22)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([40, 40, width - 40, height - 40], radius=24, fill=panel, outline="#C7C2B8", width=2)
    draw.text((margin, 72), "Who Appears Most In Historical Core Tables?", font=title_font, fill=ink)
    draw.text(
        (margin, 128),
        "Counts from Top-20 season-average Top-2 / Top-3 / Top-5 lists; one count per player per team-season entry",
        font=sub_font,
        fill=muted,
    )

    bar_left = 520
    bar_top = 230
    row_h = 66
    max_count = top_players[0][1] if top_players else 1
    scale = 920 / max_count

    legend_y = 176
    draw.rectangle([width - 540, legend_y, width - 512, legend_y + 28], fill=bar)
    draw.text((width - 500, legend_y - 2), "Top-2", font=small_font, fill=muted)
    draw.rectangle([width - 380, legend_y, width - 352, legend_y + 28], fill=bar2)
    draw.text((width - 340, legend_y - 2), "Top-3", font=small_font, fill=muted)
    draw.rectangle([width - 220, legend_y, width - 192, legend_y + 28], fill=bar3)
    draw.text((width - 180, legend_y - 2), "Top-5", font=small_font, fill=muted)

    for idx, (player, total) in enumerate(top_players):
        y = bar_top + idx * row_h
        t2 = slot_counts.get((player, "Top-2"), 0)
        t3 = slot_counts.get((player, "Top-3"), 0)
        t5 = slot_counts.get((player, "Top-5"), 0)

        draw.text((margin, y + 8), f"{idx + 1}", font=label_font, fill=accent if idx == 0 else ink)
        draw.text((margin + 55, y + 8), player, font=body_font, fill=accent if player == "LeBron James" else ink)

        x = bar_left
        for value, color in [(t2, bar), (t3, bar2), (t5, bar3)]:
            if value > 0:
                w = value * scale
                draw.rounded_rectangle([x, y + 10, x + w, y + 46], radius=9, fill=color)
                label_x = x + w / 2
                label_fill = "#FBFAF6"
                draw.text(
                    (label_x, y + 28),
                    str(value),
                    font=small_font,
                    fill=label_fill,
                    anchor="mm",
                )
                x += w
        draw.text((bar_left + total * scale + 18, y + 8), str(total), font=label_font, fill=ink)
        draw.line([margin, y + 58, width - margin, y + 58], fill=line, width=1)

    note = "LeBron appears across multiple eras and team contexts, which is why he overwhelms this view."
    draw.text((margin, height - 108), note, font=small_font, fill=muted)

    out_path = OUT_DIR / "seasonavg_core_player_frequency_pk1.png"
    img.save(out_path, quality=95)
    return out_path


def main() -> None:
    render_rankings_table()
    render_player_frequency_chart()


if __name__ == "__main__":
    main()
