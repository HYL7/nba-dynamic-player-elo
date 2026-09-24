from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "analysis" / "4.3_career_curve_clusters" / "final_40_or_8_rule"
ORIGINAL = BASE / "player_career_trajectory_classification.csv"
RAW = BASE / "player_career_trajectory_classification_raw_elo.csv"
OUTPUT = BASE / "career_trajectory_share_by_debut_year.png"
DATA_OUTPUT = BASE / "career_trajectory_share_by_debut_year.csv"

COLORS = {1: "#C2410C", 2: "#2563EB", 3: "#2E8B57"}
NAMES = {
    1: "Early peak / gradual decline",
    2: "Conventional mid-career peak",
    3: "Late peak / sustained rise",
}


def main() -> None:
    original = pd.read_csv(ORIGINAL, usecols=["player_id", "first_season"])
    raw = pd.read_csv(RAW, usecols=["player_id", "raw_elo_cluster"])
    players = original.merge(raw, on="player_id", how="inner")

    years = np.arange(int(players["first_season"].min()), 2015)
    rows = []
    for year in years:
        cohort = players[players["first_season"].between(year - 2, year + 2)]
        row = {"center_debut_year": year, "players_in_5y_window": len(cohort)}
        for cluster in range(1, 4):
            row[f"cluster_{cluster}_share"] = (cohort["raw_elo_cluster"] == cluster).mean()
        rows.append(row)
    trend = pd.DataFrame(rows)
    trend.to_csv(DATA_OUTPUT, index=False)

    scale = 2
    width, height = 1800 * scale, 960 * scale
    image = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(image, "RGBA")
    font_dir = Path("C:/Windows/Fonts")

    def font(name: str, size: int):
        return ImageFont.truetype(str(font_dir / name), size * scale)

    title_font = font("arialbd.ttf", 42)
    subtitle_font = font("arial.ttf", 19)
    body_font = font("arial.ttf", 18)
    small_font = font("arial.ttf", 15)
    bold_font = font("arialbd.ttf", 17)

    left, top, right, bottom = [v * scale for v in (120, 255, 1690, 790)]
    draw.text((left, 42 * scale), "Career trajectory shares by NBA debut year", fill="#111827", font=title_font)
    draw.text(
        (left, 101 * scale),
        "Shares are calculated within centered five-year debut cohorts; post-2014 cohorts are omitted because too few careers are complete",
        fill="#4B5563",
        font=subtitle_font,
    )

    legend_x = [120, 650, 1190]
    for cluster, x in zip(range(1, 4), legend_x):
        color = COLORS[cluster]
        draw.line((x * scale, 185 * scale, (x + 42) * scale, 185 * scale), fill=color, width=6 * scale)
        draw.text(((x + 55) * scale, 173 * scale), NAMES[cluster], fill="#374151", font=body_font)

    x_min, x_max = int(years.min()), int(years.max())
    max_share = max(float(trend[f"cluster_{cluster}_share"].max()) for cluster in range(1, 4))
    y_max = max(0.6, np.ceil((max_share + 0.03) * 10) / 10)

    def xy(year: int, share: float) -> tuple[int, int]:
        x = left + (year - x_min) / (x_max - x_min) * (right - left)
        y = bottom - share / y_max * (bottom - top)
        return int(x), int(y)

    for share in np.arange(0, y_max + 0.001, 0.1):
        _, y = xy(x_min, float(share))
        draw.line((left, y, right, y), fill="#D1D5DB", width=1 * scale)
        label = f"{share * 100:.0f}%"
        label_box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((left - (label_box[2] - label_box[0]) - 13 * scale, y - 9 * scale), label, fill="#6B7280", font=small_font)

    for year in range(1980, 2015, 5):
        x, _ = xy(year, 0)
        draw.line((x, top, x, bottom), fill="#E5E7EB", width=1 * scale)
        label = str(year)
        label_box = draw.textbbox((0, 0), label, font=body_font)
        draw.text((x - (label_box[2] - label_box[0]) / 2, bottom + 18 * scale), label, fill="#4B5563", font=body_font)

    for cluster in range(1, 4):
        shares = trend[f"cluster_{cluster}_share"].to_numpy(float)
        points = [xy(int(year), float(share)) for year, share in zip(years, shares)]
        draw.line(points, fill=COLORS[cluster], width=6 * scale, joint="curve")
        for idx in range(0, len(points), 5):
            x, y = points[idx]
            radius = 4 * scale
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=COLORS[cluster])

    draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2 * scale)
    x_label = "Center year of five-year NBA debut cohort"
    x_box = draw.textbbox((0, 0), x_label, font=bold_font)
    draw.text(((left + right - (x_box[2] - x_box[0])) / 2, bottom + 68 * scale), x_label, fill="#374151", font=bold_font)

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(OUTPUT, quality=95)


if __name__ == "__main__":
    main()
