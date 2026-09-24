from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "analysis" / "4.3_career_curve_clusters" / "final_40_or_8_rule"
CAREERS = BASE / "player_career_trajectory_classification.csv"
OUTPUT = BASE / "late_peak_share_by_debut_year_relative.png"
DATA_OUTPUT = BASE / "late_peak_share_by_debut_year_relative.csv"


def main() -> None:
    players = pd.read_csv(CAREERS, usecols=["player_id", "first_season", "cluster"])

    rows = []
    for year in range(1978, 2015):
        cohort = players[players["first_season"].between(year - 2, year + 2)]
        rows.append(
            {
                "center_debut_year": year,
                "players_in_5y_window": len(cohort),
                "late_peak_share_5y": (cohort["cluster"] == 3).mean(),
            }
        )
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

    left, top, right, bottom = [v * scale for v in (120, 230, 1690, 790)]
    draw.text((left, 42 * scale), "Late-peak players became more common after the late 1990s", fill="#111827", font=title_font)
    draw.text(
        (left, 101 * scale),
        "Share of eligible players in the late-peak / sustained-rise group, using centered five-year debut cohorts",
        fill="#4B5563",
        font=subtitle_font,
    )

    x_min, x_max = 1978, 2014
    y_min, y_max = 0.0, 0.50

    def xy(year: float, share: float) -> tuple[int, int]:
        x = left + (year - x_min) / (x_max - x_min) * (right - left)
        y = bottom - (share - y_min) / (y_max - y_min) * (bottom - top)
        return int(x), int(y)

    for share in np.arange(0, 0.51, 0.1):
        _, y = xy(x_min, float(share))
        draw.line((left, y, right, y), fill="#D1D5DB", width=1 * scale)
        label = f"{share * 100:.0f}%"
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((left - (box[2] - box[0]) - 13 * scale, y - 9 * scale), label, fill="#6B7280", font=small_font)

    for year in range(1980, 2015, 5):
        x, _ = xy(year, 0)
        draw.line((x, top, x, bottom), fill="#E5E7EB", width=1 * scale)
        label = str(year)
        box = draw.textbbox((0, 0), label, font=body_font)
        draw.text((x - (box[2] - box[0]) / 2, bottom + 18 * scale), label, fill="#4B5563", font=body_font)

    trend_points = [xy(float(row.center_debut_year), float(row.late_peak_share_5y)) for row in trend.itertuples()]
    draw.line(trend_points, fill="#2E8B57", width=7 * scale, joint="curve")
    for x, y in trend_points:
        radius = 4 * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#2E8B57")

    marker_year = 1997
    marker_share = float(trend.loc[trend["center_debut_year"] == marker_year, "late_peak_share_5y"].iloc[0])
    marker_x, marker_y = xy(marker_year, marker_share)
    draw.line((marker_x, top, marker_x, bottom), fill=(107, 114, 128, 135), width=2 * scale)
    radius = 7 * scale
    draw.ellipse((marker_x - radius, marker_y - radius, marker_x + radius, marker_y + radius), fill="#2E8B57")
    annotation = "Late-1990s low"
    draw.text((marker_x + 14 * scale, marker_y - 38 * scale), annotation, fill="#374151", font=bold_font)

    end = trend.iloc[-1]
    end_x, end_y = xy(float(end["center_debut_year"]), float(end["late_peak_share_5y"]))
    end_label = f"{end['late_peak_share_5y']:.0%}"
    draw.text((end_x - 4 * scale, end_y - 35 * scale), end_label, fill="#2E8B57", font=bold_font)

    draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2 * scale)
    x_label = "NBA debut year"
    box = draw.textbbox((0, 0), x_label, font=bold_font)
    draw.text(((left + right - (box[2] - box[0])) / 2, bottom + 68 * scale), x_label, fill="#374151", font=bold_font)

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(OUTPUT, quality=95)


if __name__ == "__main__":
    main()
