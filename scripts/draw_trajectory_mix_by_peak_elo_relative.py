from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "analysis" / "4.3_career_curve_clusters" / "final_40_or_8_rule"
SOURCE = BASE / "cluster_characteristics_player_level.csv"
OUTPUT = BASE / "career_trajectory_mix_by_peak_elo_relative.png"
DATA_OUTPUT = BASE / "career_trajectory_mix_by_peak_elo_relative.csv"

COLORS = {1: "#C2410C", 2: "#2563EB", 3: "#2E8B57"}
NAMES = {1: "Early peak", 2: "Mid-career peak", 3: "Late peak"}
LEVEL_DESCRIPTIONS = {
    "2,000+": "Top 10 in the league",
    "1,800–1,999": "All-Star to high-level starter",
    "1,600–1,799": "Regular starter to rotation player",
    "Below 1,600": "Back-end rotation player",
}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def main() -> None:
    data = pd.read_csv(SOURCE)
    labels = ["2,000+", "1,800–1,999", "1,600–1,799", "Below 1,600"]
    conditions = [
        data["raw_peak_elo"] >= 2000,
        data["raw_peak_elo"].between(1800, 2000, inclusive="left"),
        data["raw_peak_elo"].between(1600, 1800, inclusive="left"),
        data["raw_peak_elo"] < 1600,
    ]
    data["peak_elo_level"] = np.select(conditions, labels, default="")
    counts = pd.crosstab(data["peak_elo_level"], data["cluster"]).reindex(labels).fillna(0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    out = counts.copy()
    out.columns = [f"cluster_{int(column)}_players" for column in out.columns]
    for cluster in range(1, 4):
        out[f"cluster_{cluster}_share"] = shares[cluster]
    out.insert(0, "players", counts.sum(axis=1).astype(int))
    out.to_csv(DATA_OUTPUT)

    scale = 2
    width, height = 1800 * scale, 900 * scale
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

    left, top, right, bottom = [v * scale for v in (430, 245, 1680, 750)]
    draw.text((105 * scale, 42 * scale), "Career trajectory mix by career-high Elo", fill="#111827", font=title_font)
    draw.text(
        (105 * scale, 101 * scale),
        "Each bar shows the trajectory distribution within one peak-Elo level",
        fill="#4B5563",
        font=subtitle_font,
    )

    legend_x = [105, 520, 1025]
    for cluster, x in zip(range(1, 4), legend_x):
        draw.rectangle((x * scale, 174 * scale, (x + 28) * scale, 194 * scale), fill=COLORS[cluster])
        draw.text(((x + 40) * scale, 170 * scale), NAMES[cluster], fill="#374151", font=body_font)

    bar_height = 72 * scale
    y_positions = np.linspace(top + 25 * scale, bottom - 25 * scale, len(labels))
    for label, y in zip(labels, y_positions):
        total = int(counts.loc[label].sum())
        label_text = f"{label}  (n={total})"
        label_box = draw.textbbox((0, 0), label_text, font=bold_font)
        label_x = left - (label_box[2] - label_box[0]) - 20 * scale
        draw.text((label_x, y - 25 * scale), label_text, fill="#374151", font=bold_font)
        description = LEVEL_DESCRIPTIONS[label]
        description_box = draw.textbbox((0, 0), description, font=small_font)
        draw.text(
            (left - (description_box[2] - description_box[0]) - 20 * scale, y + 2 * scale),
            description,
            fill="#6B7280",
            font=small_font,
        )
        x = left
        for cluster in range(1, 4):
            share = float(shares.loc[label, cluster])
            segment_width = share * (right - left)
            draw.rectangle((x, y - bar_height / 2, x + segment_width, y + bar_height / 2), fill=COLORS[cluster])
            pct = f"{share * 100:.1f}%"
            pct_box = draw.textbbox((0, 0), pct, font=bold_font)
            if segment_width > (pct_box[2] - pct_box[0]) + 18 * scale:
                draw.text(
                    (x + (segment_width - (pct_box[2] - pct_box[0])) / 2, y - 10 * scale),
                    pct,
                    fill="#FFFFFF",
                    font=bold_font,
                )
            x += segment_width

    for pct in [0, 25, 50, 75, 100]:
        x = left + pct / 100 * (right - left)
        draw.line((x, top - 45 * scale, x, bottom + 45 * scale), fill="#D1D5DB", width=1 * scale)
        pct_text = f"{pct}%"
        pct_box = draw.textbbox((0, 0), pct_text, font=small_font)
        draw.text((x - (pct_box[2] - pct_box[0]) / 2, bottom + 58 * scale), pct_text, fill="#6B7280", font=small_font)

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(OUTPUT, quality=95)


if __name__ == "__main__":
    main()
