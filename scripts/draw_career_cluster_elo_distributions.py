from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "analysis" / "4.3_career_curve_clusters" / "final_40_or_8_rule"
SOURCE = BASE / "cluster_characteristics_player_level_raw_elo.csv"
OUTPUT = BASE / "career_elo_distributions_by_trajectory.png"

CLUSTERS = [1, 2, 3]
NAMES = ["Early peak", "Mid-career peak", "Late peak"]
COLORS = ["#C2410C", "#2563EB", "#2E8B57"]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def main() -> None:
    data = pd.read_csv(SOURCE)
    data["cluster"] = data["raw_elo_cluster"]
    scale = 2
    width, height = 1800 * scale, 900 * scale
    image = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(image, "RGBA")
    font_dir = Path("C:/Windows/Fonts")

    def font(name: str, size: int):
        return ImageFont.truetype(str(font_dir / name), size * scale)

    title_font = font("arialbd.ttf", 42)
    subtitle_font = font("arial.ttf", 19)
    panel_font = font("arialbd.ttf", 23)
    body_font = font("arial.ttf", 18)
    small_font = font("arial.ttf", 15)
    bold_font = font("arialbd.ttf", 16)

    draw.text((105 * scale, 42 * scale), "Elo level by career-trajectory group", fill="#111827", font=title_font)
    draw.text(
        (105 * scale, 100 * scale),
        "Boxes show the middle 50%; whiskers show the 10th–90th percentiles; diamonds mark group means",
        fill="#4B5563",
        font=subtitle_font,
    )

    panels = [
        ((105, 220, 835, 720), "career_mean_elo", "Career-average Elo", 1250, 2100, 100),
        ((965, 220, 1695, 720), "raw_peak_elo", "Career-high Elo", 1400, 2350, 100),
    ]

    for bounds, column, panel_title, y_min, y_max, tick_step in panels:
        left, top, right, bottom = [v * scale for v in bounds]
        draw.text((left, 163 * scale), panel_title, fill="#111827", font=panel_font)

        def y_pos(value: float) -> int:
            return int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))

        for tick in range(y_min, y_max + 1, tick_step):
            y = y_pos(tick)
            draw.line((left, y, right, y), fill="#D1D5DB", width=1 * scale)
            label = f"{tick:,}"
            label_box = draw.textbbox((0, 0), label, font=small_font)
            draw.text((left - (label_box[2] - label_box[0]) - 12 * scale, y - 9 * scale), label, fill="#6B7280", font=small_font)

        x_positions = np.linspace(left + 125 * scale, right - 125 * scale, 3)
        for cluster, name, color, x in zip(CLUSTERS, NAMES, COLORS, x_positions):
            values = data.loc[data["cluster"] == cluster, column].dropna().to_numpy(float)
            p10, p25, median, p75, p90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
            mean = float(values.mean())
            rgb = hex_to_rgb(color)
            box_half = 55 * scale
            cap_half = 25 * scale

            draw.line((x, y_pos(p10), x, y_pos(p90)), fill="#6B7280", width=3 * scale)
            draw.line((x - cap_half, y_pos(p10), x + cap_half, y_pos(p10)), fill="#6B7280", width=3 * scale)
            draw.line((x - cap_half, y_pos(p90), x + cap_half, y_pos(p90)), fill="#6B7280", width=3 * scale)
            draw.rectangle(
                (x - box_half, y_pos(p75), x + box_half, y_pos(p25)),
                fill=(*rgb, 55),
                outline=(*rgb, 255),
                width=4 * scale,
            )
            draw.line((x - box_half, y_pos(median), x + box_half, y_pos(median)), fill="#111827", width=4 * scale)
            diamond = 9 * scale
            my = y_pos(mean)
            draw.polygon([(x, my - diamond), (x + diamond, my), (x, my + diamond), (x - diamond, my)], fill=(*rgb, 255))

            value_label = f"Median {median:,.0f}"
            value_box = draw.textbbox((0, 0), value_label, font=bold_font)
            draw.text((x - (value_box[2] - value_box[0]) / 2, y_pos(p90) - 31 * scale), value_label, fill="#374151", font=bold_font)
            name_box = draw.textbbox((0, 0), name, font=body_font)
            draw.text((x - (name_box[2] - name_box[0]) / 2, bottom + 18 * scale), name, fill="#374151", font=body_font)

        draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2 * scale)

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(OUTPUT, quality=95)


if __name__ == "__main__":
    main()
