from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from build_career_curve_clusters import GRID, kmeans


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "5_discussion"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
MEMBERSHIP = (
    ROOT
    / "results"
    / "analysis"
    / "4.3_career_curve_clusters"
    / "final_40_or_8_rule"
    / "player_career_trajectory_classification_raw_elo.csv"
)
RELATIVE_MEMBERSHIP = (
    ROOT
    / "results"
    / "analysis"
    / "4.3_career_curve_clusters"
    / "final_40_or_8_rule"
    / "player_career_trajectory_classification.csv"
)
PEAK_VARIANTS = ROOT / "results" / "analysis" / "4.1_peak_elo" / "peak_elo_top10_by_playoff_k.csv"
PLAYERS = ROOT / "processed_data" / "players.csv"
SMOOTHING_GAMES = 21


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> list[float]:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoffs = np.asarray(quantiles) * cumulative[-1]
    return [float(values[min(np.searchsorted(cumulative, cutoff), len(values) - 1)]) for cutoff in cutoffs]


def period_label(season: int) -> str:
    if season < 1980:
        return "1976-79"
    if season < 1990:
        return "1980s"
    if season < 2000:
        return "1990s"
    if season < 2010:
        return "2000s"
    if season < 2020:
        return "2010s"
    return "2020s"


def standardized_curve(group: pd.DataFrame) -> np.ndarray:
    group = group.sort_values("game_date").reset_index(drop=True)
    smoothed = group["rating_after"].rolling(
        SMOOTHING_GAMES, center=True, min_periods=1
    ).mean().to_numpy()
    progress = np.linspace(0.0, 1.0, len(group))
    sampled = np.interp(GRID, progress, smoothed)
    return (sampled - sampled.mean()) / sampled.std()


def build_early_stability_chart(diagnostics: pd.DataFrame) -> None:
    data = diagnostics.copy()
    data["season_ending"] = data["season"] + 1
    post = data[data["season"] >= 1983]
    stable_center = float(post["minutes_wmean"].median())

    scale = 2
    width, height = 1800 * scale, 950 * scale
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

    left, top, right, bottom = [v * scale for v in (120, 220, 1690, 790)]
    draw.text((left, 42 * scale), "Most of the initial Elo-scale adjustment ended by 1983-84", fill="#111827", font=title_font)
    draw.text(
        (left, 101 * scale),
        "Season-end Elo average, weighted by minutes played",
        fill="#4B5563",
        font=subtitle_font,
    )

    x_min = int(data["season_ending"].min())
    x_max = int(data["season_ending"].max())
    y_min, y_max = 1400.0, 1680.0

    def xy(year: float, value: float) -> tuple[int, int]:
        x = left + (year - x_min) / (x_max - x_min) * (right - left)
        y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        return int(x), int(y)

    early_end = 1983
    early_x, _ = xy(early_end, y_min)
    draw.rectangle((left, top, early_x, bottom), fill=(194, 65, 12, 22))

    for value in range(1400, 1681, 50):
        _, y = xy(x_min, value)
        draw.line((left, y, right, y), fill="#D1D5DB", width=1 * scale)
        label = str(value)
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((left - (box[2] - box[0]) - 13 * scale, y - 9 * scale), label, fill="#6B7280", font=small_font)

    for year in [1977, 1980, 1990, 2000, 2010, 2020, 2026]:
        x, _ = xy(year, y_min)
        draw.line((x, top, x, bottom), fill="#E5E7EB", width=1 * scale)
        label = str(year)
        box = draw.textbbox((0, 0), label, font=body_font)
        draw.text((x - (box[2] - box[0]) / 2, bottom + 18 * scale), label, fill="#4B5563", font=body_font)

    center_y = xy(x_min, stable_center)[1]
    draw.line((early_x, center_y, right, center_y), fill=(107, 114, 128, 150), width=2 * scale)
    draw.text((right - 205 * scale, center_y - 30 * scale), f"Post-1983 median: {stable_center:.0f}", fill="#6B7280", font=small_font)

    points = [xy(float(row.season_ending), float(row.minutes_wmean)) for row in data.itertuples()]
    draw.line(points, fill="#2563EB", width=7 * scale, joint="curve")
    for x, y in points:
        radius = 3 * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#2563EB")

    marker = data[data["season"] == 1983].iloc[0]
    marker_x, marker_y = xy(float(marker["season_ending"]), float(marker["minutes_wmean"]))
    draw.line((marker_x, top, marker_x, bottom), fill=(55, 65, 81, 130), width=2 * scale)
    radius = 7 * scale
    draw.ellipse((marker_x - radius, marker_y - radius, marker_x + radius, marker_y + radius), fill="#2563EB")
    draw.text((marker_x + 14 * scale, marker_y - 38 * scale), "1983-84 season", fill="#374151", font=bold_font)
    draw.text((left + 22 * scale, top + 22 * scale), "Initial scale adjustment", fill="#9A3412", font=bold_font)

    draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2 * scale)
    x_label = "Season ending year"
    box = draw.textbbox((0, 0), x_label, font=bold_font)
    draw.text(((left + right - (box[2] - box[0])) / 2, bottom + 68 * scale), x_label, fill="#374151", font=bold_font)

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(OUT / "early_elo_scale_stability.png", quality=95)


def build_era_shape_chart(periods: pd.DataFrame) -> None:
    periods = periods.copy()
    periods["lower_gap"] = periods["weighted_median"] - periods["weighted_p10"]
    periods["upper_gap"] = periods["weighted_p90"] - periods["weighted_median"]

    scale = 2
    width, height = 1800 * scale, 980 * scale
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

    left, top, right, bottom = [v * scale for v in (185, 225, 1690, 800)]
    draw.text((120 * scale, 42 * scale), "The modern Elo distribution has a longer upper tail", fill="#111827", font=title_font)
    draw.text(
        (120 * scale, 101 * scale),
        "Distance from the minutes-weighted median to the 10th and 90th percentiles",
        fill="#4B5563",
        font=subtitle_font,
    )

    x_min, x_max = -250.0, 350.0

    def x_coord(value: float) -> int:
        return int(left + (value - x_min) / (x_max - x_min) * (right - left))

    zero_x = x_coord(0.0)
    for value in range(-200, 351, 50):
        x = x_coord(float(value))
        draw.line((x, top, x, bottom), fill="#E5E7EB", width=1 * scale)
        label = f"{value:+d}" if value != 0 else "Median"
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((x - (box[2] - box[0]) / 2, bottom + 19 * scale), label, fill="#6B7280", font=small_font)
    draw.line((zero_x, top, zero_x, bottom), fill="#6B7280", width=2 * scale)

    row_height = (bottom - top) / len(periods)
    for index, row in enumerate(periods.itertuples()):
        y = int(top + (index + 0.5) * row_height)
        draw.line((left, y + row_height / 2, right, y + row_height / 2), fill="#E5E7EB", width=1 * scale)
        low_x = x_coord(-float(row.lower_gap))
        high_x = x_coord(float(row.upper_gap))
        draw.line((low_x, y, zero_x, y), fill="#9DB9D5", width=12 * scale)
        draw.line((zero_x, y, high_x, y), fill="#2E8B57", width=12 * scale)
        radius = 7 * scale
        draw.ellipse((low_x - radius, y - radius, low_x + radius, y + radius), fill="#6B95BD")
        draw.ellipse((high_x - radius, y - radius, high_x + radius, y + radius), fill="#2E8B57")
        label = str(row.period)
        box = draw.textbbox((0, 0), label, font=body_font)
        draw.text((left - (box[2] - box[0]) - 22 * scale, y - 11 * scale), label, fill="#374151", font=body_font)
        draw.text((low_x - 62 * scale, y - 31 * scale), f"−{row.lower_gap:.0f}", fill="#4B6F93", font=small_font)
        draw.text((high_x + 10 * scale, y - 31 * scale), f"+{row.upper_gap:.0f}", fill="#1F7A4A", font=small_font)

    draw.text((left, 165 * scale), "10th percentile", fill="#4B6F93", font=bold_font)
    draw.text((right - 120 * scale, 165 * scale), "90th percentile", fill="#1F7A4A", font=bold_font)
    axis_label = "Elo points relative to the era median"
    box = draw.textbbox((0, 0), axis_label, font=bold_font)
    draw.text(((left + right - (box[2] - box[0])) / 2, bottom + 66 * scale), axis_label, fill="#374151", font=bold_font)
    draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2 * scale)

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(OUT / "era_elo_distribution_asymmetry.png", quality=95)


def parse_peak_cell(value: str) -> tuple[str, float]:
    match = re.match(r"^(.*) \(([0-9.]+)\)$", value)
    if not match:
        raise ValueError(value)
    return match.group(1), float(match.group(2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diagnostics = pd.read_csv(DIAGNOSTICS)
    diagnostics.assign(season_ending=diagnostics["season"] + 1).to_csv(
        OUT / "league_elo_scale_by_season.csv", index=False
    )
    build_early_stability_chart(diagnostics)

    membership = pd.read_csv(MEMBERSHIP)
    eligible_ids = set(membership["player_id"])
    updates = pd.read_csv(
        UPDATES,
        usecols=[
            "player_id", "game_date", "season", "track", "rating_after",
            "z_game_score", "game_score", "minutes_share",
        ],
        parse_dates=["game_date"],
    )
    eligible_updates = updates[updates["player_id"].isin(eligible_ids)].copy()

    ids, curves = [], []
    for player_id, group in eligible_updates.groupby("player_id", sort=True):
        ids.append(int(player_id))
        curves.append(standardized_curve(group))
    x = np.vstack(curves)
    labels, centers, _ = kmeans(x, 3, seed=42, n_init=80)
    order = np.argsort([GRID[int(np.argmax(center))] for center in centers])
    centers = centers[order]
    remap = {int(old): int(new) + 1 for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in labels])

    fitted = pd.DataFrame({"player_id": ids, "recreated_cluster": labels})
    check = membership[["player_id", "full_name", "raw_elo_cluster"]].merge(fitted, on="player_id")
    recreation_agreement = float((check["raw_elo_cluster"] == check["recreated_cluster"]).mean())

    names = pd.read_csv(PLAYERS, usecols=["personId", "firstName", "lastName"])
    names["full_name"] = names["firstName"].fillna("") + " " + names["lastName"].fillna("")
    name_to_id = names.set_index("full_name")["personId"].to_dict()
    sensitivity_rows = []
    for full_name in ["Magic Johnson", "Larry Bird"]:
        player_id = int(name_to_id[full_name])
        group = eligible_updates[eligible_updates["player_id"] == player_id].copy()
        first_season = int(group["season"].min())
        variants = [("Full career", first_season)]
        variants.extend((f"Drop first {n} season{'s' if n > 1 else ''}", first_season + n) for n in range(1, 5))
        variants.append(("Begin with 1983-84", 1983))
        for variant, cutoff in variants:
            subset = group[group["season"] >= cutoff]
            curve = standardized_curve(subset)
            distances = np.sqrt(np.square(centers - curve).sum(axis=1))
            sensitivity_rows.append(
                {
                    "player": full_name,
                    "variant": variant,
                    "first_season_used": cutoff,
                    "games_used": len(subset),
                    "assigned_cluster": int(np.argmin(distances) + 1),
                    "distance_early": float(distances[0]),
                    "distance_mid": float(distances[1]),
                    "distance_late": float(distances[2]),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    relative = pd.read_csv(RELATIVE_MEMBERSHIP, usecols=["full_name", "cluster"])
    relative = relative[relative["full_name"].isin(["Magic Johnson", "Larry Bird"])].rename(
        columns={"cluster": "season_relative_cluster"}
    )
    sensitivity = sensitivity.merge(relative, left_on="player", right_on="full_name", how="left").drop(columns="full_name")
    sensitivity.to_csv(OUT / "magic_bird_early_period_sensitivity.csv", index=False)

    regular = updates[updates["track"] == "regular"].copy()
    regular["period"] = regular["season"].map(period_label)
    era_rows = []
    for period, group in regular.groupby("period", sort=False):
        values = group["z_game_score"].to_numpy(float)
        weights = group["minutes_share"].to_numpy(float)
        mean = float(np.mean(values))
        sd = float(np.std(values))
        skew = float(np.mean(((values - mean) / sd) ** 3))
        p01, p10, p50, p90, p99 = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99])
        wp10, wp50, wp90, wp99 = weighted_quantile(values, weights, [0.10, 0.50, 0.90, 0.99])
        era_rows.append(
            {
                "period": period,
                "player_games": len(group),
                "mean_z": mean,
                "sd_z": sd,
                "skew_z": skew,
                "p01_z": p01,
                "p10_z": p10,
                "median_z": p50,
                "p90_z": p90,
                "p99_z": p99,
                "minutes_weighted_p10_z": wp10,
                "minutes_weighted_median_z": wp50,
                "minutes_weighted_p90_z": wp90,
                "minutes_weighted_p99_z": wp99,
            }
        )
    era_perf = pd.DataFrame(era_rows)
    era_perf.to_csv(OUT / "game_score_z_distribution_by_era.csv", index=False)

    peak_idx = updates.groupby("player_id")["rating_after"].idxmax()
    peaks = updates.loc[peak_idx, ["player_id", "game_date", "season", "rating_after"]].copy()
    peaks = peaks.merge(names[["personId", "full_name"]], left_on="player_id", right_on="personId", how="left")
    peaks["peak_period"] = peaks["season"].map(period_label)
    peaks = peaks.sort_values("rating_after", ascending=False).reset_index(drop=True)
    peaks["all_time_rank"] = np.arange(1, len(peaks) + 1)
    peaks.head(100).to_csv(OUT / "top100_peak_elo_with_peak_era.csv", index=False)
    count_rows = []
    for top_n in [10, 20, 50, 100]:
        counts = peaks.head(top_n)["peak_period"].value_counts()
        for period in ["1976-79", "1980s", "1990s", "2000s", "2010s", "2020s"]:
            count_rows.append({"top_n": top_n, "period": period, "players": int(counts.get(period, 0))})
    pd.DataFrame(count_rows).to_csv(OUT / "peak_elo_era_counts.csv", index=False)

    variants = pd.read_csv(PEAK_VARIANTS)
    lebron_gaps = []
    for column in ["k=0", "k=0.5", "k=1", "k=1.5", "k=2"]:
        first_name, first_value = parse_peak_cell(variants.loc[0, column])
        second_name, second_value = parse_peak_cell(variants.loc[1, column])
        lebron_gaps.append(
            {
                "playoff_k": column,
                "leader": first_name,
                "leader_peak_elo": first_value,
                "second": second_name,
                "second_peak_elo": second_value,
                "gap": first_value - second_value,
            }
        )
    pd.DataFrame(lebron_gaps).to_csv(OUT / "lebron_peak_gap_by_playoff_weight.csv", index=False)

    season_end = pd.read_csv(
        ROOT / "results" / "analysis" / "4.3_era_elo_distribution" / "period_distribution_canonical_pk1.csv"
    )
    season_end.to_csv(OUT / "season_end_elo_distribution_by_era.csv", index=False)
    build_era_shape_chart(season_end)

    early = diagnostics[diagnostics["season"].between(1976, 1983)]
    later = diagnostics[diagnostics["season"] >= 1983]
    summary = [
        "Discussion checks",
        "",
        f"Career-cluster recreation agreement: {recreation_agreement:.1%}",
        f"Minutes-weighted league mean: {early.iloc[0]['minutes_wmean']:.1f} in 1976-77; "
        f"{early.iloc[-1]['minutes_wmean']:.1f} in 1983-84; post-1983 median {later['minutes_wmean'].median():.1f}.",
        "",
        "Magic/Bird sensitivity:",
        sensitivity.to_string(index=False),
        "",
        "Game Score z-score distribution by era:",
        era_perf.round(3).to_string(index=False),
        "",
        "LeBron peak gap by playoff weight:",
        pd.DataFrame(lebron_gaps).round(2).to_string(index=False),
        "",
        "Peak-era counts:",
        pd.DataFrame(count_rows).pivot(index="period", columns="top_n", values="players").to_string(),
    ]
    (OUT / "discussion_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
