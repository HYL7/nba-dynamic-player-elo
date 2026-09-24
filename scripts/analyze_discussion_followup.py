from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from build_career_curve_clusters import GRID, kmeans


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "5_discussion"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
PLAYERS = ROOT / "processed_data" / "players.csv"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
ELIGIBLE = (
    ROOT
    / "results"
    / "analysis"
    / "4.3_career_curve_clusters"
    / "final_40_or_8_rule"
    / "player_career_trajectory_classification_raw_elo.csv"
)

SMOOTHING_GAMES = 21
NAMES = [
    "LeBron James",
    "Michael Jordan",
    "Shaquille O'Neal",
    "Kareem Abdul-Jabbar",
    "Magic Johnson",
    "Larry Bird",
    "Tim Duncan",
    "Kevin Durant",
    "Stephen Curry",
    "Nikola Jokic",
]


def sampled_standardized_curve(group: pd.DataFrame) -> np.ndarray:
    group = group.sort_values("game_date").reset_index(drop=True)
    smooth = group["rating_after"].rolling(
        SMOOTHING_GAMES, center=True, min_periods=1
    ).mean().to_numpy()
    progress = np.linspace(0.0, 1.0, len(group))
    sampled = np.interp(GRID, progress, smooth)
    return (sampled - sampled.mean()) / sampled.std()


def recreate_raw_centers(updates: pd.DataFrame, eligible_ids: set[int]):
    ids, curves = [], []
    for player_id, group in updates[updates["player_id"].isin(eligible_ids)].groupby(
        "player_id", sort=True
    ):
        ids.append(int(player_id))
        curves.append(sampled_standardized_curve(group))
    matrix = np.vstack(curves)
    labels, centers, _ = kmeans(matrix, 3, seed=42, n_init=80)
    order = np.argsort([GRID[int(np.argmax(center))] for center in centers])
    centers = centers[order]
    remap = {int(old): int(new) + 1 for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in labels])
    return ids, matrix, labels, centers


def draw_magic_bird_chart(updates, players, centers):
    name_map = players.set_index("full_name")["player_id"].to_dict()
    colors = {"Larry Bird": "#C2410C", "Magic Johnson": "#2563EB"}

    scale = 2
    width, height = 1800 * scale, 1180 * scale
    image = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(image, "RGBA")
    font_dir = Path("C:/Windows/Fonts")

    def font(name, size):
        return ImageFont.truetype(str(font_dir / name), size * scale)

    title_font = font("arialbd.ttf", 38)
    panel_font = font("arialbd.ttf", 21)
    body_font = font("arial.ttf", 16)
    small_font = font("arial.ttf", 14)

    draw.text((100 * scale, 35 * scale), "Bird and Magic: actual Elo paths and career-shape classification",
              fill="#111827", font=title_font)
    draw.text((100 * scale, 88 * scale),
              "Top: Elo by actual age. Bottom: the standardized, games-played curve used by the clustering.",
              fill="#4B5563", font=body_font)

    panels = {
        (0, 0): (105, 180, 850, 585),
        (0, 1): (990, 180, 1735, 585),
        (1, 0): (105, 700, 850, 1105),
        (1, 1): (990, 700, 1735, 1105),
    }

    def line(draw_obj, points, fill, width_px=3):
        draw_obj.line([(int(x), int(y)) for x, y in points], fill=fill,
                      width=width_px * scale, joint="curve")

    summary_rows = []
    for col, name in enumerate(["Larry Bird", "Magic Johnson"]):
        group = updates[updates["player_id"] == name_map[name]].sort_values("game_date").copy()
        group["elo_21g"] = group["rating_after"].rolling(
            SMOOTHING_GAMES, center=True, min_periods=1
        ).mean()
        group["career_progress"] = np.linspace(0, 100, len(group))
        peak = group.loc[group["elo_21g"].idxmax()]
        season = group.groupby("season", as_index=False).agg(
            age=("age", "mean"), elo=("rating_after", "mean"), games=("rating_after", "size")
        )

        left, top, right, bottom = [v * scale for v in panels[(0, col)]]
        x_min, x_max = float(group["age"].min()), float(group["age"].max())
        y_min = np.floor((group["elo_21g"].min() - 40) / 100) * 100
        y_max = np.ceil((group["elo_21g"].max() + 40) / 100) * 100

        def xy_age(x, y):
            return (
                left + (x - x_min) / (x_max - x_min) * (right - left),
                bottom - (y - y_min) / (y_max - y_min) * (bottom - top),
            )

        for yv in np.arange(y_min, y_max + 1, 100):
            _, yy = xy_age(x_min, yv)
            draw.line((left, yy, right, yy), fill="#E5E7EB", width=1 * scale)
            draw.text((left - 48 * scale, yy - 8 * scale), f"{int(yv)}", fill="#6B7280", font=small_font)
        for xv in np.arange(np.ceil(x_min), np.floor(x_max) + 1, 2):
            xx, _ = xy_age(xv, y_min)
            draw.text((xx - 10 * scale, bottom + 10 * scale), f"{int(xv)}", fill="#6B7280", font=small_font)
        gap_breaks = np.where(group["game_date"].diff().dt.days.fillna(0).to_numpy() > 250)[0]
        segment_starts = np.r_[0, gap_breaks]
        segment_ends = np.r_[gap_breaks, len(group)]
        for start, end in zip(segment_starts, segment_ends):
            line(draw, [xy_age(x, y) for x, y in zip(
                group["age"].iloc[start:end], group["elo_21g"].iloc[start:end]
            )], colors[name], 3)
        for row in season.itertuples():
            xx, yy = xy_age(row.age, row.elo)
            r = 4 * scale
            draw.ellipse((xx-r, yy-r, xx+r, yy+r), fill=colors[name])
        px, py = xy_age(float(peak["age"]), float(peak["elo_21g"]))
        r = 6 * scale
        draw.ellipse((px-r, py-r, px+r, py+r), fill="#111827")
        draw.text((px + 9 * scale, py - 32 * scale),
                  f"Peak: age {peak['age']:.1f}, {peak['elo_21g']:.0f}", fill="#111827", font=small_font)
        draw.text((left, top - 40 * scale), f"{name}: Elo by actual age", fill="#111827", font=panel_font)
        draw.text(((left + right)//2 - 22*scale, bottom + 42*scale), "Age", fill="#374151", font=body_font)
        draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2 * scale)

        curve = sampled_standardized_curve(group)
        distances = np.sqrt(np.square(centers - curve).sum(axis=1))
        left, top, right, bottom = [v * scale for v in panels[(1, col)]]
        all_values = np.concatenate([curve, centers[1], centers[2]])
        y_min = np.floor((all_values.min() - 0.1) * 2) / 2
        y_max = np.ceil((all_values.max() + 0.1) * 2) / 2

        def xy_curve(x, y):
            return (
                left + x / 100 * (right - left),
                bottom - (y - y_min) / (y_max - y_min) * (bottom - top),
            )

        for yv in np.arange(np.ceil(y_min), np.floor(y_max) + 0.1, 1):
            _, yy = xy_curve(0, yv)
            draw.line((left, yy, right, yy), fill="#E5E7EB", width=1 * scale)
            draw.text((left - 28 * scale, yy - 8 * scale), str(yv), fill="#6B7280", font=small_font)
        for xv in [0, 25, 50, 75, 100]:
            xx, _ = xy_curve(xv, y_min)
            draw.text((xx - 12 * scale, bottom + 10 * scale), f"{xv}%", fill="#6B7280", font=small_font)
        line(draw, [xy_curve(x*100, y) for x, y in zip(GRID, centers[1])], "#6B7280", 2)
        line(draw, [xy_curve(x*100, y) for x, y in zip(GRID, centers[2])], "#2E8B57", 2)
        line(draw, [xy_curve(x*100, y) for x, y in zip(GRID, curve)], colors[name], 3)
        draw.text((left, top - 40 * scale), f"{name}: curve used by clustering", fill="#111827", font=panel_font)
        draw.text(((left + right)//2 - 95*scale, bottom + 42*scale), "Career progress (games played)",
                  fill="#374151", font=body_font)
        draw.text((left + 12*scale, top + 10*scale), "Player", fill=colors[name], font=small_font)
        draw.text((left + 92*scale, top + 10*scale), "Mid center", fill="#6B7280", font=small_font)
        draw.text((left + 190*scale, top + 10*scale), "Late center", fill="#2E8B57", font=small_font)
        draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2 * scale)

        first = group.iloc[0]
        last = group.iloc[-1]
        summary_rows.append({
            "player": name,
            "games": len(group),
            "first_date": first["game_date"],
            "last_date": last["game_date"],
            "first_age": first["age"],
            "last_age": last["age"],
            "peak_date_21g": peak["game_date"],
            "peak_age_21g": peak["age"],
            "peak_progress_21g": peak["career_progress"],
            "peak_elo_21g": peak["elo_21g"],
            "distance_to_mid_center": distances[1],
            "distance_to_late_center": distances[2],
            "seasons_with_under_20_games": ", ".join(
                str(int(row.season + 1)) for row in season.itertuples() if row.games < 20
            ),
        })

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    image.save(OUT / "magic_bird_career_elo_paths.png", quality=95)
    return pd.DataFrame(summary_rows)


def build_superstar_comparison(updates, players, diagnostics):
    season_mean = diagnostics.set_index("season")["minutes_wmean"]
    data = updates.copy()
    data["elo_relative_to_season_mean"] = data["rating_after"] - data["season"].map(season_mean)
    target = players[players["full_name"].isin(NAMES)][["player_id", "full_name"]]
    data = data.merge(target, on="player_id", how="inner")
    rows = []
    for name, group in data.groupby("full_name"):
        weights = group["minutes_share"].clip(lower=0)
        rows.append({
            "player": name,
            "games": len(group),
            "career_mean_elo_per_appearance": group["rating_after"].mean(),
            "minutes_weighted_mean_elo": np.average(group["rating_after"], weights=weights),
            "career_mean_elo_above_season_mean": group["elo_relative_to_season_mean"].mean(),
            "minutes_weighted_elo_above_season_mean": np.average(
                group["elo_relative_to_season_mean"], weights=weights
            ),
            "peak_elo": group["rating_after"].max(),
            "mean_game_score_z": group["z_game_score"].mean(),
            "minutes_weighted_game_score_z": np.average(group["z_game_score"], weights=weights),
            "first_season": int(group["season"].min()),
            "last_season": int(group["season"].max()),
        })
    result = pd.DataFrame(rows).sort_values("career_mean_elo_per_appearance", ascending=False)
    result.to_csv(OUT / "superstar_career_elo_comparison.csv", index=False)
    return result


def build_eligible_career_mean_ranking(updates, players, diagnostics, eligible_ids):
    season_mean = diagnostics.set_index("season")["minutes_wmean"]
    data = updates[updates["player_id"].isin(eligible_ids)].copy()
    data["elo_relative_to_season_mean"] = data["rating_after"] - data["season"].map(season_mean)
    result = data.groupby("player_id").agg(
        games=("rating_after", "size"),
        career_mean_elo=("rating_after", "mean"),
        career_mean_relative_elo=("elo_relative_to_season_mean", "mean"),
        peak_elo=("rating_after", "max"),
        mean_game_score_z=("z_game_score", "mean"),
        first_season=("season", "min"),
        last_season=("season", "max"),
    ).reset_index().merge(players[["player_id", "full_name"]], on="player_id", how="left")
    result = result.sort_values("career_mean_elo", ascending=False).reset_index(drop=True)
    result["career_mean_rank"] = np.arange(1, len(result) + 1)
    result["elo_above_season_mean_in_expected_perf_units"] = result["career_mean_relative_elo"] / 285.0
    result.to_csv(OUT / "eligible_player_career_mean_elo_ranking.csv", index=False)
    return result


def build_season_tail_detail(updates):
    regular = updates[updates["track"] == "regular"].copy()
    rows = []
    for season, group in regular.groupby("season"):
        last = group.sort_values("game_date").groupby("player_id", as_index=False).tail(1)
        values = last["rating_after"].to_numpy()
        weights = group.groupby("player_id")["minutes_share"].sum().reindex(last["player_id"]).to_numpy()
        order = np.argsort(values)
        v, w = values[order], weights[order]
        cum = np.cumsum(w) / np.sum(w)
        q = {p: float(v[min(np.searchsorted(cum, p), len(v) - 1)]) for p in [0.1, 0.5, 0.9, 0.95, 0.99]}
        rows.append({
            "season": season,
            "players": len(last),
            "p10": q[0.1], "median": q[0.5], "p90": q[0.9], "p95": q[0.95], "p99": q[0.99],
            "max": float(values.max()),
            "lower_half_width": q[0.5] - q[0.1],
            "upper_tail_p90": q[0.9] - q[0.5],
            "upper_tail_p95": q[0.95] - q[0.5],
            "upper_tail_p99": q[0.99] - q[0.5],
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "season_end_elo_tail_detail.csv", index=False)
    return result


def draw_tail_over_time(tails):
    data = tails.copy().sort_values("season")
    for col in ["lower_half_width", "upper_tail_p90", "upper_tail_p99"]:
        data[col + "_smooth"] = data[col].rolling(5, center=True, min_periods=3).mean()

    scale = 2
    width, height = 1700 * scale, 860 * scale
    image = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(image, "RGBA")
    font_dir = Path("C:/Windows/Fonts")

    def font(name, size):
        return ImageFont.truetype(str(font_dir / name), size * scale)

    title_font = font("arialbd.ttf", 37)
    panel_font = font("arialbd.ttf", 19)
    body_font = font("arial.ttf", 15)
    small_font = font("arial.ttf", 13)
    draw.text((95*scale, 35*scale), "The lower half compressed before the top tail clearly expanded",
              fill="#111827", font=title_font)
    draw.text((95*scale, 86*scale), "Five-season centered averages of season-end Elo distances from the median",
              fill="#4B5563", font=body_font)

    panels = [(105, 180, 790, 720), (970, 180, 1655, 720)]
    configs = [
        (["lower_half_width_smooth"], ["Median to 10th percentile"], ["#2563EB"],
         "Lower half of the distribution", 140, 240),
        (["upper_tail_p90_smooth", "upper_tail_p99_smooth"],
         ["Median to 90th percentile", "Median to 99th percentile"],
         ["#2E8B57", "#C2410C"], "Upper tail of the distribution", 180, 600),
    ]
    x_min, x_max = int(data["season"].min()), int(data["season"].max())
    for panel, config in zip(panels, configs):
        cols, labels, colors, title, y_min, y_max = config
        left, top, right, bottom = [v*scale for v in panel]

        def xy(x, y):
            return (left + (x-x_min)/(x_max-x_min)*(right-left),
                    bottom - (y-y_min)/(y_max-y_min)*(bottom-top))

        for yv in np.linspace(y_min, y_max, 5):
            _, yy = xy(x_min, yv)
            draw.line((left, yy, right, yy), fill="#E5E7EB", width=scale)
            draw.text((left-42*scale, yy-7*scale), f"{yv:.0f}", fill="#6B7280", font=small_font)
        for year in [1980, 1990, 2000, 2010, 2020]:
            xx, _ = xy(year, y_min)
            draw.text((xx-16*scale, bottom+11*scale), str(year), fill="#6B7280", font=small_font)
        for col, label, color in zip(cols, labels, colors):
            valid = data.dropna(subset=[col])
            draw.line([tuple(map(int, xy(r.season, getattr(r, col)))) for r in valid.itertuples()],
                      fill=color, width=4*scale, joint="curve")
        draw.text((left, top-39*scale), title, fill="#111827", font=panel_font)
        for index, (label, color) in enumerate(zip(labels, colors)):
            draw.text((left + index*225*scale, top+10*scale), label, fill=color, font=small_font)
        draw.text(((left+right)//2-50*scale, bottom+46*scale), "Season", fill="#374151", font=body_font)
        draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2*scale)
    image = image.resize((width//scale, height//scale), Image.Resampling.LANCZOS)
    image.save(OUT / "elo_distribution_tail_over_time.png", quality=95)


def draw_career_average_ranking(ranking):
    data = ranking.head(15).sort_values("career_mean_elo")
    scale = 2
    width, height = 1500*scale, 980*scale
    image = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(image, "RGBA")
    font_dir = Path("C:/Windows/Fonts")

    def font(name, size):
        return ImageFont.truetype(str(font_dir / name), size*scale)

    title_font = font("arialbd.ttf", 37)
    body_font = font("arial.ttf", 16)
    small_font = font("arial.ttf", 14)
    draw.text((85*scale, 35*scale), "Career-average Elo among the highest-rated long-career players",
              fill="#111827", font=title_font)
    draw.text((85*scale, 87*scale),
              "Raw Elo values; eligible completed or late-stage careers",
              fill="#4B5563", font=body_font)
    left, top, right, bottom = [v*scale for v in (295, 160, 1390, 885)]
    x_min, x_max = 1900, 2180

    def xcoord(value):
        return left + (value-x_min)/(x_max-x_min)*(right-left)

    row_h = (bottom-top)/len(data)
    for i, row in enumerate(data.itertuples()):
        y = top + (i+0.5)*row_h
        color = "#C2410C" if row.full_name == "LeBron James" else ("#2563EB" if row.full_name == "Michael Jordan" else "#9CA3AF")
        draw.rectangle((xcoord(x_min), y-8*scale, xcoord(row.career_mean_elo), y+8*scale), fill=color)
        name_box = draw.textbbox((0, 0), row.full_name, font=small_font)
        draw.text((left-name_box[2]+name_box[0]-15*scale, y-8*scale), row.full_name, fill="#374151", font=small_font)
        draw.text((xcoord(row.career_mean_elo)+8*scale, y-8*scale),
                  f"{row.career_mean_elo:.0f}", fill="#374151", font=small_font)
    for value in [1900, 1950, 2000, 2050, 2100, 2150]:
        x = xcoord(value)
        draw.line((x, top, x, bottom), fill="#E5E7EB", width=scale)
        draw.text((x-12*scale, bottom+12*scale), str(value), fill="#6B7280", font=small_font)
    draw.text(((left+right)//2-70*scale, bottom+50*scale), "Career-average Elo",
              fill="#374151", font=body_font)
    draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2*scale)
    image = image.resize((width//scale, height//scale), Image.Resampling.LANCZOS)
    image.save(OUT / "career_average_elo_top15.png", quality=95)


def draw_jordan_lebron_paths(updates, players):
    name_map = players.set_index("full_name")["player_id"].to_dict()
    colors = {"Michael Jordan": "#2563EB", "LeBron James": "#C2410C"}
    groups = {}
    rows = []
    for name in colors:
        group = updates[updates["player_id"] == name_map[name]].sort_values("game_date").copy()
        group["elo_21g"] = group["rating_after"].rolling(21, center=True, min_periods=1).mean()
        group["progress"] = np.linspace(0, 100, len(group))
        groups[name] = group
        peak = group.loc[group["rating_after"].idxmax()]
        smooth_peak = group.loc[group["elo_21g"].idxmax()]
        rows.append({
            "player": name,
            "games": len(group),
            "first_age": group.iloc[0]["age"],
            "last_age": group.iloc[-1]["age"],
            "career_mean_elo": group["rating_after"].mean(),
            "instant_peak_elo": peak["rating_after"],
            "instant_peak_date": peak["game_date"],
            "instant_peak_age": peak["age"],
            "smooth_peak_elo": smooth_peak["elo_21g"],
            "smooth_peak_date": smooth_peak["game_date"],
            "smooth_peak_age": smooth_peak["age"],
            "share_games_at_2100_plus": (group["rating_after"] >= 2100).mean(),
            "share_games_at_2200_plus": (group["rating_after"] >= 2200).mean(),
            "first_20pct_mean": group.iloc[:max(1, len(group)//5)]["rating_after"].mean(),
            "middle_60pct_mean": group.iloc[len(group)//5:4*len(group)//5]["rating_after"].mean(),
            "last_20pct_mean": group.iloc[4*len(group)//5:]["rating_after"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "jordan_lebron_career_elo_summary.csv", index=False)

    scale = 2
    width, height = 1750*scale, 960*scale
    image = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(image, "RGBA")
    font_dir = Path("C:/Windows/Fonts")

    def font(name, size):
        return ImageFont.truetype(str(font_dir / name), size*scale)

    title_font = font("arialbd.ttf", 37)
    panel_font = font("arialbd.ttf", 19)
    body_font = font("arial.ttf", 15)
    small_font = font("arial.ttf", 13)
    draw.text((90*scale, 34*scale), "Jordan sustained a higher average; LeBron reached the higher single peak",
              fill="#111827", font=title_font)
    draw.text((90*scale, 85*scale), "21-game rolling Elo; gaps longer than 250 days are not connected",
              fill="#4B5563", font=body_font)
    panels = [(110, 180, 810, 790), (965, 180, 1665, 790)]

    for panel_index, (left0, top0, right0, bottom0) in enumerate(panels):
        left, top, right, bottom = [v*scale for v in (left0, top0, right0, bottom0)]
        x_min, x_max = ((18, 41) if panel_index == 0 else (0, 100))
        y_min, y_max = 1400, 2350

        def xy(x, y):
            return (left+(x-x_min)/(x_max-x_min)*(right-left),
                    bottom-(y-y_min)/(y_max-y_min)*(bottom-top))

        for yv in range(1400, 2351, 100):
            _, yy = xy(x_min, yv)
            draw.line((left, yy, right, yy), fill="#E5E7EB", width=scale)
            draw.text((left-48*scale, yy-7*scale), str(yv), fill="#6B7280", font=small_font)
        xticks = ([20, 25, 30, 35, 40] if panel_index == 0 else [0, 25, 50, 75, 100])
        for xv in xticks:
            xx, _ = xy(xv, y_min)
            label = f"{xv}%" if panel_index else str(xv)
            draw.text((xx-13*scale, bottom+11*scale), label, fill="#6B7280", font=small_font)

        for name, group in groups.items():
            xcol = "age" if panel_index == 0 else "progress"
            if panel_index == 0:
                gaps = np.where(group["game_date"].diff().dt.days.fillna(0).to_numpy() > 250)[0]
                starts, ends = np.r_[0, gaps], np.r_[gaps, len(group)]
            else:
                starts, ends = [0], [len(group)]
            for start, end in zip(starts, ends):
                points = [xy(x, y) for x, y in zip(group[xcol].iloc[start:end], group["elo_21g"].iloc[start:end])]
                draw.line([tuple(map(int, p)) for p in points], fill=colors[name], width=4*scale, joint="curve")
            peak = group.loc[group["elo_21g"].idxmax()]
            px, py = xy(float(peak[xcol]), float(peak["elo_21g"]))
            r = 5*scale
            draw.ellipse((px-r, py-r, px+r, py+r), fill=colors[name])
            draw.text((px+7*scale, py-23*scale), f"{peak['elo_21g']:.0f}", fill=colors[name], font=small_font)

        title = "Elo by actual age" if panel_index == 0 else "Elo by career progress"
        xlabel = "Age" if panel_index == 0 else "Career progress based on games played"
        draw.text((left, top-39*scale), title, fill="#111827", font=panel_font)
        draw.text(((left+right)//2-80*scale, bottom+48*scale), xlabel, fill="#374151", font=body_font)
        draw.text((left+12*scale, top+10*scale), "Michael Jordan", fill=colors["Michael Jordan"], font=small_font)
        draw.text((left+135*scale, top+10*scale), "LeBron James", fill=colors["LeBron James"], font=small_font)
        draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2*scale)

    image = image.resize((width//scale, height//scale), Image.Resampling.LANCZOS)
    image.save(OUT / "jordan_lebron_career_elo_paths.png", quality=95)
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    players = pd.read_csv(
        PLAYERS, usecols=["personId", "firstName", "lastName", "birthDate"]
    ).rename(columns={"personId": "player_id"})
    players["full_name"] = players["firstName"].fillna("") + " " + players["lastName"].fillna("")
    players["birthDate"] = pd.to_datetime(players["birthDate"], errors="coerce")
    birth = players.set_index("player_id")["birthDate"]

    updates = pd.read_csv(
        UPDATES,
        usecols=["player_id", "game_date", "season", "track", "rating_after", "minutes_share", "z_game_score"],
        parse_dates=["game_date"],
    )
    updates["birthDate"] = updates["player_id"].map(birth)
    updates["age"] = (updates["game_date"] - updates["birthDate"]).dt.days / 365.2425

    eligible = pd.read_csv(ELIGIBLE)
    ids, matrix, labels, centers = recreate_raw_centers(updates, set(eligible["player_id"]))
    bird_magic = draw_magic_bird_chart(updates, players, centers)
    bird_magic.to_csv(OUT / "magic_bird_career_elo_summary.csv", index=False)

    diagnostics = pd.read_csv(DIAGNOSTICS)
    stars = build_superstar_comparison(updates, players, diagnostics)
    career_ranking = build_eligible_career_mean_ranking(
        updates, players, diagnostics, set(eligible["player_id"])
    )
    draw_career_average_ranking(career_ranking)
    jordan_lebron = draw_jordan_lebron_paths(updates, players)
    tails = build_season_tail_detail(updates)
    draw_tail_over_time(tails)

    era = tails.assign(era=pd.cut(
        tails["season"], bins=[1975, 1979, 1989, 1999, 2009, 2019, 2026],
        labels=["1976-79", "1980s", "1990s", "2000s", "2010s", "2020s"]
    )).groupby("era", observed=True).agg(
        seasons=("season", "size"), players=("players", "mean"),
        lower_half_width=("lower_half_width", "mean"),
        upper_tail_p90=("upper_tail_p90", "mean"),
        upper_tail_p95=("upper_tail_p95", "mean"),
        upper_tail_p99=("upper_tail_p99", "mean"),
        maximum=("max", "mean"),
    ).reset_index()
    era.to_csv(OUT / "season_end_elo_tail_by_era.csv", index=False)

    print("\nBird/Magic")
    print(bird_magic.round(2).to_string(index=False))
    print("\nSuperstar comparison")
    print(stars.round(2).to_string(index=False))
    print("\nTop eligible career-average Elo")
    print(career_ranking.head(15).round(2).to_string(index=False))
    print("\nJordan and LeBron paths")
    print(jordan_lebron.round(3).to_string(index=False))
    print("\nEra tail detail")
    print(era.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
