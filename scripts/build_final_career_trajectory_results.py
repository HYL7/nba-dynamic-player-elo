from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from build_career_curve_clusters import GRID, kmeans, silhouette


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "4.3_career_curve_clusters" / "final_40_or_8_rule"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
PLAYERS = ROOT / "processed_data" / "players.csv"

MIN_GAMES_PER_QUALIFYING_SEASON = 40
MIN_QUALIFYING_SEASONS = 5
ACTIVE_MIN_AGE = 35.0
SMOOTHING_GAMES = 21

CLUSTER_NAMES = {
    1: "Early peak / gradual decline",
    2: "Conventional mid-career peak",
    3: "Late peak / sustained rise",
}
COLORS = {
    1: "#C2410C",
    2: "#2563EB",
    3: "#2E8B57",
}


def prepare_curve(group: pd.DataFrame) -> dict | None:
    group = group.sort_values("game_date").reset_index(drop=True)
    relative = group["relative_elo"].rolling(
        SMOOTHING_GAMES, center=True, min_periods=1
    ).mean().to_numpy()
    rating = group["rating_after"].rolling(
        SMOOTHING_GAMES, center=True, min_periods=1
    ).mean().to_numpy()
    progress = np.linspace(0.0, 1.0, len(group))
    sampled = np.interp(GRID, progress, relative)
    sd = sampled.std()
    if sd < 1e-9:
        return None
    standardized = (sampled - sampled.mean()) / sd
    peak_idx = int(np.argmax(relative))
    age = group["age"].to_numpy()
    sampled_age = np.interp(GRID, progress, age)
    return {
        "standardized": standardized,
        "sampled_age": sampled_age,
        "peak_progress": float(progress[peak_idx]),
        "peak_date": group.loc[peak_idx, "game_date"],
        "peak_age": float(age[peak_idx]),
        "peak_relative_elo_21g": float(relative[peak_idx]),
        "peak_elo_21g": float(rating[peak_idx]),
    }


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def draw_chart(
    curves: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    age_grid: np.ndarray,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    scale = 2
    width, height = 2000 * scale, 1200 * scale
    img = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(img, "RGBA")
    font_dir = Path("C:/Windows/Fonts")

    def font(name: str, size: int):
        return ImageFont.truetype(str(font_dir / name), size * scale)

    title_font = font("arialbd.ttf", 45)
    subtitle_font = font("arial.ttf", 22)
    body_font = font("arial.ttf", 20)
    small_font = font("arial.ttf", 17)
    bold_font = font("arialbd.ttf", 20)

    left, top, right, bottom = [v * scale for v in (145, 310, 1880, 1000)]
    q25 = np.vstack(
        [np.quantile(curves[labels == cluster], 0.25, axis=0) for cluster in range(1, 4)]
    )
    q75 = np.vstack(
        [np.quantile(curves[labels == cluster], 0.75, axis=0) for cluster in range(1, 4)]
    )
    extent = max(abs(float(q25.min())), abs(float(q75.max())), 1.5)
    y_limit = np.ceil((extent + 0.1) * 2) / 2

    def xy(progress: float, value: float) -> tuple[int, int]:
        x = left + progress * (right - left)
        y = bottom - (value + y_limit) / (2 * y_limit) * (bottom - top)
        return int(x), int(y)

    draw.text((left, 38 * scale), "Career Elo trajectory archetypes", fill="#111827", font=title_font)
    draw.text(
        (left, 100 * scale),
        "Players entering the NBA in 1976-77 or later; five 40+ game seasons or eight total seasons",
        fill="#4B5563",
        font=subtitle_font,
    )
    draw.text(
        (left, 136 * scale),
        "Current players under age 35 are excluded; shaded bands show the middle 50% within each group",
        fill="#4B5563",
        font=subtitle_font,
    )
    draw.text(
        (left, 174 * scale),
        "Each player's curve is standardized, so the vertical axis compares career shape rather than ability",
        fill="#6B7280",
        font=body_font,
    )

    for tick in np.arange(-y_limit, y_limit + 0.01, 0.5):
        _, y = xy(0, float(tick))
        draw.line((left, y, right, y), fill="#D1D5DB", width=2 * scale)
        label = f"{tick:.1f}"
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((left - (box[2] - box[0]) - 14 * scale, y - 10 * scale), label, fill="#6B7280", font=small_font)

    for pct in [0, 25, 50, 75, 100]:
        x, _ = xy(pct / 100, -y_limit)
        draw.line((x, top, x, bottom), fill="#E5E7EB", width=1 * scale)
        label = f"{pct}%"
        box = draw.textbbox((0, 0), label, font=body_font)
        draw.text((x - (box[2] - box[0]) / 2, bottom + 19 * scale), label, fill="#4B5563", font=body_font)
        age_idx = int(round(pct / 100 * (len(GRID) - 1)))
        age_label = f"~{age_grid[age_idx]:.1f}"
        age_box = draw.textbbox((0, 0), age_label, font=small_font)
        draw.text((x - (age_box[2] - age_box[0]) / 2, top - 32 * scale), age_label, fill="#6B7280", font=small_font)

    draw.text((left, top - 66 * scale), "Approximate median age", fill="#6B7280", font=small_font)

    for cluster in range(1, 4):
        rgb = hex_to_rgb(COLORS[cluster])
        upper = [xy(float(p), float(v)) for p, v in zip(GRID, q75[cluster - 1])]
        lower = [xy(float(p), float(v)) for p, v in zip(GRID[::-1], q25[cluster - 1][::-1])]
        draw.polygon(upper + lower, fill=(*rgb, 35))
        points = [xy(float(p), float(v)) for p, v in zip(GRID, centers[cluster - 1])]
        draw.line(points, fill=(*rgb, 255), width=6 * scale, joint="curve")
        peak_idx = int(np.argmax(centers[cluster - 1]))
        px, py = xy(float(GRID[peak_idx]), float(centers[cluster - 1][peak_idx]))
        radius = 7 * scale
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(*rgb, 255))
        peak_age = float(summary.loc[summary["cluster"] == cluster, "age_at_center_peak"].iloc[0])
        peak_text = f"{GRID[peak_idx] * 100:.1f}%  (~{peak_age:.1f})"
        draw.text((px + 12 * scale, py - 28 * scale), peak_text, fill=(*rgb, 255), font=bold_font)

    legend_xs = [145 * scale, 760 * scale, 1390 * scale]
    legend_y = 218 * scale
    for cluster in range(1, 4):
        rgb = hex_to_rgb(COLORS[cluster])
        n = int((labels == cluster).sum())
        legend_x = legend_xs[cluster - 1]
        y = legend_y
        draw.line((legend_x, y + 10 * scale, legend_x + 42 * scale, y + 10 * scale), fill=(*rgb, 255), width=6 * scale)
        draw.text(
            (legend_x + 55 * scale, y),
            f"{CLUSTER_NAMES[cluster]}  (n={n})",
            fill="#374151",
            font=body_font,
        )

    draw.text(((left + right) // 2 - 75 * scale, 1080 * scale), "Career progress", fill="#374151", font=bold_font)
    axis_label = Image.new("RGBA", (300 * scale, 34 * scale), (0, 0, 0, 0))
    axis_draw = ImageDraw.Draw(axis_label)
    axis_draw.text((0, 0), "Within-player standardized Elo", fill="#4B5563", font=small_font)
    axis_label = axis_label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    img.paste(
        axis_label,
        (18 * scale, int((top + bottom - axis_label.height) / 2)),
        axis_label,
    )
    draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2 * scale)

    img = img.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    img.save(path, quality=95)


def draw_cluster_count_selection(selection: pd.DataFrame, path: Path) -> None:
    """Draw the two standard diagnostics used to compare candidate k values."""
    scale = 2
    width, height = 1800 * scale, 900 * scale
    img = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(img, "RGBA")
    font_dir = Path("C:/Windows/Fonts")

    def font(name: str, size: int):
        return ImageFont.truetype(str(font_dir / name), size * scale)

    title_font = font("arialbd.ttf", 42)
    subtitle_font = font("arial.ttf", 21)
    body_font = font("arial.ttf", 19)
    small_font = font("arial.ttf", 16)
    bold_font = font("arialbd.ttf", 19)

    draw.text((105 * scale, 42 * scale), "Choosing the number of career-trajectory clusters", fill="#111827", font=title_font)
    draw.text(
        (105 * scale, 100 * scale),
        "The silhouette score favors k = 2; inertia shows the diminishing improvement from adding more groups",
        fill="#4B5563",
        font=subtitle_font,
    )

    panels = [
        {
            "box": (105, 205, 835, 720),
            "column": "silhouette",
            "title": "Silhouette score",
            "subtitle": "Higher is better",
            "color": "#2563EB",
            "format": lambda value: f"{value:.2f}",
        },
        {
            "box": (965, 205, 1695, 720),
            "column": "inertia",
            "title": "Within-cluster inertia",
            "subtitle": "Lower is better; look for an elbow",
            "color": "#C2410C",
            "format": lambda value: f"{value / 1000:.0f}k",
        },
    ]

    for panel in panels:
        left, top, right, bottom = [v * scale for v in panel["box"]]
        values = selection[panel["column"]].to_numpy(float)
        ks = selection["k"].to_numpy(int)
        value_min, value_max = float(values.min()), float(values.max())
        pad = max((value_max - value_min) * 0.18, 0.01)
        y_min, y_max = value_min - pad, value_max + pad

        draw.text((left, 155 * scale), panel["title"], fill="#111827", font=bold_font)
        draw.text((left + 220 * scale, 157 * scale), panel["subtitle"], fill="#6B7280", font=small_font)

        def xy(k: int, value: float) -> tuple[int, int]:
            x = left + (k - ks.min()) / (ks.max() - ks.min()) * (right - left)
            y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
            return int(x), int(y)

        for frac in np.linspace(0, 1, 5):
            value = y_min + frac * (y_max - y_min)
            _, y = xy(int(ks.min()), value)
            draw.line((left, y, right, y), fill="#D1D5DB", width=1 * scale)
            label = panel["format"](value)
            box = draw.textbbox((0, 0), label, font=small_font)
            draw.text((left - (box[2] - box[0]) - 12 * scale, y - 9 * scale), label, fill="#6B7280", font=small_font)

        points = [xy(int(k), float(value)) for k, value in zip(ks, values)]
        draw.line(points, fill=panel["color"], width=5 * scale, joint="curve")
        for k, value, (x, y) in zip(ks, values, points):
            radius = 6 * scale
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=panel["color"])
            value_label = f"{value:.3f}" if panel["column"] == "silhouette" else f"{value / 1000:.1f}k"
            label_box = draw.textbbox((0, 0), value_label, font=small_font)
            label_x = min(max(x - (label_box[2] - label_box[0]) / 2, left), right - (label_box[2] - label_box[0]))
            draw.text((label_x, y - 29 * scale), value_label, fill="#374151", font=small_font)
            k_label = str(k)
            k_box = draw.textbbox((0, 0), k_label, font=body_font)
            draw.text((x - (k_box[2] - k_box[0]) / 2, bottom + 16 * scale), k_label, fill="#4B5563", font=body_font)

        draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2 * scale)
        x_label = "Number of clusters (k)"
        x_box = draw.textbbox((0, 0), x_label, font=body_font)
        draw.text(((left + right - (x_box[2] - x_box[0])) / 2, bottom + 55 * scale), x_label, fill="#374151", font=body_font)

    best = selection.loc[selection["silhouette"].idxmax()]
    note = f"Highest silhouette: k = {int(best['k'])} ({best['silhouette']:.3f})"
    draw.text((105 * scale, 815 * scale), note, fill="#2563EB", font=bold_font)
    img = img.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    img.save(path, quality=95)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    players = pd.read_csv(
        PLAYERS,
        usecols=[
            "personId", "firstName", "lastName", "birthDate", "fromYear", "toYear",
            "draftYear", "draftRound", "draftNumber",
        ],
    ).rename(columns={"personId": "player_id"})
    players["birthDate"] = pd.to_datetime(players["birthDate"], errors="coerce")
    players["full_name"] = players["firstName"].fillna("") + " " + players["lastName"].fillna("")

    diagnostics = pd.read_csv(DIAGNOSTICS).set_index("season")
    season_mean = diagnostics["minutes_wmean"].to_dict()
    updates = pd.read_csv(
        UPDATES,
        usecols=["player_id", "game_date", "season", "rating_after"],
        parse_dates=["game_date"],
    ).sort_values(["player_id", "game_date"])
    updates["relative_elo"] = updates["rating_after"] - updates["season"].map(season_mean)
    latest_season = int(updates["season"].max())

    meta = updates.groupby("player_id").agg(
        games=("rating_after", "size"),
        seasons_observed=("season", "nunique"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        first_game=("game_date", "min"),
        last_game=("game_date", "max"),
    ).reset_index().merge(players, on="player_id", how="left").set_index("player_id")
    meta["debut_age"] = (meta["first_game"] - meta["birthDate"]).dt.days / 365.2425
    meta["age_at_last"] = (meta["last_game"] - meta["birthDate"]).dt.days / 365.2425

    season_games = updates.groupby(["player_id", "season"]).size().rename("games_in_season").reset_index()
    qualifying = (
        season_games[season_games["games_in_season"] >= MIN_GAMES_PER_QUALIFYING_SEASON]
        .groupby("player_id").size().rename("qualifying_seasons")
    )
    meta = meta.join(qualifying).fillna({"qualifying_seasons": 0})
    meta["full_start_observed"] = (
        (meta["first_season"] >= 1976) & (meta["first_season"] == meta["fromYear"])
    )
    meta["appeared_latest_season"] = meta["last_season"] == latest_season
    meta["provisional_active"] = meta["appeared_latest_season"] & (meta["age_at_last"] >= ACTIVE_MIN_AGE)
    meta["eligible_end"] = (~meta["appeared_latest_season"]) | meta["provisional_active"]
    keep = meta[
        meta["full_start_observed"]
        & meta["eligible_end"]
        & (
            (meta["qualifying_seasons"] >= MIN_QUALIFYING_SEASONS)
            | (meta["seasons_observed"] >= 8)
        )
    ].index

    birth_map = meta["birthDate"]
    updates["birthDate"] = updates["player_id"].map(birth_map)
    updates["age"] = (updates["game_date"] - updates["birthDate"]).dt.days / 365.2425

    ids, curves, age_curves, peak_rows = [], [], [], []
    for player_id, group in updates[updates["player_id"].isin(keep)].groupby("player_id", sort=True):
        prepared = prepare_curve(group.dropna(subset=["age", "relative_elo"]))
        if prepared is None:
            continue
        ids.append(int(player_id))
        curves.append(prepared["standardized"])
        age_curves.append(prepared["sampled_age"])
        peak_rows.append(
            {
                "player_id": int(player_id),
                "individual_peak_progress": prepared["peak_progress"],
                "individual_peak_date": prepared["peak_date"],
                "individual_peak_age": prepared["peak_age"],
                "individual_peak_relative_elo_21g": prepared["peak_relative_elo_21g"],
                "individual_peak_elo_21g": prepared["peak_elo_21g"],
            }
        )
    x = np.vstack(curves)
    age_x = np.vstack(age_curves)

    selection_rows = []
    fits = {}
    for k in range(2, 9):
        labels, centers, inertia = kmeans(x, k, seed=42, n_init=80)
        selection_rows.append(
            {"k": k, "silhouette": silhouette(x, labels), "inertia": inertia}
        )
        fits[k] = (labels, centers)
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(OUT / "cluster_count_selection.csv", index=False)
    draw_cluster_count_selection(selection, OUT / "cluster_count_selection.png")

    labels, centers = fits[3]
    order = np.argsort([GRID[int(np.argmax(center))] for center in centers])
    centers = centers[order]
    remap = {int(old): int(new) + 1 for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in labels])

    membership = pd.DataFrame({"player_id": ids, "cluster": labels}).merge(
        pd.DataFrame(peak_rows), on="player_id", how="left"
    ).merge(meta.reset_index(), on="player_id", how="left")
    membership["cluster_name"] = membership["cluster"].map(CLUSTER_NAMES)
    membership["distance_to_cluster_center"] = [
        float(np.sqrt(np.square(curve - centers[label - 1]).sum()))
        for curve, label in zip(x, labels)
    ]
    membership = membership[
        [
            "player_id", "full_name", "cluster", "cluster_name",
            "distance_to_cluster_center", "individual_peak_progress", "individual_peak_date",
            "individual_peak_age", "individual_peak_relative_elo_21g", "individual_peak_elo_21g",
            "games", "seasons_observed", "qualifying_seasons", "first_season", "last_season",
            "debut_age", "age_at_last", "draftYear", "draftRound", "draftNumber",
            "appeared_latest_season", "provisional_active",
        ]
    ].sort_values(["cluster", "distance_to_cluster_center"])
    membership.to_csv(OUT / "player_career_trajectory_classification.csv", index=False)

    age_grid = np.nanmedian(age_x, axis=0)
    age_mapping = pd.DataFrame(
        {
            "career_progress": GRID,
            "median_age": np.nanmedian(age_x, axis=0),
            "age_p25": np.nanquantile(age_x, 0.25, axis=0),
            "age_p75": np.nanquantile(age_x, 0.75, axis=0),
        }
    )
    age_mapping.to_csv(OUT / "career_progress_age_mapping.csv", index=False)

    center_rows, band_rows, summary_rows, representative_rows = [], [], [], []
    for cluster in range(1, 4):
        mask = labels == cluster
        center = centers[cluster - 1]
        peak_index = int(np.argmax(center))
        members = membership[membership["cluster"] == cluster]
        center_row = {
            "cluster": cluster,
            "cluster_name": CLUSTER_NAMES[cluster],
            "players": int(mask.sum()),
            "center_peak_progress": float(GRID[peak_index]),
        }
        center_row.update({f"p{int(p * 100):03d}": float(v) for p, v in zip(GRID, center)})
        center_rows.append(center_row)
        for idx, progress in enumerate(GRID):
            band_rows.append(
                {
                    "cluster": cluster,
                    "cluster_name": CLUSTER_NAMES[cluster],
                    "career_progress": progress,
                    "center": center[idx],
                    "p25": np.quantile(x[mask, idx], 0.25),
                    "median": np.quantile(x[mask, idx], 0.50),
                    "p75": np.quantile(x[mask, idx], 0.75),
                }
            )
        valid_draft = members["draftNumber"].dropna()
        summary_rows.append(
            {
                "cluster": cluster,
                "cluster_name": CLUSTER_NAMES[cluster],
                "players": len(members),
                "provisional_active_players": int(members["provisional_active"].sum()),
                "center_peak_progress": float(GRID[peak_index]),
                "age_at_center_peak": float(np.nanmedian(age_x[mask, peak_index])),
                "individual_peak_progress_median": members["individual_peak_progress"].median(),
                "individual_peak_age_median": members["individual_peak_age"].median(),
                "debut_age_median": members["debut_age"].median(),
                "last_age_median": members["age_at_last"].median(),
                "career_games_median": members["games"].median(),
                "qualifying_seasons_median": members["qualifying_seasons"].median(),
                "draft_pick_median": valid_draft.median(),
                "top_14_draft_share": members["draftNumber"].between(1, 14).mean(),
            }
        )
        reps = members.nsmallest(15, "distance_to_cluster_center").copy()
        reps.insert(0, "representative_rank", range(1, len(reps) + 1))
        representative_rows.append(
            reps[
                [
                    "representative_rank", "cluster", "cluster_name", "player_id", "full_name",
                    "distance_to_cluster_center", "games", "qualifying_seasons",
                    "individual_peak_progress", "individual_peak_age", "provisional_active",
                ]
            ]
        )

    pd.DataFrame(center_rows).to_csv(OUT / "cluster_centers.csv", index=False)
    pd.DataFrame(band_rows).to_csv(OUT / "cluster_curve_bands.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "cluster_summary.csv", index=False)
    pd.concat(representative_rows, ignore_index=True).to_csv(
        OUT / "representative_players.csv", index=False
    )

    draw_chart(
        curves=x,
        labels=labels,
        centers=centers,
        age_grid=age_grid,
        summary=summary,
        path=OUT / "career_trajectory_archetypes_final.png",
    )

    summary_lines = [
        "Career trajectory clustering - final five-40-game-seasons-or-eight-total-seasons rule",
        "",
        "Eligibility:",
        "- NBA debut in 1976-77 or later, with the career start fully observed.",
        "- At least five seasons with 40 or more appearances, or at least eight total NBA seasons.",
        "- Players appearing in the latest season are included only at age 35 or older and remain provisional.",
        "",
        f"Players: {len(membership)}",
        f"Latest season in data: {latest_season}",
        f"Provisional active players: {int(membership['provisional_active'].sum())}",
        f"Three-cluster silhouette: {selection.loc[selection['k'] == 3, 'silhouette'].iloc[0]:.4f}",
        "",
        "Cluster summary:",
    ]
    for row in summary.itertuples():
        summary_lines.append(
            f"- {row.cluster_name}: {row.players} players; center peak at "
            f"{row.center_peak_progress * 100:.1f}% of career (~age {row.age_at_center_peak:.1f})."
        )
    (OUT / "result_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print("\nCluster-count selection")
    print(selection.round(4).to_string(index=False))
    print("\nDetailed summary")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
