from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
PLAYERS = ROOT / "processed_data" / "players.csv"
OUT = ROOT / "results" / "analysis" / "4.3_career_curve_clusters"

MIN_GAMES = 600
MIN_SEASONS = 8
FIRST_SEASON_MIN = 1976
LAST_SEASON_MAX = 2023
ACTIVE_SEASON_MIN = 2024
ACTIVE_AGE_MIN = 33.0
GRID = np.linspace(0.0, 1.0, 41)


def kmeans(x: np.ndarray, k: int, seed: int = 42, n_init: int = 40) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = x[rng.choice(len(x), size=k, replace=False)].copy()
        labels = np.zeros(len(x), dtype=int)
        for _ in range(150):
            dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                members = x[labels == j]
                centers[j] = members.mean(axis=0) if len(members) else x[rng.integers(len(x))]
        inertia = float(((x - centers[labels]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (labels.copy(), centers.copy(), inertia)
    assert best is not None
    return best


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    sq = (x * x).sum(axis=1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0))
    vals = []
    for i in range(len(x)):
        same = labels == labels[i]
        a = d[i, same].sum() / max(int(same.sum()) - 1, 1)
        b = min(d[i, labels == other].mean() for other in np.unique(labels) if other != labels[i])
        vals.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(vals))


def draw_chart(curves: np.ndarray, labels: np.ndarray, centers: np.ndarray, path: Path) -> None:
    width, height = 1800, 1080
    img = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(img)
    font_dir = Path("C:/Windows/Fonts")
    title_font = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 46)
    subtitle_font = ImageFont.truetype(str(font_dir / "arial.ttf"), 23)
    body_font = ImageFont.truetype(str(font_dir / "arial.ttf"), 21)
    bold_font = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 24)
    colors = ["#C2410C", "#2563EB", "#2E8B57", "#7C3AED", "#B45309", "#0F766E"]
    archetypes = (
        ["Early peak and long decline", "Conventional mid-career peak", "Late-developing peak"]
        if len(centers) == 3
        else [f"Cluster {j + 1}" for j in range(len(centers))]
    )

    left, top, right, bottom = 135, 180, 1690, 900
    y_min, y_max = -1.8, 1.8

    def xy(progress: float, value: float) -> tuple[int, int]:
        x = left + progress * (right - left)
        y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        return int(x), int(y)

    draw.text((left, 42), "Career Elo trajectory archetypes", fill="#111827", font=title_font)
    draw.text(
        (left, 104),
        "Completed careers plus late-career active players; 600+ games and 8+ seasons",
        fill="#4B5563",
        font=subtitle_font,
    )
    draw.text(
        (left, 137),
        "Vertical scale is standardized within each player, so the chart compares shape rather than ability",
        fill="#6B7280",
        font=body_font,
    )
    for val in [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]:
        _, y = xy(0, val)
        draw.line((left, y, right, y), fill="#D1D5DB", width=2)
        draw.text((72, y - 11), f"{val:.1f}", fill="#6B7280", font=body_font)
    for pct in [0, 25, 50, 75, 100]:
        x, _ = xy(pct / 100, y_min)
        draw.text((x - 18, bottom + 22), f"{pct}%", fill="#6B7280", font=body_font)

    for j, center in enumerate(centers):
        pts = [xy(float(p), float(v)) for p, v in zip(GRID, center)]
        draw.line(pts, fill=colors[j], width=7, joint="curve")

    lx, ly = 980, 200
    for j in range(len(centers)):
        n = int((labels == j).sum())
        peak_pct = int(round(GRID[int(np.argmax(centers[j]))] * 100))
        draw.line((lx, ly + 9, lx + 38, ly + 9), fill=colors[j], width=7)
        draw.text((lx + 52, ly - 4), f"{archetypes[j]}: {n} players, peak near {peak_pct}%", fill="#374151", font=body_font)
        ly += 42

    draw.text((left + 650, 950), "Career progress", fill="#4B5563", font=bold_font)
    img.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diagnostics = pd.read_csv(DIAGNOSTICS).set_index("season")
    season_mean = diagnostics["minutes_wmean"].to_dict()
    updates = pd.read_csv(
        UPDATES,
        usecols=["player_id", "game_date", "season", "rating_after"],
        parse_dates=["game_date"],
    ).sort_values(["player_id", "game_date"])
    updates["relative_elo"] = updates["rating_after"] - updates["season"].map(season_mean)

    players = pd.read_csv(
        PLAYERS,
        usecols=["personId", "firstName", "lastName", "birthDate", "fromYear", "toYear"],
    ).rename(columns={"personId": "player_id"})
    players["full_name"] = players["firstName"].fillna("") + " " + players["lastName"].fillna("")
    players["birthDate"] = pd.to_datetime(players["birthDate"], errors="coerce")

    meta = updates.groupby("player_id").agg(
        games=("rating_after", "size"),
        seasons=("season", "nunique"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        last_game=("game_date", "max"),
    ).reset_index().merge(
        players[["player_id", "birthDate", "fromYear", "toYear"]], on="player_id", how="left"
    ).set_index("player_id")
    meta["age_at_last"] = (meta["last_game"] - meta["birthDate"]).dt.days / 365.2425
    meta["completed"] = (
        (meta["last_season"] <= LAST_SEASON_MAX)
        & (meta["last_season"] >= meta["toYear"] - 1)
    )
    meta["late_active"] = (
        (meta["last_season"] >= ACTIVE_SEASON_MIN)
        & (meta["age_at_last"] >= ACTIVE_AGE_MIN)
    )
    keep = meta[
        (meta["games"] >= MIN_GAMES)
        & (meta["seasons"] >= MIN_SEASONS)
        & (meta["first_season"] >= FIRST_SEASON_MIN)
        & (meta["first_season"] == meta["fromYear"])
        & (meta["completed"] | meta["late_active"])
    ].index
    updates = updates[updates["player_id"].isin(keep)]

    rows = []
    ids = []
    peak_progress = []
    for player_id, group in updates.groupby("player_id", sort=True):
        vals = group["relative_elo"].rolling(21, center=True, min_periods=1).mean().to_numpy()
        progress = np.linspace(0.0, 1.0, len(vals))
        curve = np.interp(GRID, progress, vals)
        sd = curve.std()
        if sd < 1e-9:
            continue
        standardized = (curve - curve.mean()) / sd
        rows.append(standardized)
        ids.append(int(player_id))
        peak_progress.append(float(GRID[int(np.argmax(standardized))]))
    curves = np.vstack(rows)

    scores = []
    fits = {}
    for k in range(3, 7):
        labels, centers, inertia = kmeans(curves, k)
        score = silhouette(curves, labels)
        scores.append({"k": k, "silhouette": score, "inertia": inertia})
        fits[k] = (labels, centers)
    score_df = pd.DataFrame(scores)
    best_k = int(score_df.loc[score_df["silhouette"].idxmax(), "k"])
    labels, centers = fits[best_k]
    order = np.argsort([GRID[int(np.argmax(center))] for center in centers])
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(label)] for label in labels])
    centers = centers[order]

    membership = pd.DataFrame(
        {"player_id": ids, "cluster": labels + 1, "individual_peak_progress": peak_progress}
    ).merge(players[["player_id", "full_name"]], on="player_id", how="left")
    membership = membership.merge(meta.reset_index(), on="player_id", how="left")
    membership.to_csv(OUT / "career_curve_cluster_membership_with_late_active.csv", index=False)
    score_df.to_csv(OUT / "career_curve_cluster_selection_with_late_active.csv", index=False)

    center_rows = []
    for j, center in enumerate(centers):
        row = {"cluster": j + 1, "players": int((labels == j).sum()), "peak_progress": GRID[int(np.argmax(center))]}
        row.update({f"p{int(p * 100):03d}": float(v) for p, v in zip(GRID, center)})
        center_rows.append(row)
    pd.DataFrame(center_rows).to_csv(OUT / "career_curve_cluster_centers_with_late_active.csv", index=False)
    draw_chart(curves, labels, centers, OUT / "career_curve_archetypes_with_late_active.png")

    print(score_df.round(4).to_string(index=False))
    print(f"best_k={best_k}; players={len(membership)}")
    print("late_active_included=")
    print(
        membership[membership["late_active"]]
        .sort_values("games", ascending=False)[["full_name", "games", "seasons", "age_at_last"]]
        .round({"age_at_last": 1})
        .to_string(index=False)
    )
    for cluster in range(1, best_k + 1):
        names = membership[membership["cluster"] == cluster].sort_values("games", ascending=False).head(12)
        print(f"\nCluster {cluster}")
        print(names[["full_name", "games", "first_season", "last_season"]].to_string(index=False))


if __name__ == "__main__":
    main()
