from pathlib import Path
import sqlite3

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DB = ROOT / "results" / "nba_elo.db"
OUT = ROOT / "results" / "analysis" / "4.2_career_trajectories"

PLAYERS = [
    "Michael Jordan",
    "LeBron James",
    "Shaquille O'Neal",
    "Nikola Jokic",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB) as con:
        marks = ",".join("?" for _ in PLAYERS)
        players = pd.read_sql_query(
            f"""
            SELECT player_id, full_name, birth_date
            FROM players
            WHERE full_name IN ({marks})
            """,
            con,
            params=PLAYERS,
        )

    if set(players["full_name"]) != set(PLAYERS):
        missing = sorted(set(PLAYERS) - set(players["full_name"]))
        raise ValueError(f"Missing players: {missing}")

    wanted_ids = set(players["player_id"].astype(int))
    parts = []
    usecols = ["game_date", "season", "track", "player_id", "rating_before", "rating_after"]
    for chunk in pd.read_csv(UPDATES, usecols=usecols, chunksize=400_000):
        keep = chunk[chunk["player_id"].isin(wanted_ids)]
        if not keep.empty:
            parts.append(keep)

    events = pd.concat(parts, ignore_index=True)
    events["game_date"] = pd.to_datetime(events["game_date"])
    players["birth_date"] = pd.to_datetime(players["birth_date"])
    events = events.merge(players, on="player_id", how="left", validate="many_to_one")
    events = events.sort_values(["full_name", "game_date", "season"])
    events["age"] = (events["game_date"] - events["birth_date"]).dt.days / 365.2425

    monthly = (
        events.assign(month=events["game_date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["full_name", "month"], as_index=False)
        .last()
    )

    summaries = []
    for name in PLAYERS:
        g = events[events["full_name"] == name].reset_index(drop=True)
        peak_row = g.loc[g["rating_after"].idxmax()]
        summaries.append(
            {
                "player": name,
                "first_game": g.iloc[0]["game_date"].date().isoformat(),
                "last_game": g.iloc[-1]["game_date"].date().isoformat(),
                "games_in_model": len(g),
                "starting_elo": round(float(g.iloc[0]["rating_before"]), 2),
                "peak_elo": round(float(peak_row["rating_after"]), 2),
                "peak_date": peak_row["game_date"].date().isoformat(),
                "age_at_peak": round(float(peak_row["age"]), 1),
                "games_to_peak": int(peak_row.name + 1),
            }
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "selected_career_trajectory_summary.csv", index=False)
    monthly[["full_name", "month", "age", "rating_after"]].to_csv(
        OUT / "selected_career_trajectory_monthly.csv", index=False
    )

    draw_chart(monthly, summary, OUT / "selected_career_trajectories.png")

    print(summary.to_string(index=False))


def draw_chart(monthly: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    colors = {
        "Michael Jordan": "#C8102E",
        "LeBron James": "#6F263D",
        "Shaquille O'Neal": "#552583",
        "Nikola Jokic": "#0E2240",
    }
    width, height = 1800, 1120
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 48)
        panel_font = ImageFont.truetype("arialbd.ttf", 31)
        body_font = ImageFont.truetype("arial.ttf", 23)
        small_font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        title_font = panel_font = body_font = small_font = ImageFont.load_default()

    draw.text((85, 45), "Four different paths to an all-time Elo peak", fill="#111111", font=title_font)
    panels = [(85, 145, 865, 575), (955, 145, 1735, 575), (85, 650, 865, 1080), (955, 650, 1735, 1080)]
    x_min, x_max, y_min, y_max = 18.0, 42.0, 1200.0, 2360.0

    def point(age: float, elo: float, box: tuple[int, int, int, int]) -> tuple[int, int]:
        left, top, right, bottom = box
        plot_l, plot_t, plot_r, plot_b = left + 68, top + 65, right - 20, bottom - 55
        x = plot_l + (age - x_min) / (x_max - x_min) * (plot_r - plot_l)
        y = plot_b - (elo - y_min) / (y_max - y_min) * (plot_b - plot_t)
        return int(x), int(y)

    for box, name in zip(panels, PLAYERS):
        left, top, right, bottom = box
        draw.text((left, top), name, fill="#111111", font=panel_font)
        for tick in [1400, 1800, 2200]:
            x0, y = point(x_min, tick, box)
            x1, _ = point(x_max, tick, box)
            draw.line((x0, y, x1, y), fill="#DDDDDD", width=2)
            draw.text((left, y - 12), str(tick), fill="#666666", font=small_font)
        for tick in [20, 25, 30, 35, 40]:
            x, y0 = point(tick, y_min, box)
            draw.text((x - 12, y0 + 12), str(tick), fill="#666666", font=small_font)

        g = monthly[monthly["full_name"] == name].sort_values("age").copy()
        g["segment"] = g["month"].diff().dt.days.gt(120).cumsum()
        for _, segment in g.groupby("segment"):
            pts = [point(float(row.age), float(row.rating_after), box) for row in segment.itertuples()]
            if len(pts) > 1:
                draw.line(pts, fill=colors[name], width=5, joint="curve")
        s = summary[summary["player"] == name].iloc[0]
        px, py = point(float(s["age_at_peak"]), float(s["peak_elo"]), box)
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=colors[name])
        label = f"Peak {s['peak_elo']:.0f} at age {s['age_at_peak']:.1f}"
        tx = min(px + 12, right - 260)
        ty = max(top + 52, py - 34)
        draw.text((tx, ty), label, fill=colors[name], font=body_font)

    draw.text((835, 1080), "Player age", fill="#444444", font=body_font)
    canvas.save(path)


if __name__ == "__main__":
    main()
