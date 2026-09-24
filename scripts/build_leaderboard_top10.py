"""Build canonical top-10 leaderboard snapshots and 7-day animation frames.

Outputs:
    results/leaderboard/top10_daily_canonical_pk1.csv
    results/leaderboard/frames_7d_canonical_pk1.csv
    results/leaderboard/frames_7d_canonical_pk1_meta.csv
    results/leaderboard/frames_7d_canonical_pk1.json
    results/leaderboard/preview.html

The daily file records the top 10 players after each NBA game date for the
canonical playoff_k=1.0 timeline.  The frame file samples those daily dates
every 7 calendar days within each season, plus each season's first and last
game date, so the video can hold each frame for a few seconds.

Usage:
    python scripts/build_leaderboard_top10.py
"""

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
LEADERBOARD = RESULTS / "leaderboard"
DB_PATH = RESULTS / "nba_elo.db"
INITIAL_STATE = RESULTS / "final" / "final_elo_initial_state_1976_canonical.csv"
HEADSHOT_URL = "https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"


def parse_date(value):
    return date.fromisoformat(value[:10])


def season_label(season_start):
    return f"{season_start}-{str(season_start + 1)[2:]}"


def normalize_season_start(raw_season, season_type, game_date):
    """Fix mislabeled playoff season buckets without breaking valid delayed playoffs.

    Some playoff games that occur in July can be labeled with the same integer
    as the calendar year they occur in, which incorrectly moves them into the
    following season bucket for a live leaderboard.

    But delayed playoffs like August 2020 can still correctly belong to the
    prior season bucket already, so we should only shift when the raw season
    equals the game's calendar year.
    """
    season_start = int(raw_season)
    game_year = parse_date(game_date).year
    if season_type == "Playoffs" and season_start == game_year:
        season_start -= 1
    return season_start


def load_initial_ratings(path):
    initial = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            initial[int(row["player_id"])] = float(row["rating"])
    return initial


def load_players(con):
    return {
        row[0]: row[1]
        for row in con.execute("SELECT player_id, full_name FROM players")
        if row[1]
    }


def build_daily_top10(con, variant, initial, players):
    track = variant.split("_pk")[0]
    sql = """
        SELECT re.game_id, re.game_date, re.season, g.season_type, re.player_id, re.rating_after
        FROM rating_events re
        JOIN games g
          ON g.game_id = re.game_id
        WHERE playoff_k = 1.0 AND source IN ('shared', ?)
        ORDER BY re.game_date, re.game_id, re.player_id
    """
    daily_rows = []
    daily_dates = []
    current = dict(initial)
    season_active = set()
    season_games = defaultdict(int)
    current_season = None
    current_date = None

    def flush(date_str, season_start):
        ranked = sorted(
            (
                (player_id, rating, season_games.get(player_id, 0))
                for player_id, rating in current.items()
                if player_id in season_active
            ),
            key=lambda row: (-row[1], row[0]),
        )
        for rank, (player_id, rating, games) in enumerate(ranked[:10], 1):
            daily_rows.append(
                {
                    "frame_date": date_str,
                    "season_start": season_start,
                    "season": season_start + 1,
                    "season_label": season_label(season_start),
                    "rank": rank,
                    "player_id": player_id,
                    "player_name": players.get(player_id, str(player_id)),
                    "rating": rating,
                    "games_this_season": games,
                }
            )

    for game_id, game_date, season, season_type, player_id, rating_after in con.execute(sql, (track,)):
        date_str = game_date[:10]
        season_start = normalize_season_start(season, season_type, game_date)
        if date_str != current_date:
            if current_date is not None:
                flush(current_date, current_season)
                daily_dates.append((date.fromisoformat(current_date), current_season))
            current_date = date_str
        if season_start != current_season:
            current_season = season_start
            season_active = set()
            if season_start == 1976:
                season_active.update(initial)
            season_games.clear()
        current[player_id] = rating_after
        season_active.add(player_id)
        season_games[player_id] += 1

    if current_date is not None:
        flush(current_date, current_season)
        daily_dates.append((date.fromisoformat(current_date), current_season))

    return daily_rows, daily_dates


def group_by_date(daily_rows):
    grouped = {}
    for row in daily_rows:
        grouped.setdefault(row["frame_date"], []).append(row)
    return grouped


def select_frames(daily_dates, interval_days):
    frame_dates = []
    season_start = None
    season_dates = []

    def flush_season():
        nonlocal season_dates
        if not season_dates:
            return
        first = season_dates[0]
        last = season_dates[-1]
        selected = [first]
        for d in season_dates[1:]:
            if (d - selected[-1]).days >= interval_days:
                selected.append(d)
        if selected[-1] != last:
            selected.append(last)
        for d in selected:
            role = []
            if d == first:
                role.append("season_first")
            if d == last:
                role.append("season_last")
            if not role:
                role.append("weekly")
            frame_dates.append((d, "_".join(role)))

    for d, season in daily_dates:
        if season_start is None:
            season_start = season
            season_dates = []
        if season != season_start:
            flush_season()
            season_start = season
            season_dates = []
        season_dates.append(d)
    flush_season()
    return frame_dates


def write_daily_csv(rows, path):
    fields = [
        "frame_date", "season_start", "season", "season_label",
        "rank", "player_id", "player_name", "rating", "games_this_season",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_frame_outputs(daily_rows, frame_dates, out_dir, hold_seconds, transition_seconds, interval_days):
    by_date = group_by_date(daily_rows)
    frame_rows = []
    meta_rows = []
    frames = []
    prev_rank = {}
    prev_rating = {}
    cumulative = 0.0

    for frame_id, (frame_date, role) in enumerate(frame_dates):
        date_str = frame_date.isoformat()
        top_rows = by_date[date_str]
        season_start = top_rows[0]["season_start"]
        frame = {
            "frame_id": frame_id,
            "date": date_str,
            "season_start": season_start,
            "season": season_start + 1,
            "season_label": season_label(season_start),
            "role": role,
            "top": [],
        }
        current_rank = {}
        current_rating = {}
        for row in top_rows:
            player_id = row["player_id"]
            rating = row["rating"]
            current_rank[player_id] = row["rank"]
            current_rating[player_id] = rating
            prev = prev_rank.get(player_id)
            frame_rows.append(
                {
                    "frame_id": frame_id,
                    "frame_date": date_str,
                    "season_start": season_start,
                    "season": season_start + 1,
                    "season_label": season_label(season_start),
                    "frame_role": role,
                    "rank": row["rank"],
                    "player_id": player_id,
                    "player_name": row["player_name"],
                    "rating": rating,
                    "games_this_season": row["games_this_season"],
                    "previous_rank": prev if prev is not None else "",
                    "rank_change": (prev - row["rank"]) if prev is not None else "",
                    "rating_change": (
                        rating - prev_rating.get(player_id)
                        if player_id in prev_rating else ""
                    ),
                }
            )
            frame["top"].append(
                {
                    "rank": row["rank"],
                    "player_id": player_id,
                    "name": row["player_name"],
                    "rating": rating,
                    "games": row["games_this_season"],
                    "headshot": HEADSHOT_URL.format(player_id=player_id),
                }
            )

        segment_seconds = hold_seconds + transition_seconds
        meta_rows.append(
            {
                "frame_id": frame_id,
                "frame_date": date_str,
                "season_start": season_start,
                "season": season_start + 1,
                "season_label": season_label(season_start),
                "frame_role": role,
                "hold_seconds": hold_seconds,
                "transition_seconds": transition_seconds,
                "segment_seconds": segment_seconds,
                "start_seconds": cumulative,
                "end_seconds": cumulative + segment_seconds,
            }
        )
        cumulative += segment_seconds
        frames.append(frame)
        prev_rank = current_rank
        prev_rating = current_rating

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_fields = [
        "frame_id", "frame_date", "season_start", "season", "season_label",
        "frame_role", "rank", "player_id", "player_name", "rating",
        "games_this_season", "previous_rank", "rank_change", "rating_change",
    ]
    with open(out_dir / f"frames_{interval_days}d_canonical_pk1.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(frame_rows)

    meta_fields = [
        "frame_id", "frame_date", "season_start", "season", "season_label",
        "frame_role", "hold_seconds", "transition_seconds", "segment_seconds",
        "start_seconds", "end_seconds",
    ]
    with open(out_dir / f"frames_{interval_days}d_canonical_pk1_meta.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta_fields)
        writer.writeheader()
        writer.writerows(meta_rows)

    payload = {
        "variant": "canonical_pk1",
        "track": "canonical",
        "playoff_k": 1.0,
        "cadence_days": interval_days,
        "hold_seconds": hold_seconds,
        "transition_seconds": transition_seconds,
        "frame_count": len(frames),
        "total_seconds": cumulative,
        "frames": frames,
    }
    with open(out_dir / f"frames_{interval_days}d_canonical_pk1.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    return payload


def write_readme(out_dir, payload):
    text = """# Leaderboard data for the canonical top-10 animation

This folder contains as-of top-10 snapshots for `canonical_pk1` (K=45,
scale=285, alpha=1.0, playoff_k=1.0), which is the all-time comparable
single track.

Files
-----
- `top10_daily_canonical_pk1.csv`: top 10 after every NBA game date.
- `frames_7d_canonical_pk1.csv`: sampled 7-day frames in long format.
- `frames_7d_canonical_pk1_meta.csv`: one row per frame with video timing.
- `frames_7d_canonical_pk1.json`: nested frame data for a JS/After Effects
  style renderer.
- `preview.html`: self-contained browser preview; open it directly.

Cadence
-------
Frames are sampled from game dates. Within each season the first game date
is always a frame, then every 7 calendar days, then the season's final game
date is added so the year closes on the last top 10.

Timing
------
`hold_seconds` is the time the frame stays still. `transition_seconds` is
the morph time into the next frame. `start_seconds` and `end_seconds` are
cumulative video timestamps.

Headshots
---------
`headshot` URLs use the official NBA CDN:
https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png
Historical players may 404; fall back to initials or a local image set.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_preview(out_dir, payload):
    frames = payload["frames"]
    js_frames = json.dumps(frames, ensure_ascii=False)
    hold_ms = int(payload["hold_seconds"] * 1000)
    transition_ms = int(payload["transition_seconds"] * 1000)
    html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>NBA Elo Top 10 Preview</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0;
    background: #0b1020;
    color: #f2f5ff;
    font-family: "Segoe UI", Arial, sans-serif;
  }
  .wrap { max-width: 960px; margin: 0 auto; padding: 24px 20px 40px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .subtitle { color: #9aa7c7; font-size: 14px; margin-bottom: 16px; }
  .hero {
    display: flex;
    align-items: center;
    gap: 16px;
    background: #141b33;
    border: 1px solid #2a3558;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
  }
  .avatar {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    object-fit: cover;
    background: #27335a;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: 700;
    color: #f8c16c;
    flex: 0 0 64px;
    background-size: cover;
    background-position: center;
  }
  .hero-name { font-size: 28px; font-weight: 800; }
  .hero-rating { color: #f8c16c; font-size: 20px; font-weight: 700; margin-left: auto; }
  .hero-meta { color: #9aa7c7; font-size: 13px; }
  .board { display: flex; flex-direction: column; gap: 8px; }
  .row {
    display: grid;
    grid-template-columns: 40px 180px 1fr 76px;
    gap: 12px;
    align-items: center;
    background: #121832;
    border: 1px solid #242f50;
    border-radius: 6px;
    padding: 8px 12px;
    min-height: 38px;
  }
  .rank { font-size: 18px; font-weight: 800; color: #7e8cb8; }
  .rank-1 .rank { color: #f8c16c; }
  .name { font-size: 14px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .track { height: 18px; background: #1c2544; border-radius: 4px; overflow: hidden; }
  .bar {
    height: 100%;
    width: 0%;
    background: #3d6ef5;
    border-radius: 4px;
    transition: width 0.6s ease;
  }
  .rank-1 .bar { background: #f8a43c; }
  .rating { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }
  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-top: 18px;
  }
  button {
    background: #243153;
    color: #f2f5ff;
    border: 1px solid #3a4a78;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 14px;
    cursor: pointer;
  }
  button:hover { background: #2c3c66; }
  input[type=range] { flex: 1 1 240px; min-width: 180px; }
  select {
    background: #243153;
    color: #f2f5ff;
    border: 1px solid #3a4a78;
    border-radius: 6px;
    padding: 8px;
  }
  .counter { color: #9aa7c7; font-size: 13px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>NBA Player Elo Top 10</h1>
  <div class="subtitle" id="subtitle"></div>
  <div class="hero">
    <div class="avatar" id="heroAvatar"></div>
    <div>
      <div class="hero-name" id="heroName"></div>
      <div class="hero-meta" id="heroMeta"></div>
    </div>
    <div class="hero-rating" id="heroRating"></div>
  </div>
  <div class="board" id="board"></div>
  <div class="controls">
    <button id="prev">Prev</button>
    <button id="play">Play</button>
    <button id="next">Next</button>
    <select id="speed">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
    <input type="range" id="slider" min="0" max="0" value="0">
    <div class="counter" id="counter"></div>
  </div>
</div>
<script>
const FRAMES = __FRAMES__;
const HOLD = __HOLD_MS__;
const TRANSITION = __TRANSITION_MS__;
let index = 0;
let playing = false;
let timer = null;
let speed = 1;

const board = document.getElementById('board');
const slider = document.getElementById('slider');
const subtitle = document.getElementById('subtitle');
const counter = document.getElementById('counter');
slider.max = FRAMES.length - 1;

function initials(name) {
  return name.trim().split(/\\s+/).slice(0, 2).map(w => w[0].toUpperCase()).join('');
}

function render() {
  const frame = FRAMES[index];
  const top = frame.top;
  const first = top[0];
  subtitle.textContent = frame.date + '  ' + frame.season_label;
  heroName.textContent = first.name;
  heroRating.textContent = Math.round(first.rating);
  heroMeta.textContent = 'rank 1  ·  games ' + first.games + ' this season';
  const avatar = document.getElementById('heroAvatar');
  avatar.textContent = initials(first.name);
  avatar.style.backgroundImage = '';
  const img = new Image();
  img.onload = () => {
    avatar.textContent = '';
    avatar.style.backgroundImage = 'url(' + first.headshot + ')';
  };
  img.src = first.headshot;

  const maxRating = Math.max(...top.map(p => p.rating));
  while (board.firstChild) board.removeChild(board.firstChild);
  top.forEach(p => {
    const row = document.createElement('div');
    row.className = 'row' + (p.rank === 1 ? ' rank-1' : '');
    const rank = document.createElement('div');
    rank.className = 'rank';
    rank.textContent = p.rank;
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = p.name;
    const track = document.createElement('div');
    track.className = 'track';
    const bar = document.createElement('div');
    bar.className = 'bar';
    track.appendChild(bar);
    const rating = document.createElement('div');
    rating.className = 'rating';
    rating.textContent = Math.round(p.rating);
    row.appendChild(rank);
    row.appendChild(name);
    row.appendChild(track);
    row.appendChild(rating);
    board.appendChild(row);
    requestAnimationFrame(() => { bar.style.width = (p.rating / maxRating * 100).toFixed(2) + '%'; });
  });
  slider.value = index;
  counter.textContent = (index + 1) + ' / ' + FRAMES.length;
}

function playFrame() {
  if (!playing) return;
  render();
  index = (index + 1) % FRAMES.length;
  timer = setTimeout(playFrame, (HOLD + TRANSITION) / speed);
}

function stop() {
  playing = false;
  clearTimeout(timer);
  document.getElementById('play').textContent = 'Play';
}

function toggle() {
  playing = !playing;
  document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
  if (playing) {
    clearTimeout(timer);
    timer = setTimeout(playFrame, (HOLD + TRANSITION) / speed);
  } else {
    clearTimeout(timer);
  }
}

document.getElementById('prev').onclick = () => { stop(); index = (index - 1 + FRAMES.length) % FRAMES.length; render(); };
document.getElementById('next').onclick = () => { stop(); index = (index + 1) % FRAMES.length; render(); };
document.getElementById('play').onclick = toggle;
document.getElementById('speed').onchange = e => { speed = parseFloat(e.target.value); };
slider.oninput = e => { stop(); index = parseInt(e.target.value, 10); render(); };

render();
</script>
</body>
</html>
"""
    html = html.replace("__FRAMES__", js_frames)
    html = html.replace("__HOLD_MS__", str(hold_ms))
    html = html.replace("__TRANSITION_MS__", str(transition_ms))
    (out_dir / "preview.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out", type=Path, default=LEADERBOARD)
    parser.add_argument("--variant", default="canonical_pk1")
    parser.add_argument("--interval-days", type=int, default=7)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--transition-seconds", type=float, default=0.6)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(args.db)
    initial = load_initial_ratings(INITIAL_STATE)
    players = load_players(con)

    print("building daily as-of top 10...")
    daily_rows, daily_dates = build_daily_top10(con, args.variant, initial, players)
    print(f"daily top-10 rows: {len(daily_rows)}")
    print(f"daily game dates: {len(daily_dates)}")

    print("sampling 7-day frames...")
    frame_dates = select_frames(daily_dates, args.interval_days)
    print(f"frame dates: {len(frame_dates)}")

    write_daily_csv(daily_rows, args.out / "top10_daily_canonical_pk1.csv")
    payload = write_frame_outputs(
        daily_rows, frame_dates, args.out,
        args.hold_seconds, args.transition_seconds,
        args.interval_days,
    )
    write_readme(args.out, payload)
    write_preview(args.out, payload)
    con.close()

    print(f"wrote outputs to {args.out}")
    print(
        f"total video time at {args.hold_seconds:.1f}s hold + "
        f"{args.transition_seconds:.1f}s transition: "
        f"{payload['total_seconds'] / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
