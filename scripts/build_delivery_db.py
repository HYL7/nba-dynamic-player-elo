"""Build results/nba_elo.db, the delivery database (D-036).

The database is the single source of truth for delivery queries.  It
contains every engine event from the final dual-track runs plus the
playoff_k variants, and materializes the derived season/snapshot layers.

Variant identity
----------------
Each row in `params` is one (track, playoff_k) combination:

    canonical_pk0    canonical_pk0.5  canonical_pk1  canonical_pk1.5  canonical_pk2
    modern_pk0       modern_pk0.5     modern_pk1     modern_pk1.5     modern_pk2

canonical_pk1/modern_pk1 reuse results/final from E-031; the other
playoff_k values come from results/playoff_variants (E-034).

Source segments
---------------
For every playoff_k value the pre-1996 events are identical for canonical
and modern (same shared checkpoint), so they are stored once with
source='shared'.  Continuation events (1996+) carry source='canonical' or
source='modern'.  A variant's full timeline is:

    shared(playoff_k) + continuation(playoff_k, track)

`run_variants` expands the stored segments into the 8 full timelines used
by `variant_timeline`, `season_ratings` and `snapshots`.

Usage:
    python scripts/build_delivery_db.py [--force] [--check-only] [--smoke]
"""

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / "results"
FINAL = RESULTS / "final"
VARIANTS = RESULTS / "playoff_variants"
CLEAN = PROJECT_ROOT / "clean_data"
SCRATCH = PROJECT_ROOT / "scratch"
DB_PATH = RESULTS / "nba_elo.db"
SMOKE_DB_PATH = SCRATCH / "nba_elo_smoke.db"

PLAYOFF_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]
TRACKS = ["canonical", "modern"]

CANONICAL = {
    "k": 45.0,
    "scale": 285.0,
    "theta": 0.0,
    "h": 70.0,
    "alpha_modern": 1.0,
    "alpha_ref": 1.0,
    "rookie_boost": 6.0,
    "rookie_tau": 60.0,
    "rookie_start": 1425.0,
    "on_court_mode": "raw",
}

MODERN = {
    **CANONICAL,
    "scale": 318.0,
    "alpha_modern": 0.9,
    "alpha_ref": 0.9,
}

EXPECTED_EVENTS = {
    "shared": 422_838,
    "canonical": 779_201,
    "modern": 779_201,
}
EXPECTED_TOTAL = sum(EXPECTED_EVENTS.values()) * len(PLAYOFF_VALUES)
EXPECTED_MINUTES = 1_202_039
EXPECTED_TIMELINE_EVENTS = 1_202_039

EVENT_COLUMNS = [
    "game_id", "game_date", "season", "track", "player_id", "team_id",
    "home", "rating_before", "rating_after", "delta_raw", "delta_adj",
    "zero_mean_offset", "minutes_share", "perf_i", "perf_used",
    "game_score", "z_game_score", "on_court_rate", "z_on_court_rate",
    "s_team", "e_team", "k_effective",
]

CORE_NUMERIC_COLUMNS = [
    "rating_before", "rating_after", "delta_raw", "delta_adj",
    "zero_mean_offset", "minutes_share", "perf_i", "perf_used",
    "game_score", "z_game_score", "on_court_rate", "z_on_court_rate",
    "s_team", "e_team", "k_effective",
]


def variant_id(track, playoff_k):
    return f"{track}_pk{playoff_k:g}"


def source_paths(playoff_k):
    """Return dict of source file paths for one playoff_k value."""
    tag = f"pk{playoff_k:g}"
    if playoff_k == 1.0:
        return {
            "shared": FINAL / "final_elo_updates_canonical.csv.gz",
            "canonical": FINAL / "final_elo_updates_canonical.csv.gz",
            "modern": FINAL / "final_elo_updates_modern.csv.gz",
        }
    return {
        "shared": VARIANTS / "shared" / f"final_elo_updates_1976-1995_{tag}.csv.gz",
        "canonical": VARIANTS / "canonical" / f"final_elo_updates_canonical_{tag}.csv.gz",
        "modern": VARIANTS / "modern" / f"final_elo_updates_modern_{tag}.csv.gz",
    }


def params_hash():
    blob = json.dumps(
        {
            track: {k: v for k, v in params.items()}
            for track, params in {"canonical": CANONICAL, "modern": MODERN}.items()
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def init_db(conn):
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;

        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE params (
            variant_id TEXT PRIMARY KEY,
            track TEXT NOT NULL,
            playoff_k REAL NOT NULL,
            k REAL, scale REAL, theta REAL, h REAL,
            alpha_modern REAL, alpha_ref REAL,
            rookie_boost REAL, rookie_tau REAL, rookie_start REAL,
            on_court_mode TEXT,
            params_hash TEXT
        );

        CREATE TABLE run_variants (
            variant_id TEXT PRIMARY KEY,
            track TEXT NOT NULL,
            playoff_k REAL NOT NULL,
            continuation_source TEXT NOT NULL
        );

        CREATE TABLE games (
            game_id INTEGER PRIMARY KEY,
            game_date TEXT,
            season INTEGER,
            season_id TEXT,
            season_type TEXT,
            track TEXT,
            hometeam_id INTEGER,
            awayteam_id INTEGER,
            hometeam_name TEXT,
            awayteam_name TEXT,
            home_score INTEGER,
            away_score INTEGER,
            winner INTEGER,
            is_usable INTEGER
        );

        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            full_name TEXT,
            birth_date TEXT,
            position TEXT,
            draft_year INTEGER,
            draft_round INTEGER,
            draft_number REAL,
            from_year INTEGER,
            to_year INTEGER
        );

        CREATE TABLE rating_events (
            playoff_k REAL NOT NULL,
            source TEXT NOT NULL,
            game_id INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            season INTEGER NOT NULL,
            track TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            team_id INTEGER,
            home INTEGER,
            rating_before REAL,
            rating_after REAL,
            delta_raw REAL,
            delta_adj REAL,
            zero_mean_offset REAL,
            minutes_share REAL,
            perf_i REAL,
            perf_used REAL,
            game_score REAL,
            z_game_score REAL,
            on_court_rate REAL,
            z_on_court_rate REAL,
            s_team REAL,
            e_team REAL,
            k_effective REAL,
            PRIMARY KEY (playoff_k, source, game_id, player_id)
        );
        CREATE INDEX idx_events_date ON rating_events (playoff_k, game_date);
        CREATE INDEX idx_events_player ON rating_events (playoff_k, player_id, game_date);
        CREATE INDEX idx_events_game ON rating_events (playoff_k, game_id);

        CREATE TABLE boxscores_minutes (
            game_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            scaled_minutes REAL NOT NULL,
            PRIMARY KEY (game_id, player_id)
        );
        """
    )


def load_games_players(conn):
    games = pd.read_csv(CLEAN / "games_clean.csv", low_memory=False)
    keep = [
        "game_id", "game_date", "season", "season_id", "season_type", "track",
        "hometeamId", "awayteamId", "hometeamName", "awayteamName",
        "homeScore", "awayScore", "winner", "is_usable",
    ]
    games = games[keep].rename(
        columns={
            "hometeamId": "hometeam_id",
            "awayteamId": "awayteam_id",
            "hometeamName": "hometeam_name",
            "awayteamName": "awayteam_name",
            "homeScore": "home_score",
            "awayScore": "away_score",
        }
    )
    games["game_date"] = games["game_date"].astype(str).str.slice(0, 10)
    games.to_sql("games", conn, if_exists="append", index=False)

    players = pd.read_csv(CLEAN / "players_clean.csv", low_memory=False)
    players["full_name"] = (
        players["firstName"].fillna("") + " " + players["lastName"].fillna("")
    ).str.strip()
    keep = [
        "personId", "firstName", "lastName", "full_name", "birthDate",
        "position", "draftYear", "draftRound", "draftNumber",
        "fromYear", "toYear",
    ]
    players = players[keep].rename(columns={
        "personId": "player_id",
        "firstName": "first_name",
        "lastName": "last_name",
        "birthDate": "birth_date",
        "draftYear": "draft_year",
        "draftRound": "draft_round",
        "draftNumber": "draft_number",
        "fromYear": "from_year",
        "toYear": "to_year",
    })
    players.to_sql("players", conn, if_exists="append", index=False)


def load_params(conn, values):
    rows = []
    for track in TRACKS:
        params = CANONICAL if track == "canonical" else MODERN
        for pk in values:
            rows.append(
                (
                    variant_id(track, pk), track, pk,
                    params["k"], params["scale"], params["theta"],
                    params["h"], params["alpha_modern"], params["alpha_ref"],
                    params["rookie_boost"], params["rookie_tau"],
                    params["rookie_start"], params["on_court_mode"],
                    params_hash(),
                )
            )
    conn.executemany(
        "INSERT INTO params VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.executemany(
        "INSERT INTO run_variants VALUES (?,?,?,?)",
        [
            (variant_id(track, pk), track, pk, track)
            for track in TRACKS
            for pk in values
        ],
    )


def load_minutes(conn):
    """Store scaled minutes per (game, player) from the clean boxscores."""
    reader = pd.read_csv(
        CLEAN / "boxscores_clean.csv",
        usecols=["game_id", "season", "personId", "scaled_minutes", "usable_game"],
        chunksize=500_000,
        low_memory=False,
    )
    for chunk in reader:
        chunk = chunk[chunk["usable_game"] == 1]
        if chunk.empty:
            continue
        chunk = chunk.rename(columns={"personId": "player_id"})[
            ["game_id", "player_id", "season", "scaled_minutes"]
        ]
        chunk.to_sql("boxscores_minutes", conn, if_exists="append", index=False)
    conn.execute(
        "CREATE INDEX idx_minutes_season ON boxscores_minutes (season, player_id)"
    )


def _to_sql_rows(chunk):
    """Convert a pandas chunk into plain Python rows safe for sqlite3."""
    chunk = chunk.replace({np.nan: None})
    rows = []
    for t in chunk.itertuples(index=False, name=None):
        row = []
        for v in t:
            if v is None:
                row.append(None)
            elif isinstance(v, float) and math.isnan(v):
                row.append(None)
            elif hasattr(v, "item"):
                row.append(v.item())
            else:
                row.append(v)
        rows.append(tuple(row))
    return rows


def stream_events(conn, path, playoff_k, source):
    """Stream one updates gz into rating_events with its variant metadata."""
    reader = pd.read_csv(
        path, chunksize=250_000, compression="gzip", low_memory=False
    )
    total = 0
    for chunk in reader:
        chunk = chunk[EVENT_COLUMNS]
        if source == "shared":
            chunk = chunk[chunk["season"] <= 1995]
        else:
            chunk = chunk[chunk["season"] >= 1996]
        if chunk.empty:
            continue
        chunk["game_date"] = chunk["game_date"].astype(str).str.slice(0, 10)
        chunk["playoff_k"] = playoff_k
        chunk["source"] = source
        rows = _to_sql_rows(chunk)
        conn.executemany(
            f"INSERT INTO rating_events ({', '.join(EVENT_COLUMNS)}, playoff_k, source) "
            f"VALUES ({', '.join('?' * (len(EVENT_COLUMNS) + 2))})",
            rows,
        )
        total += len(rows)
    conn.commit()
    return total


def load_events(conn, values):
    counts = {}
    for pk in values:
        paths = source_paths(pk)
        n = stream_events(conn, paths["shared"], pk, "shared")
        counts[f"shared_pk{pk:g}"] = n
        for track in TRACKS:
            n = stream_events(conn, paths[track], pk, track)
            counts[f"{track}_pk{pk:g}"] = n
    return counts


def build_season_ratings(conn):
    """Materialize player x season x variant rows.

    rating/games come from the last event of the season; minutes from the
    clean boxscores; rank is a dense rank over the season's final ratings.
    """
    conn.execute(
        """
        CREATE TABLE season_ratings AS
        WITH ev AS (
            SELECT rv.variant_id, rv.track, rv.playoff_k, re.source,
                   re.game_id, re.game_date, re.season,
                   re.player_id, re.rating_after
            FROM rating_events re
            JOIN run_variants rv
              ON rv.playoff_k = re.playoff_k
             AND (re.source = 'shared' OR re.source = rv.continuation_source)
        ),
        final AS (
            SELECT e.variant_id, e.track, e.playoff_k, e.source,
                   e.season, e.player_id, e.rating_after,
                   ROW_NUMBER() OVER (
                       PARTITION BY e.variant_id, e.season, e.player_id
                       ORDER BY e.game_date DESC, e.game_id DESC
                   ) AS rn
            FROM ev e
        ),
        ratings AS (
            SELECT variant_id, track, playoff_k, source,
                   season, player_id, rating_after AS rating
            FROM final WHERE rn = 1
        ),
        mins AS (
            SELECT season, player_id, SUM(scaled_minutes) AS minutes,
                   COUNT(*) AS games
            FROM boxscores_minutes GROUP BY season, player_id
        )
        SELECT r.variant_id, r.track, r.playoff_k, r.source,
               r.season AS season_start,
               r.season + 1 AS season,
               r.player_id, r.rating,
               COALESCE(m.games, 0) AS games,
               COALESCE(m.minutes, 0.0) AS minutes,
               RANK() OVER (
                   PARTITION BY r.variant_id, r.season
                   ORDER BY r.rating DESC
               ) AS rank
        FROM ratings r
        LEFT JOIN mins m
          ON m.season = r.season AND m.player_id = r.player_id
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX idx_season_ratings_pk "
        "ON season_ratings (variant_id, season, player_id)"
    )
    conn.execute(
        "CREATE INDEX idx_season_ratings_rank "
        "ON season_ratings (variant_id, season, rank)"
    )


def build_snapshots(conn):
    """Materialize allstar (Feb 1) and season_end snapshots per variant."""
    conn.execute(
        """
        CREATE TABLE snapshots (
            variant_id TEXT NOT NULL,
            track TEXT NOT NULL,
            playoff_k REAL NOT NULL,
            source TEXT NOT NULL,
            season_start INTEGER NOT NULL,
            season INTEGER NOT NULL,
            snapshot_type TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            rank INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO snapshots (
            variant_id, track, playoff_k, source, season_start, season,
            snapshot_type, snapshot_date, player_id, rating, rank
        )
        WITH ev AS (
            SELECT rv.variant_id, rv.track, rv.playoff_k, re.source,
                   re.game_id, re.game_date, re.season,
                   re.player_id, re.rating_after
            FROM rating_events re
            JOIN run_variants rv
              ON rv.playoff_k = re.playoff_k
             AND (re.source = 'shared' OR re.source = rv.continuation_source)
        ),
        ends AS (
            SELECT variant_id, season, MAX(game_date) AS end_date
            FROM ev GROUP BY variant_id, season
        ),
        final AS (
            SELECT e.variant_id, e.track, e.playoff_k, e.source,
                   e.season, e.player_id, e.rating_after, e.game_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY e.variant_id, e.season, e.player_id
                       ORDER BY e.game_date DESC, e.game_id DESC
                   ) AS rn
            FROM ev e
        )
        SELECT f.variant_id, f.track, f.playoff_k, f.source,
               f.season, f.season + 1, 'season_end',
               d.end_date, f.player_id, f.rating_after,
               RANK() OVER (
                   PARTITION BY f.variant_id, f.season
                   ORDER BY f.rating_after DESC
               )
        FROM final f
        JOIN ends d
          ON d.variant_id = f.variant_id AND d.season = f.season
        WHERE f.rn = 1
        """
    )
    conn.execute(
        """
        INSERT INTO snapshots (
            variant_id, track, playoff_k, source, season_start, season,
            snapshot_type, snapshot_date, player_id, rating, rank
        )
        WITH ev AS (
            SELECT rv.variant_id, rv.track, rv.playoff_k, re.source,
                   re.game_id, re.game_date, re.season,
                   re.player_id, re.rating_after
            FROM rating_events re
            JOIN run_variants rv
              ON rv.playoff_k = re.playoff_k
             AND (re.source = 'shared' OR re.source = rv.continuation_source)
        ),
        cutoffs AS (
            SELECT DISTINCT variant_id, season,
                   printf('%04d-02-01', season + 1) AS cutoff
            FROM ev
        ),
        final AS (
            SELECT e.variant_id, e.track, e.playoff_k, e.source,
                   e.season, e.player_id, e.rating_after, e.game_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY e.variant_id, e.season, e.player_id
                       ORDER BY e.game_date DESC, e.game_id DESC
                   ) AS rn
            FROM ev e
            JOIN cutoffs c
              ON c.variant_id = e.variant_id AND c.season = e.season
            WHERE e.game_date <= c.cutoff
        )
        SELECT f.variant_id, f.track, f.playoff_k, f.source,
               f.season, f.season + 1, 'allstar',
               c.cutoff, f.player_id, f.rating_after,
               RANK() OVER (
                   PARTITION BY f.variant_id, f.season
                   ORDER BY f.rating_after DESC
               )
        FROM final f
        JOIN cutoffs c
          ON c.variant_id = f.variant_id AND c.season = f.season
        WHERE f.rn = 1
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX idx_snapshots_pk ON snapshots "
        "(variant_id, season, snapshot_type, player_id)"
    )
    conn.execute(
        "CREATE INDEX idx_snapshots_rank ON snapshots "
        "(variant_id, season, snapshot_type, rank)"
    )


def create_views(conn):
    conn.executescript(
        """
        CREATE VIEW variant_timeline AS
        SELECT rv.variant_id,
               rv.track AS track,
               rv.playoff_k,
               re.source,
               re.game_id,
               re.game_date,
               re.season AS season_start,
               re.season + 1 AS season,
               re.track AS game_track,
               re.player_id,
               re.team_id,
               re.home,
               re.rating_before,
               re.rating_after,
               re.delta_raw,
               re.delta_adj,
               re.zero_mean_offset,
               re.minutes_share,
               re.perf_i,
               re.perf_used,
               re.game_score,
               re.z_game_score,
               re.on_court_rate,
               re.z_on_court_rate,
               re.s_team,
               re.e_team,
               re.k_effective
        FROM rating_events re
        JOIN run_variants rv
          ON rv.playoff_k = re.playoff_k
         AND (re.source = 'shared' OR re.source = rv.continuation_source);

        CREATE VIEW dual_track_season AS
        SELECT c.season,
               c.season_start,
               c.playoff_k,
               c.player_id,
               c.rating AS rating_canonical,
               m.rating AS rating_modern,
               c.rating - m.rating AS rating_delta,
               c.rank AS rank_canonical,
               m.rank AS rank_modern,
               c.rank - m.rank AS rank_diff,
               c.games AS games_canonical,
               m.games AS games_modern,
               c.minutes AS minutes_canonical,
               m.minutes AS minutes_modern
        FROM season_ratings c
        JOIN season_ratings m
          ON m.playoff_k = c.playoff_k
         AND m.season = c.season
         AND m.player_id = c.player_id
         AND m.track = 'modern'
        WHERE c.track = 'canonical';
        """
    )


def write_meta(conn, counts, values):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    checkpoints = {
        "initial_1976": file_sha256(
            FINAL / "final_elo_initial_state_1976_canonical.csv"
        )
    }
    for pk in values:
        if pk == 1.0:
            path = FINAL / "final_elo_checkpoint_1995_canonical.csv"
        else:
            path = VARIANTS / "checkpoints" / f"checkpoint_1995_pk{pk:g}.csv"
        checkpoints[f"checkpoint_1995_pk{pk:g}"] = file_sha256(path)

    source_files = {}
    for pk in values:
        for role, path in source_paths(pk).items():
            source_files[f"pk{pk:g}_{role}"] = str(path)
    source_files["clean_games"] = str(CLEAN / "games_clean.csv")
    source_files["clean_players"] = str(CLEAN / "players_clean.csv")
    source_files["clean_boxscores"] = str(CLEAN / "boxscores_clean.csv")

    meta = {
        "schema_version": "2",
        "generated_at": now,
        "params_hash": params_hash(),
        "initial_state_hash": checkpoints.pop("initial_1976"),
        "checkpoint_hashes_json": json.dumps(checkpoints, sort_keys=True),
        "event_counts_json": json.dumps(counts, sort_keys=True),
        "source_files_json": json.dumps(source_files, sort_keys=True),
    }
    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        sorted(meta.items()),
    )


def _pre96_dual_check(conn, pk):
    n_c = conn.execute(
        "SELECT COUNT(*) FROM season_ratings "
        "WHERE track='canonical' AND playoff_k=? AND season <= 1996",
        (pk,),
    ).fetchone()[0]
    n_m = conn.execute(
        "SELECT COUNT(*) FROM season_ratings "
        "WHERE track='modern' AND playoff_k=? AND season <= 1996",
        (pk,),
    ).fetchone()[0]
    missing = conn.execute(
        """
        SELECT COUNT(*) FROM season_ratings c
        WHERE c.track='canonical' AND c.playoff_k=? AND c.season <= 1996
          AND NOT EXISTS (
              SELECT 1 FROM season_ratings m
              WHERE m.playoff_k=c.playoff_k AND m.season=c.season
                AND m.player_id=c.player_id AND m.track='modern'
          )
        """,
        (pk,),
    ).fetchone()[0]
    mismatch = conn.execute(
        """
        SELECT COUNT(*) FROM season_ratings c
        JOIN season_ratings m
          ON m.playoff_k=c.playoff_k AND m.season=c.season
         AND m.player_id=c.player_id AND m.track='modern'
        WHERE c.track='canonical' AND c.playoff_k=? AND c.season <= 1996
          AND (ABS(c.rating - m.rating) > 1e-9 OR c.rank != m.rank)
        """,
        (pk,),
    ).fetchone()[0]
    return n_c, n_m, missing, mismatch


def validate(conn, values, smoke):
    checks = []

    total_events = 0
    for pk in values:
        for source in ("shared", "canonical", "modern"):
            exp = EXPECTED_EVENTS[source]
            n = conn.execute(
                "SELECT COUNT(*) FROM rating_events "
                "WHERE playoff_k=? AND source=?",
                (pk, source),
            ).fetchone()[0]
            total_events += n
            ok = n == exp
            checks.append(
                (
                    f"events pk{pk:g} {source}",
                    ok,
                    f"{n:,} (expected {exp:,})",
                )
            )
    checks.append(
        ("events total", total_events == sum(EXPECTED_EVENTS.values()) * len(values),
         f"{total_events:,}")
    )

    bad_zero_sum = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT playoff_k, source, game_id, SUM(delta_adj) AS s
            FROM rating_events
            GROUP BY playoff_k, source, game_id
            HAVING ABS(s) > 1e-6
        )
        """
    ).fetchone()[0]
    checks.append(("zero-sum per game", bad_zero_sum == 0, f"{bad_zero_sum} bad games"))

    null_cols = ", ".join(CORE_NUMERIC_COLUMNS)
    bad_null = conn.execute(
        f"""
        SELECT COUNT(*) FROM rating_events
        WHERE { " OR ".join(c + " IS NULL" for c in CORE_NUMERIC_COLUMNS) }
           OR rating_after NOT BETWEEN -100000 AND 100000
        """
    ).fetchone()[0]
    checks.append(("event null/range", bad_null == 0, f"{bad_null} bad rows"))

    n_rv = conn.execute("SELECT COUNT(*) FROM run_variants").fetchone()[0]
    checks.append(("run_variants", n_rv == 2 * len(values), f"{n_rv} rows"))

    season_counts = conn.execute(
        """
        SELECT variant_id, COUNT(DISTINCT season), MIN(season), MAX(season)
        FROM season_ratings GROUP BY variant_id ORDER BY variant_id
        """
    ).fetchall()
    ok_seasons = all(n == 50 and lo == 1977 and hi == 2026 for _, n, lo, hi in season_counts)
    checks.append(
        (
            "season_ratings coverage",
            ok_seasons and len(season_counts) == 2 * len(values),
            "; ".join(f"{v}:{n} ({lo}-{hi})" for v, n, lo, hi in season_counts),
        )
    )

    for pk in values:
        n_c, n_m, missing, mismatch = _pre96_dual_check(conn, pk)
        ok = n_c == n_m and missing == 0 and mismatch == 0
        checks.append(
            (
                f"pre-96 dual pk{pk:g}",
                ok,
                f"canonical={n_c:,} modern={n_m:,} missing={missing:,} mismatch={mismatch:,}",
            )
        )

    if smoke:
        n_minutes = 0
    else:
        n_minutes = conn.execute(
            "SELECT COUNT(*) FROM boxscores_minutes"
        ).fetchone()[0]
    expected_minutes = 0 if smoke else EXPECTED_MINUTES
    checks.append(
        (
            "boxscores_minutes",
            n_minutes == expected_minutes,
            f"{n_minutes:,} (expected {expected_minutes:,})",
        )
    )

    timeline_counts = conn.execute(
        """
        SELECT rv.variant_id, COUNT(*)
        FROM rating_events re
        JOIN run_variants rv
          ON rv.playoff_k = re.playoff_k
         AND (re.source = 'shared' OR re.source = rv.continuation_source)
        GROUP BY rv.variant_id ORDER BY rv.variant_id
        """
    ).fetchall()
    ok_timeline = all(
        n == EXPECTED_TIMELINE_EVENTS for _, n in timeline_counts
    ) and len(timeline_counts) == 2 * len(values)
    checks.append(
        (
            "variant_timeline coverage",
            ok_timeline,
            "; ".join(f"{v}:{n:,}" for v, n in timeline_counts),
        )
    )

    if VARIANTS.joinpath("run_summary.csv").exists():
        summary = pd.read_csv(VARIANTS / "run_summary.csv")
        summary_key = {
            (r.track, float(r.playoff_k)): r
            for r in summary.itertuples(index=False)
        }
        for track in TRACKS:
            for pk in values:
                vid = variant_id(track, pk)
                row = summary_key[(track, pk)]
                q = conn.execute(
                    """
                    SELECT AVG(rating), AVG(rating * rating),
                           SUM(rating * minutes), SUM(minutes),
                           MAX(rating), COUNT(*)
                    FROM season_ratings
                    WHERE variant_id=? AND season=2026
                    """,
                    (vid,),
                ).fetchone()
                avg, avg_sq, sum_rw, sum_w, top, n = q
                if n is None or n == 0:
                    checks.append((f"end metrics {vid}", False, "no rows"))
                    continue
                avg = float(avg)
                sd = math.sqrt(max(0.0, float(avg_sq) - avg * avg))
                top = float(top)
                ok_sd = abs(sd - float(row.last_sd)) <= 0.05
                ok_top = abs(top - float(row.last_top_rating)) <= 0.05
                detail = (
                    f"sd={sd:.2f} vs {row.last_sd:.2f}, "
                    f"top={top:.2f} vs {row.last_top_rating:.2f}"
                )
                ok = ok_sd and ok_top
                if not smoke and sum_w and sum_w > 0:
                    wmean = float(sum_rw) / float(sum_w)
                    ok_w = abs(wmean - float(row.last_minutes_wmean)) <= 0.05
                    ok = ok and ok_w
                    detail += (
                        f", wmean={wmean:.2f} vs {row.last_minutes_wmean:.2f}"
                    )
                checks.append((f"end metrics {vid}", ok, detail))

    n_snap = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    n_dual = conn.execute("SELECT COUNT(*) FROM dual_track_season").fetchone()[0]
    checks.append(("snapshots rows", n_snap > 0, f"{n_snap:,}"))
    checks.append(("dual_track_season rows", n_dual > 0, f"{n_dual:,}"))

    return checks


def _remove_db(path):
    if path.exists():
        path.unlink()
    for suffix in ("-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the db exists")
    parser.add_argument("--check-only", action="store_true",
                        help="validate existing db and exit")
    parser.add_argument("--smoke", action="store_true",
                        help="build only pk1.0 into scratch/nba_elo_smoke.db")
    args = parser.parse_args()

    values = [1.0] if args.smoke else PLAYOFF_VALUES
    db_path = SMOKE_DB_PATH if args.smoke else DB_PATH

    if args.check_only:
        if not db_path.exists():
            raise SystemExit(f"{db_path} does not exist")
        conn = sqlite3.connect(db_path)
        try:
            checks = validate(conn, values, smoke=args.smoke)
            failed = [c for c in checks if not c[1]]
            for label, ok, detail in checks:
                print(f"[{'ok' if ok else 'FAIL'}] {label}: {detail}", flush=True)
            if failed:
                raise SystemExit(f"{len(failed)} check(s) failed")
            print("all checks passed", flush=True)
        finally:
            conn.close()
        return

    if db_path.exists() and not args.force:
        raise SystemExit(f"{db_path} already exists; use --force to rebuild")

    t0 = time.time()
    _remove_db(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)
        print("loading games/players ...", flush=True)
        load_games_players(conn)
        print("loading params/variants ...", flush=True)
        load_params(conn, values)
        if not args.smoke:
            print("loading clean minutes ...", flush=True)
            load_minutes(conn)
        print("loading rating events ...", flush=True)
        counts = load_events(conn, values)
        conn.commit()
        print("building season_ratings ...", flush=True)
        build_season_ratings(conn)
        print("building snapshots ...", flush=True)
        build_snapshots(conn)
        print("building views ...", flush=True)
        create_views(conn)
        write_meta(conn, counts, values)
        conn.commit()
        checks = validate(conn, values, smoke=args.smoke)
        failed = [c for c in checks if not c[1]]
        for label, ok, detail in checks:
            print(f"[{'ok' if ok else 'FAIL'}] {label}: {detail}", flush=True)
        if failed:
            raise SystemExit(f"{len(failed)} check(s) failed")
        ev = conn.execute("SELECT COUNT(*) FROM rating_events").fetchone()[0]
        sr = conn.execute("SELECT COUNT(*) FROM season_ratings").fetchone()[0]
        sp = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        print(
            f"done in {time.time() - t0:.1f}s: "
            f"events={ev:,} season_ratings={sr:,} snapshots={sp:,}",
            flush=True,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
