from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "5_discussion"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"


def season_label(season):
    return f"{int(season)}-{str(int(season) + 1)[-2:]}"


def five_year_centered(values):
    return pd.Series(values).rolling(5, center=True, min_periods=3).mean().to_numpy()


def build_tables(data):
    regular = data[data["track"] == "regular"].copy()
    regular["game_date"] = pd.to_datetime(regular["game_date"])

    game = regular.groupby(["season", "game_id"], as_index=False).agg(
        participants=("player_id", "size"),
        max_z=("z_game_score", "max"),
        min_z=("z_game_score", "min"),
    )
    game["participants_per_team"] = game["participants"] / 2

    team_season = regular[["season", "team_id", "player_id"]].drop_duplicates()
    team_counts = team_season.groupby("season", as_index=False).agg(
        player_team_stints=("player_id", "size"),
        teams=("team_id", "nunique"),
    )
    team_counts["distinct_players_used_per_team"] = (
        team_counts["player_team_stints"] / team_counts["teams"]
    )

    unique_counts = regular.groupby("season", as_index=False).agg(
        league_players=("player_id", "nunique")
    )
    unique_counts = unique_counts.merge(team_counts[["season", "teams"]], on="season")
    unique_counts["unique_players_per_team"] = (
        unique_counts["league_players"] / unique_counts["teams"]
    )

    season = game.groupby("season", as_index=False).agg(
        games=("game_id", "size"),
        players_per_game=("participants", "mean"),
        players_per_team_game=("participants_per_team", "mean"),
        mean_game_max_z=("max_z", "mean"),
        mean_game_min_z=("min_z", "mean"),
    )
    season = season.merge(team_counts, on="season").merge(
        unique_counts[["season", "league_players", "unique_players_per_team"]], on="season"
    )

    # Player-season movement. The average rating during the season defines the
    # tier; this avoids calling a rookie "low" solely because he starts at 1425.
    ps = regular.sort_values(["season", "player_id", "game_date", "game_id"]).groupby(
        ["season", "player_id"], as_index=False
    ).agg(
        games=("game_id", "size"),
        avg_rating=("rating_after", "mean"),
        start_rating=("rating_before", "first"),
        end_rating=("rating_after", "last"),
        negative_elo=("delta_adj", lambda x: x[x < 0].sum()),
        positive_elo=("delta_adj", lambda x: x[x > 0].sum()),
        avg_minutes_share=("minutes_share", "mean"),
    )
    ps["net_change"] = ps["end_rating"] - ps["start_rating"]
    ps["negative_elo_per_game"] = ps["negative_elo"] / ps["games"]
    ps["net_change_per_game"] = ps["net_change"] / ps["games"]
    ps["rating_pct"] = ps.groupby("season")["avg_rating"].rank(pct=True)
    ps["tier"] = pd.cut(
        ps["rating_pct"], [0, 0.4, 0.8, 1.0], labels=["Bottom 40%", "Middle 40%", "Top 20%"],
        include_lowest=True,
    )
    movement = ps.groupby(["season", "tier"], observed=True, as_index=False).agg(
        players=("player_id", "size"),
        median_games=("games", "median"),
        mean_games=("games", "mean"),
        mean_negative_elo=("negative_elo", "mean"),
        mean_negative_elo_per_game=("negative_elo_per_game", "mean"),
        mean_net_change=("net_change", "mean"),
        mean_net_change_per_game=("net_change_per_game", "mean"),
        mean_minutes_share=("avg_minutes_share", "mean"),
    )

    game_bins = pd.cut(
        game["participants"], [-np.inf, 17, 19, 21, 23, np.inf],
        labels=["17 or fewer", "18-19", "20-21", "22-23", "24 or more"],
    )
    game_effect = game.assign(participant_bin=game_bins).groupby(
        "participant_bin", observed=True, as_index=False
    ).agg(
        games=("game_id", "size"),
        mean_max_z=("max_z", "mean"),
        mean_min_z=("min_z", "mean"),
    )

    season.to_csv(OUT / "player_pool_by_season.csv", index=False)
    movement.to_csv(OUT / "elo_movement_by_season_and_tier.csv", index=False)
    game_effect.to_csv(OUT / "game_participant_count_effect.csv", index=False)
    return season, movement, game_effect


def draw_chart(season, movement, game_effect):
    W, H = 1800, 1180
    S = 2
    im = Image.new("RGB", (W * S, H * S), "#FAFAF8")
    d = ImageDraw.Draw(im, "RGBA")
    fdir = Path("C:/Windows/Fonts")
    def font(name, size):
        return ImageFont.truetype(str(fdir / name), size * S)
    ft = font("arialbd.ttf", 38)
    fs = font("arial.ttf", 17)
    fp = font("arialbd.ttf", 22)
    fa = font("arial.ttf", 14)

    d.text((90*S, 35*S), "How a larger player pool can favor the top of the Elo distribution", fill="#111827", font=ft)
    d.text((90*S, 88*S), "More players per game expand the performance tail; more players per team spread lower-tier minutes and updates across a deeper pool.", fill="#4B5563", font=fs)

    panels = [(95, 180, 855, 535), (955, 180, 1715, 535), (95, 685, 855, 1060), (955, 685, 1715, 1060)]

    def axes(box, title, xlab, ylab, xmin, xmax, ymin, ymax, xticks, yticks):
        l,t,r,b = [v*S for v in box]
        d.text((l, t-42*S), title, fill="#111827", font=fp)
        for yv in yticks:
            yy=b-(yv-ymin)/(ymax-ymin)*(b-t)
            d.line((l,yy,r,yy), fill="#E5E7EB", width=S)
            d.text((l-58*S,yy-8*S), f"{yv:g}", fill="#6B7280", font=fa)
        for xv,label in xticks:
            xx=l+(xv-xmin)/(xmax-xmin)*(r-l)
            d.text((xx-24*S,b+10*S), str(label), fill="#6B7280", font=fa)
        d.rectangle((l,t,r,b), outline="#9CA3AF", width=2*S)
        d.text(((l+r)//2-70*S,b+48*S), xlab, fill="#374151", font=fs)
        d.text((l, b+80*S), ylab, fill="#374151", font=fa)
        return lambda x,y:(l+(x-xmin)/(xmax-xmin)*(r-l), b-(y-ymin)/(ymax-ymin)*(b-t))

    years=season["season"].to_numpy()
    # Panel 1: per-game depth
    vals=five_year_centered(season["players_per_team_game"])
    xy=axes(panels[0], "A. Players appearing per team-game", "Season", "5-season centered average", years.min(), years.max(), 9.7, 10.8,
            [(1980,"1980"),(1990,"1990"),(2000,"2000"),(2010,"2010"),(2020,"2020")], [9.8,10.0,10.2,10.4,10.6,10.8])
    pts=[xy(x,y) for x,y in zip(years,vals) if np.isfinite(y)]
    d.line(pts, fill="#2563EB", width=4*S, joint="curve")

    # Panel 2: league-wide unique players per team. A traded player is counted
    # only once in the league total, avoiding a mechanical trade-frequency effect.
    vals2=five_year_centered(season["unique_players_per_team"])
    ymin=np.floor(np.nanmin(vals2)); ymax=np.ceil(np.nanmax(vals2))
    xy=axes(panels[1], "B. League-wide unique players per NBA team", "Season", "Each player is counted once, even if traded", years.min(), years.max(), ymin, ymax,
            [(1980,"1980"),(1990,"1990"),(2000,"2000"),(2010,"2010"),(2020,"2020")], list(np.arange(ymin,ymax+0.1,2)))
    d.line([xy(x,y) for x,y in zip(years,vals2) if np.isfinite(y)], fill="#7C3AED", width=4*S, joint="curve")

    # Panel 3: number of competitors and the maximum z in a game
    gx=np.arange(len(game_effect))
    gvals=game_effect["mean_max_z"].to_numpy()
    xy=axes(panels[2], "C. More participants raise the top Game Score z", "Players in game", "Average highest Game Score z", -0.5, len(gx)-0.5, 2.0, 2.65,
            [(i,lab) for i,lab in enumerate(game_effect["participant_bin"].astype(str))], [2.0,2.2,2.4,2.6])
    for i,v in enumerate(gvals):
        x0,y0=xy(i-0.28,2.0); x1,y1=xy(i+0.28,v)
        d.rectangle((x0,y1,x1,y0), fill="#2563EB")
        d.text((x0+8*S,y1-25*S),f"{v:.2f}",fill="#111827",font=fa)

    # Panel 4: a direct era comparison is easier to interpret than a time series
    # of cumulative negative updates. Fewer appearances mean fewer total updates
    # are accumulated by each lower- or middle-tier player.
    era_movement = movement[movement["tier"].isin(["Bottom 40%", "Middle 40%"])].copy()
    era_movement["era"] = np.where(
        era_movement["season"].between(1980, 1989), "1980s",
        np.where(era_movement["season"].between(2020, 2025), "2020s", "Other"),
    )
    era_movement = era_movement[era_movement["era"] != "Other"].groupby(
        ["tier", "era"], observed=True, as_index=False
    )["mean_games"].mean()
    xy=axes(panels[3], "D. A larger pool gives each player fewer updates", "Player tier", "Mean games per player-season", -0.5, 1.5, 0, 75,
            [(0,"Bottom 40%"),(1,"Middle 40%")], [0,15,30,45,60,75])
    era_colors={"1980s":"#6B7280","2020s":"#7C3AED"}
    for i,tier in enumerate(["Bottom 40%","Middle 40%"]):
        for era,offset in [("1980s",-0.18),("2020s",0.18)]:
            value=float(era_movement.loc[(era_movement["tier"]==tier)&(era_movement["era"]==era),"mean_games"].iloc[0])
            x0,y0=xy(i+offset-0.15,0); x1,y1=xy(i+offset+0.15,value)
            d.rectangle((x0,y1,x1,y0),fill=era_colors[era])
            d.text((x0+10*S,y1-25*S),f"{value:.0f}",fill="#111827",font=fa)
    l,t,r,b=[v*S for v in panels[3]]
    d.text((l+14*S,t+12*S),"1980s",fill=era_colors["1980s"],font=fa)
    d.text((l+82*S,t+12*S),"2020s",fill=era_colors["2020s"],font=fa)

    im=im.resize((W,H),Image.Resampling.LANCZOS)
    im.save(OUT / "player_pool_structural_effect.png", quality=95)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data=pd.read_csv(UPDATES)
    season,movement,game_effect=build_tables(data)
    draw_chart(season,movement,game_effect)
    print(season[["season","players_per_team_game","distinct_players_used_per_team","league_players","teams"]].tail(15).to_string(index=False))
    print("\nGame bins\n",game_effect.to_string(index=False))
    print("\nEra summaries\n")
    season["era"]=(season["season"]//10*10).astype(str)+"s"
    print(season.groupby("era")[["players_per_team_game","distinct_players_used_per_team","unique_players_per_team","league_players"]].mean().to_string())
    movement["era"]=(movement["season"]//10*10).astype(str)+"s"
    print("\nTier movement\n",movement.groupby(["era","tier"],observed=True)[["mean_games","mean_negative_elo","mean_negative_elo_per_game","mean_net_change","mean_minutes_share"]].mean().to_string())


if __name__ == "__main__":
    main()
