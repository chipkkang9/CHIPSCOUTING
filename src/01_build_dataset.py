"""1단계: 원본 CSV -> (선수, 기준일) 학습 테이블.

매년 7월 1일을 기준일로 잡고, 그날 21세 이하이면서 최근 1년 안에 몸값이
찍힌 선수를 한 행으로 만든다. 입력(feature)은 기준일 이전 정보만 쓰고,
정답(label)은 기준일 3년 뒤의 몸값이다.

실행:  python src/01_build_dataset.py
출력:  data/processed/dataset.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

FIRST_YEAR = 2013   # 출전 기록이 2012년 하반기부터 있어서 2013년 7월이 첫 기준일
LAST_YEAR = 2026    # 2024년 이후는 정답이 없는 "현재 유망주" 행
MAX_AGE = 21
HORIZON_YEARS = 3
YEAR = pd.Timedelta(days=365)


def last_value_before(vals, t):
    """각 선수의 t 시점 이전 마지막 몸값 기록."""
    return vals[vals["date"] <= t].groupby("player_id").tail(1).set_index("player_id")


def appearance_stats(apps, start, end, suffix):
    w = apps[(apps["date"] > start) & (apps["date"] <= end)]
    g = w.groupby("player_id")
    out = pd.DataFrame({
        f"games_{suffix}": g.size(),
        f"minutes_{suffix}": g["minutes_played"].sum(),
        f"goals_{suffix}": g["goals"].sum(),
        f"assists_{suffix}": g["assists"].sum(),
        f"yellows_{suffix}": g["yellow_cards"].sum(),
        f"league_minutes_{suffix}": w[w["comp_type"] == "domestic_league"].groupby("player_id")["minutes_played"].sum(),
        f"intl_minutes_{suffix}": w[w["comp_type"] == "international_cup"].groupby("player_id")["minutes_played"].sum(),
    })
    return out


def build_snapshot(year, players, vals, apps, transfers, data_end):
    t = pd.Timestamp(year, 7, 1)

    # --- 대상 선수: 최근 1년 안에 몸값이 있고, 기준일에 MAX_AGE세 이하 ---
    now = last_value_before(vals, t)
    now = now[now["date"] > t - YEAR]
    df = pd.DataFrame({
        "value_now": now["market_value_in_eur"],
        "club_id": now["current_club_id"],
        "league": now["player_club_domestic_competition_id"],
    })
    df = df.join(players, how="inner")
    df["age"] = (t - df["dob"]).dt.days / 365.25
    df = df[(df["age"] >= 15) & (df["age"] <= MAX_AGE)].copy()

    # --- 몸값 이력 feature ---
    past = vals[vals["date"] <= t]
    g = past.groupby("player_id")["market_value_in_eur"]
    df["value_1y_ago"] = last_value_before(vals, t - YEAR)["market_value_in_eur"]
    df["value_2y_ago"] = last_value_before(vals, t - 2 * YEAR)["market_value_in_eur"]
    df["value_max_so_far"] = g.max()
    df["n_valuations"] = g.size()
    df["years_since_first_valuation"] = (t - past.groupby("player_id")["date"].min()).dt.days / 365.25

    # --- 출전 기록 feature (1부 리그·컵·유럽 대회만 잡힘. 없으면 0) ---
    for start, suffix in [(t - YEAR, "1y"), (t - 2 * YEAR, "2y")]:
        df = df.join(appearance_stats(apps, start, t, suffix))
    app_cols = [c for c in df.columns if c.endswith(("_1y", "_2y"))]
    df[app_cols] = df[app_cols].fillna(0)
    df["has_appearances"] = (df["games_2y"] > 0).astype(int)

    # --- 이적 feature (미래 날짜로 예정된 이적은 제외) ---
    tr = transfers[transfers["transfer_date"] <= t]
    g = tr.groupby("player_id")
    df["n_transfers"] = g.size()
    df["max_transfer_fee"] = g["transfer_fee"].max()
    df["years_since_last_transfer"] = (t - g["transfer_date"].max()).dt.days / 365.25
    df["n_transfers"] = df["n_transfers"].fillna(0)

    # --- 정답: 3년 뒤(직전 1년 창 안) 마지막 몸값 ---
    t_future = t + pd.DateOffset(years=HORIZON_YEARS)
    df["value_future"] = np.nan
    if t_future - YEAR < data_end:
        fut = last_value_before(vals, t_future)
        fut = fut[fut["date"] > t_future - YEAR]
        df["value_future"] = fut["market_value_in_eur"]

    df.insert(0, "snapshot_year", year)
    return df.reset_index(names="player_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "dataset.parquet"))
    args = ap.parse_args()

    print("원본 CSV 읽는 중...")
    players = pd.read_csv(RAW / "players.csv", usecols=[
        "player_id", "name", "date_of_birth", "position", "sub_position",
        "foot", "height_in_cm", "country_of_citizenship"])
    players["dob"] = pd.to_datetime(players.pop("date_of_birth"), errors="coerce")
    players = players.dropna(subset=["dob"]).set_index("player_id")

    vals = pd.read_csv(RAW / "player_valuations.csv", parse_dates=["date"])
    vals = vals[vals["market_value_in_eur"] > 0].sort_values(["player_id", "date"])
    data_end = vals["date"].max()

    comps = pd.read_csv(RAW / "competitions.csv", usecols=["competition_id", "type"])
    apps = pd.read_csv(RAW / "appearances.csv", parse_dates=["date"], usecols=[
        "player_id", "date", "competition_id", "yellow_cards", "goals", "assists", "minutes_played"])
    apps["comp_type"] = apps["competition_id"].map(comps.set_index("competition_id")["type"])

    transfers = pd.read_csv(RAW / "transfers.csv", parse_dates=["transfer_date"],
                            usecols=["player_id", "transfer_date", "transfer_fee"])

    frames = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        frames.append(build_snapshot(year, players, vals, apps, transfers, data_end))
        print(f"  {year}년 7월 1일: {len(frames[-1]):,}명")
    ds = pd.concat(frames, ignore_index=True)

    ds["has_label"] = ds["value_future"].notna().astype(int)
    # 몸값은 수만~수억 유로로 치우쳐 있어서 로그로 바꿔 학습한다
    ds["log_value_now"] = np.log10(ds["value_now"])
    ds["log_value_future"] = np.log10(ds["value_future"])
    ds["log_growth"] = ds["log_value_future"] - ds["log_value_now"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.drop(columns=["dob"]).to_parquet(out, index=False)

    print(f"\n저장: {out}  ({len(ds):,}행, {ds.shape[1]}열)")
    summary = ds.groupby("snapshot_year").agg(
        선수수=("player_id", "size"),
        정답있음=("has_label", "mean"),
        출전기록있음=("has_appearances", "mean"),
        현재몸값_중앙값=("value_now", "median"),
        미래몸값_중앙값=("value_future", "median"),
    ).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()
