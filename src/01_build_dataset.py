"""1단계: 원본 CSV -> (선수, 기준일) 학습 테이블.

매년 7월 1일을 기준일로 잡고, 그날 21세 이하이면서 최근 1년 안에 몸값이
찍힌 선수를 한 행으로 만든다. 입력(feature)은 기준일 이전 정보만 쓰고,
정답(label)은 기준일 3년 뒤의 몸값이다.

원본: data/raw_salimt/ (Kaggle "5.7M+ Records - Most Comprehensive Football Dataset")

실행:  python src/01_build_dataset.py
출력:  data/processed/dataset.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_salimt"

FIRST_YEAR = 2012   # 2010~2011년은 3년 뒤 몸값이 없는 비율이 높아서 뺀다
LAST_YEAR = 2025    # 2023년 이후는 정답이 없는 "현재 유망주" 행 (몸값이 2025년 9월까지 있다)
MAX_AGE = 21
HORIZON_YEARS = 3
YEAR = pd.Timedelta(days=365)
VALUE_FLOOR = 10_000    # 몸값 기록이 없는 선수를 대회 수준 계산에 넣을 때 쓰는 최저값
NO_CLUB = ["Retired", "Without Club", "Unknown", "Career break"]   # 클럽이 아닌 상태 표시


# ---------------------------------------------------------------------------
# 원본 읽기
# ---------------------------------------------------------------------------
def load_raw():
    players = pd.read_csv(RAW / "player_profiles.csv", low_memory=False, usecols=[
        "player_id", "player_name", "date_of_birth", "height", "citizenship",
        "position", "main_position", "foot"])
    players["name"] = players.pop("player_name").str.replace(r"\s*\(\d+\)$", "", regex=True)
    players["dob"] = pd.to_datetime(players.pop("date_of_birth"), errors="coerce")
    players["height"] = players["height"].where(players["height"] > 0)   # 0은 "모름"이라는 뜻
    players = players.dropna(subset=["dob"]).set_index("player_id")

    vals = pd.read_csv(RAW / "player_market_value.csv")
    vals["date"] = pd.to_datetime(vals.pop("date_unix"), errors="coerce")
    vals = vals[(vals["value"] > 0) & vals["date"].notna()].sort_values(["player_id", "date"])

    transfers = pd.read_csv(RAW / "transfer_history.csv", low_memory=False, usecols=[
        "player_id", "transfer_date", "to_team_id", "to_team_name", "transfer_type", "transfer_fee"])
    transfers["date"] = pd.to_datetime(transfers.pop("transfer_date"), errors="coerce")
    transfers = transfers.dropna(subset=["date"]).sort_values(["player_id", "date"])

    perf = pd.read_csv(RAW / "player_performances.csv", low_memory=False, usecols=[
        "player_id", "season_name", "competition_id", "competition_name", "team_id",
        "nb_in_group", "nb_on_pitch", "goals", "assists", "subed_in",
        "yellow_cards", "second_yellow_cards", "direct_red_cards", "goals_conceded", "clean_sheets"])
    # 시즌 이름이 "17/18"이면 2018년에 끝나는 시즌, "2017"이면 2017년 한 해 동안 치르는 시즌
    season = perf.pop("season_name").astype(str)
    perf["calendar_year"] = ~season.str.contains("/")
    tail = pd.to_numeric(season.str[-2:], errors="coerce")
    perf["season_end"] = np.where(perf["calendar_year"], pd.to_numeric(season, errors="coerce"),
                                  np.where(tail >= 50, 1900 + tail, 2000 + tail))
    perf["goals"] = perf["goals"].fillna(0)
    perf["starts"] = (perf["nb_on_pitch"] - perf["subed_in"]).clip(lower=0)
    perf["cards"] = perf["yellow_cards"] + perf["second_yellow_cards"] + perf["direct_red_cards"]
    return players, vals, transfers, perf


# ---------------------------------------------------------------------------
# 작은 도구들
# ---------------------------------------------------------------------------
def last_before(df, t):
    """각 선수의 t 시점 이전 마지막 기록 (df는 선수·날짜순으로 정렬돼 있어야 한다)."""
    return df[df["date"] <= t].groupby("player_id").tail(1).set_index("player_id")


def completed_season(perf, year, back=0):
    """기준일(year년 7월 1일)에 이미 끝난 시즌의 출전 기록.

    가을에 시작하는 리그는 year년 여름에 끝난 시즌을, 한 해 동안 치르는 리그는
    year-1년 시즌을 쓴다. year년 시즌은 7월에 아직 진행 중이라 쓰면 미래 정보가 샌다.
    back=1이면 그보다 한 시즌 전.
    """
    end = np.where(perf["calendar_year"], year - 1 - back, year - back)
    return perf[perf["season_end"] == end]


def competition_levels(season_rows, log_value):
    """대회 수준 = 그 대회에서 5경기 이상 뛴 선수들의 (기준일) 몸값 중앙값, 로그."""
    r = season_rows[season_rows["nb_on_pitch"] >= 5]
    lv = r["player_id"].map(log_value).fillna(np.log10(VALUE_FLOOR))
    g = lv.groupby(r["competition_id"])
    return g.median()[g.size() >= 20]      # 20명 미만인 대회는 믿기 어려워 뺀다


def season_stats(season_rows, levels, suffix):
    """한 시즌 출전 기록을 선수별로 합친다."""
    r = season_rows.copy()
    r["level"] = r["competition_id"].map(levels)
    r["w"] = r["nb_on_pitch"].where(r["level"].notna(), 0)
    r["level_x_w"] = (r["level"] * r["w"]).fillna(0)
    g = r.groupby("player_id")
    out = pd.DataFrame({
        f"squad_{suffix}": g["nb_in_group"].sum(),          # 명단에 든 경기 수
        f"games_{suffix}": g["nb_on_pitch"].sum(),          # 실제 출전 경기 수
        f"starts_{suffix}": g["starts"].sum(),              # 선발 출전 수
        f"goals_{suffix}": g["goals"].sum(),
        f"assists_{suffix}": g["assists"].sum(),
        f"cards_{suffix}": g["cards"].sum(),
        f"conceded_{suffix}": g["goals_conceded"].sum(),    # 골키퍼용
        f"clean_sheets_{suffix}": g["clean_sheets"].sum(),  # 골키퍼용
        f"n_comps_{suffix}": g["competition_id"].nunique(),
        # 어느 수준의 대회에서 뛰었나: 출전 경기 수로 가중 평균한 대회 수준
        f"level_played_{suffix}": g["level_x_w"].sum() / g["w"].sum().replace(0, np.nan),
        # 3경기 이상 뛴 대회 중 가장 수준 높은 대회
        f"level_max_{suffix}": r[r["nb_on_pitch"] >= 3].groupby("player_id")["level"].max(),
    })
    return out


def club_context(club_now, value_now):
    """기준일 당시의 클럽 수준과 팀 내 몸값 순위.

    club_now: 선수별 기준일 당시 소속 클럽 (이적 이력에서 구한 것, 전체 선수)
    value_now: 선수별 기준일 몸값 (최근 1년 안에 기록이 있는 선수만)
    """
    squad = pd.DataFrame({"club_id": club_now, "value": value_now}).dropna()
    g = squad.groupby("club_id")["value"]
    clubs = pd.DataFrame({
        "club_squad_size": g.size(),                                   # 데이터에 잡힌 소속 선수 수
        "club_total_value": g.sum(),
        "club_top11_value": g.apply(lambda v: v.nlargest(11).mean()),  # 상위 11명 평균
    })
    rank_in_club = g.rank(ascending=False, method="min")               # 1 = 팀 내 최고 몸값
    return clubs, rank_in_club


# ---------------------------------------------------------------------------
# 기준일 하나의 표 만들기
# ---------------------------------------------------------------------------
def build_snapshot(year, players, vals, transfers, perf, data_end):
    t = pd.Timestamp(year, 7, 1)

    # --- 그날의 전체 선수 상태 (유망주만이 아니라 모든 나이) ---
    now = last_before(vals, t)
    now = now[now["date"] > t - YEAR]                 # 최근 1년 안에 몸값이 찍힌 선수
    value_now = now["value"]

    moves = transfers[transfers["date"] <= t]         # 미래 날짜로 예정된 이적은 제외
    last_move = moves.groupby("player_id").tail(1).set_index("player_id")
    in_club = ~last_move["to_team_name"].isin(NO_CLUB)
    club_now = last_move.loc[in_club, "to_team_id"]   # 임대 중이면 임대 간 팀
    club_name = last_move.loc[in_club, "to_team_name"]

    # --- 대회 수준, 팀별 주 대회(리그), 클럽 수준 ---
    last_season = completed_season(perf, year)
    levels = competition_levels(last_season, np.log10(value_now))
    team_games = last_season.groupby(["team_id", "competition_id"])["nb_on_pitch"].sum()
    team_league = team_games.groupby("team_id").idxmax().map(lambda pair: pair[1])   # 가장 많이 뛴 대회
    clubs, rank_in_club = club_context(club_now, value_now)

    # --- 대상 선수: 최근 1년 안에 몸값이 있고, 기준일에 MAX_AGE세 이하 ---
    df = pd.DataFrame({"value_now": value_now}).join(players, how="inner")
    df["age"] = (t - df["dob"]).dt.days / 365.25
    df = df[(df["age"] >= 15) & (df["age"] <= MAX_AGE)].copy()

    # --- 소속 클럽, 리그, 팀 내 위치 ---
    df["club_id"] = club_now
    df["club_name"] = club_name
    df = df.join(clubs, on="club_id")
    df["league"] = df["club_id"].map(team_league)
    df["league_level"] = df["league"].map(levels)
    df["rank_in_club"] = rank_in_club
    df["rank_pct_in_club"] = df["rank_in_club"] / df["club_squad_size"]   # 0에 가까울수록 팀 내 상위
    df["value_vs_club_top11"] = df["value_now"] / df["club_top11_value"]  # 1이면 주전급 몸값

    # --- 몸값 이력 ---
    past = vals[vals["date"] <= t]
    g = past.groupby("player_id")
    df["value_1y_ago"] = last_before(vals, t - YEAR)["value"]
    df["value_2y_ago"] = last_before(vals, t - 2 * YEAR)["value"]
    df["value_max_so_far"] = g["value"].max()
    df["n_valuations"] = g.size()
    df["years_since_first_valuation"] = (t - g["date"].min()).dt.days / 365.25

    # --- 이적 이력 ---
    g = moves.groupby("player_id")
    df["n_transfers"] = moves[moves["transfer_type"] == "Transfer"].groupby("player_id").size()
    df["n_loans"] = moves[moves["transfer_type"] == "Loan"].groupby("player_id").size()
    df[["n_transfers", "n_loans"]] = df[["n_transfers", "n_loans"]].fillna(0)
    df["on_loan"] = (last_move["transfer_type"] == "Loan").astype(float)
    df["max_transfer_fee"] = g["transfer_fee"].max()
    df["years_at_club"] = (t - last_move["date"]).dt.days / 365.25

    # --- 출전 기록: 지난 시즌(1s)과 그 전 시즌(2s) ---
    df = df.join(season_stats(last_season, levels, "1s"))
    df = df.join(season_stats(completed_season(perf, year, back=1), levels, "2s"))
    count_cols = [c for c in df.columns if c.endswith(("_1s", "_2s")) and not c.startswith("level_")]
    df[count_cols] = df[count_cols].fillna(0)         # 기록이 없으면 0경기 (수준 값은 빈 채로 둔다)

    # --- 정답: 3년 뒤(직전 1년 창 안) 마지막 몸값 ---
    t_future = t + pd.DateOffset(years=HORIZON_YEARS)
    df["value_future"] = np.nan
    if t_future <= data_end:
        future = last_before(vals, t_future)
        df["value_future"] = future.loc[future["date"] > t_future - YEAR, "value"]

    df.insert(0, "snapshot_year", year)
    return df.reset_index(names="player_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "dataset.parquet"))
    args = ap.parse_args()

    print("원본 CSV 읽는 중...")
    players, vals, transfers, perf = load_raw()
    data_end = vals["date"].max()
    print(f"선수 {len(players):,}명 | 몸값 {len(vals):,}건 (~{data_end.date()}) | "
          f"이적 {len(transfers):,}건 | 출전 {len(perf):,}건\n")

    frames = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        frames.append(build_snapshot(year, players, vals, transfers, perf, data_end))
        print(f"  {year}년 7월 1일: {len(frames[-1]):,}명")
    ds = pd.concat(frames, ignore_index=True).drop(columns=["dob"])

    ds["has_label"] = ds["value_future"].notna().astype(int)
    # 몸값은 수만~수억 유로로 치우쳐 있어서 로그로 바꿔 학습한다
    ds["log_value_now"] = np.log10(ds["value_now"])
    ds["log_value_future"] = np.log10(ds["value_future"])
    ds["log_growth"] = ds["log_value_future"] - ds["log_value_now"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out, index=False)

    print(f"\n저장: {out}  ({len(ds):,}행, {ds.shape[1]}열)")
    pd.set_option("display.width", 200)
    summary = ds.groupby("snapshot_year").agg(
        선수수=("player_id", "size"),
        정답있음=("has_label", "mean"),
        클럽있음=("club_id", lambda s: s.notna().mean()),
        리그수준있음=("league_level", lambda s: s.notna().mean()),
        지난시즌출전=("games_1s", lambda s: (s > 0).mean()),
        현재몸값_중앙값=("value_now", "median"),
        미래몸값_중앙값=("value_future", "median"),
    ).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()
