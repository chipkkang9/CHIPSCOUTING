"""여러 단계가 같이 쓰는 도구: 대상 선수 고르기, feature 만들기, 시간 순서 검증, 채점.

모든 모델(기준선, 트리, 신경망)을 같은 데이터, 같은 검증 방식, 같은 채점 방식으로
비교하려고 한 곳에 모아 둔다.

시간이 흐르는 데이터라서 지키는 원칙 네 가지:
  1. 기준일 당시 알 수 있었던 정보만 쓴다          (point-in-time)
  2. 과거로 배우고 그 이후를 맞힌다                (walk-forward)
  3. 정답이 확정되기 전의 행으로는 배우지 않는다    (3년 간격, embargo)
  4. 돈의 단위는 같은 해 선수들과 비교한 값으로 바꾼다 (해마다 시장 전체가 오르므로)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "dataset.parquet"
FBREF_DATASET = ROOT / "data" / "processed" / "fbref_features.parquet"   # 01b_add_fbref.py가 만든다
EA_DATASET = ROOT / "data" / "processed" / "ea_ratings.parquet"         # 01c_add_ea.py가 만든다

PROSPECT_AGE = 21                        # "유망주" 나이 상한 (01_build_dataset.py와 같아야 한다)
HORIZON_YEARS = 3                        # 정답은 기준일 3년 뒤 (01_build_dataset.py와 같아야 한다)
DEV_YEARS = [2016, 2017, 2018, 2019]     # 개발용: 설정을 고를 때 본다
TEST_YEARS = [2020, 2021, 2022]          # 시험용: 마지막 확인
EVAL_YEARS = DEV_YEARS + TEST_YEARS
TOP_FRACTION = 0.05                      # "특급" = 그 해 상위 5%

# ---------------------------------------------------------------------------
# 모델 입력(feature) 정의. 트리 모델과 신경망이 같은 목록을 쓴다.
# ---------------------------------------------------------------------------
CATEGORICAL = ["position", "main_position", "foot"]

NUMERIC = [
    # 프로필
    "age", "height",
    # 몸값 이력 (rel_ = 같은 해 유망주 중앙값 대비)
    "rel_value_now", "rel_growth_1y", "rel_growth_2y", "drop_from_peak",
    "n_valuations", "years_since_first_valuation",
    # 클럽·리그 수준, 팀 내 위치
    "club_squad_size", "rel_club_top11", "rel_league_level",
    "rank_pct_in_club", "log_value_vs_club_top11",
    # 이적 이력
    "n_transfers", "n_loans", "on_loan", "log_max_transfer_fee", "years_at_club",
    # 지난 시즌 출전 기록
    "squad_1s", "games_1s", "starts_1s", "goals_1s", "assists_1s", "cards_1s",
    "conceded_1s", "clean_sheets_1s", "n_comps_1s",
    "rel_level_played_1s", "rel_level_max_1s",
    "start_share_1s", "play_share_1s", "goals_per_game_1s", "plays_below_club_1s",
    # 그 전 시즌 출전 기록
    "squad_2s", "games_2s", "starts_2s", "goals_2s", "assists_2s",
    "rel_level_played_2s", "rel_level_max_2s", "level_change",
]
FEATURES = NUMERIC + CATEGORICAL

# FBref 세부 지표 (5대 리그에서 지난 시즌 450분 이상 뛴 선수에게만 있다. 나머지는 빈 값)
FBREF_FEATURES = [
    # 공격
    "fb_goals_p90", "fb_assists_p90", "fb_xg_p90", "fb_npxg_p90", "fb_xag_p90",
    "fb_shots_p90", "fb_shots_on_target_p90", "fb_shot_on_target_pct", "fb_goals_minus_xg_p90",
    "fb_key_passes_p90", "fb_shot_creating_actions_p90",
    "fb_touches_p90", "fb_touches_att_third_p90", "fb_touches_att_pen_p90",
    "fb_take_ons_p90", "fb_take_on_pct", "fb_dispossessed_p90", "fb_miscontrols_p90",
    # 패스
    "fb_passes_final_third_p90", "fb_passes_penalty_area_p90", "fb_progressive_passes_p90",
    "fb_through_balls_p90", "fb_progressive_carries_p90", "fb_progressive_received_p90",
    "fb_pass_pct", "fb_long_completed_p90", "fb_long_pass_pct",
    # 수비
    "fb_tackles_p90", "fb_tackle_win_pct", "fb_dribbler_tackle_pct", "fb_interceptions_p90",
    "fb_clearances_p90", "fb_blocks_p90", "fb_recoveries_p90", "fb_errors_p90",
    # 경합, 파울
    "fb_aerials_won_p90", "fb_aerial_win_pct", "fb_fouls_p90", "fb_fouled_p90",
    # 출전
    "fb_minutes",
]
TARGET = "rel_growth_future"       # 같은 해 유망주 평균 대비 3년간 성장 (로그)

# 해마다 시장 전체가 움직이는 값들. 같은 해 중앙값을 빼서 "그 해 기준 상대값"으로 바꾼다.
YEAR_RELATIVE = {
    "log_value_now": "rel_value_now",
    "growth_1y": "rel_growth_1y",
    "growth_2y": "rel_growth_2y",
    "log_club_top11": "rel_club_top11",
    "league_level": "rel_league_level",
    "level_played_1s": "rel_level_played_1s",
    "level_max_1s": "rel_level_max_1s",
    "level_played_2s": "rel_level_played_2s",
    "level_max_2s": "rel_level_max_2s",
}


def add_features(df):
    """원본 열을 조합해 모델에 넣을 열을 만든다 (feature engineering)."""
    df = df.copy()
    log_now = np.log10(df["value_now"])
    # 몸값 추세: 1년 전, 2년 전 대비 몇 배가 됐는지 (로그). 과거 기록이 없으면 빈 값.
    df["growth_1y"] = log_now - np.log10(df["value_1y_ago"])
    df["growth_2y"] = log_now - np.log10(df["value_2y_ago"])
    # 최고점 대비 하락: 0이면 지금이 최고점, 음수면 꺾인 적이 있다
    df["drop_from_peak"] = log_now - np.log10(df["value_max_so_far"])
    df["log_club_top11"] = np.log10(df["club_top11_value"])
    df["log_value_vs_club_top11"] = np.log10(df["value_vs_club_top11"])
    df["log_max_transfer_fee"] = np.log10(df["max_transfer_fee"].where(df["max_transfer_fee"] > 0))

    # 출전 기회: 포지션과 상관없이 "감독이 얼마나 믿고 내보내는가"
    games = df["games_1s"].where(df["games_1s"] >= 5)            # 5경기 미만이면 비율이 튀므로 빈 값
    df["start_share_1s"] = df["starts_1s"] / games               # 출전 경기 중 선발 비율
    df["play_share_1s"] = df["games_1s"] / df["squad_1s"].where(df["squad_1s"] >= 5)   # 명단 대비 출전 비율
    df["goals_per_game_1s"] = df["goals_1s"] / games
    # 소속 클럽의 리그보다 낮은 수준에서 뛰었나 (양수면 2군이나 유스 팀에서 주로 뛴 것)
    df["plays_below_club_1s"] = df["league_level"] - df["level_played_1s"]
    # 한 시즌 사이에 뛰는 무대가 올라갔나
    df["level_change"] = df["level_played_1s"] - df["level_played_2s"]

    # 같은 해 기준 상대값: 그 해 "유망주"(21세 이하)들의 중앙값을 뺀다 (기준일 당시 정보만 쓴다).
    # 22~25세 행이 섞여 있어도 기준은 유망주로 고정한다. 그래야 표에 누가 들어 있든 같은 값이 나온다.
    prospects = df[df["age"] <= PROSPECT_AGE].groupby("snapshot_year")
    for raw, rel in YEAR_RELATIVE.items():
        df[rel] = df[raw] - df["snapshot_year"].map(prospects[raw].median())
    # 정답도 같은 해 유망주 평균을 뺀다. "또래 평균보다 얼마나 더 컸는가"를 맞히는 문제가 된다.
    df[TARGET] = df["log_growth"] - df["snapshot_year"].map(prospects["log_growth"].mean())

    # 키와 주발의 빈 값 채우기. 프로필이 비어 있다는 것 자체가 "지금 무명이다"라는
    # 미래 정보라서, 모델이 그걸 신호로 쓰지 못하게 평범한 값으로 메운다.
    df["height"] = (df["height"]
                    .fillna(df.groupby("position")["height"].transform("median"))
                    .fillna(df["height"].median()))
    df["foot"] = df["foot"].fillna("right")
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def load_dataset(with_fbref=False, max_age=PROSPECT_AGE):
    """학습 테이블을 읽어 feature를 붙인다.

    with_fbref=True면 FBref 세부 지표도 붙인다.
    max_age: 이 나이까지의 행만 쓴다. 기본은 유망주(21세)까지이고, 04처럼 학습 사례를
    늘리고 싶을 때만 25로 준다.
    """
    ds = pd.read_parquet(DATASET)
    ds = ds[ds["age"] <= max_age]
    if with_fbref:
        fb = pd.read_parquet(FBREF_DATASET, columns=["player_id", "snapshot_year"] + FBREF_FEATURES)
        ds = ds.merge(fb, on=["player_id", "snapshot_year"], how="left")
    return add_features(ds)


# ---------------------------------------------------------------------------
# 기량(EA 평점) 붙이기
# ---------------------------------------------------------------------------
# 기준일에 이미 공개돼 있던 가장 최근 EA 평점. 입력(feature)으로 쓸 수 있다.
EA_FEATURES = ["ea_overall_prev", "ea_potential_prev", "ea_room_prev", "ea_months_old"]


def attach_ea(ds):
    """각 (선수, 기준일) 행에 EA 평점 세 가지를 붙인다.

    ea_*_prev        기준일(7월 1일) 이전 400일 안에 공개된 가장 최근 평점. 입력으로 쓴다.
    ability_now      기준일 직후(7월 2일~12월 31일)에 공개된 평점 = "그 여름의 기량". 현재 기량 모델의 정답.
    ability_future   3년 뒤 같은 시기에 공개된 평점 = "3년 뒤 기량". 성장 모델의 정답.

    EA는 매년 9월 무렵 새 판을 내면서 지난 시즌 활약을 반영해 평점을 다시 매긴다. 그래서 9월 평점을
    "그해 여름 기준 기량"으로 본다. 2024년 9월 판(FC 25 출시 평점)은 구하지 못해서 2024년 정답은 비어 있다.
    """
    ea = pd.read_parquet(EA_DATASET).sort_values("published")
    parts = []
    for year, g in ds.groupby("snapshot_year"):
        t = pd.Timestamp(year, 7, 1)
        rows = pd.DataFrame(index=g.index)
        pid = g["player_id"]

        prev = ea[(ea["published"] <= t) & (ea["published"] > t - pd.Timedelta(days=400))]
        prev = prev.groupby("player_id").tail(1).set_index("player_id")
        rows["ea_overall_prev"] = pid.map(prev["overall"])
        rows["ea_potential_prev"] = pid.map(prev["potential"])
        rows["ea_months_old"] = pid.map((t - prev["published"]).dt.days / 30.4)

        for col, y in [("ability_now", year), ("ability_future", year + HORIZON_YEARS)]:
            start, end = pd.Timestamp(y, 7, 1), pd.Timestamp(y, 12, 31)
            lab = ea[(ea["published"] > start) & (ea["published"] <= end)]
            rows[col] = pid.map(lab.groupby("player_id")["overall"].last())
        parts.append(rows)
    out = ds.join(pd.concat(parts))
    out["ea_room_prev"] = out["ea_potential_prev"] - out["ea_overall_prev"]   # EA가 본 남은 성장 여지
    return out


# ---------------------------------------------------------------------------
# 시간 순서 검증 (walk-forward)
# ---------------------------------------------------------------------------
def walk_forward(ds, years=EVAL_YEARS):
    """평가 연도마다 (연도, 학습 데이터, 평가 데이터)를 차례로 내놓는다.

    평가 연도가 Y이면 학습에는 기준일이 Y-3년 이하인 행만 쓴다. 그보다 뒤의 행은
    Y년 7월 1일에 아직 정답(3년 뒤 몸값)이 나오지 않았기 때문이다.
    """
    labeled = ds[ds["has_label"] == 1]
    for year in years:
        test = labeled[labeled["snapshot_year"] == year]
        train = labeled[labeled["snapshot_year"] <= year - HORIZON_YEARS]
        # 평가에 나오는 선수는 학습에서 뺀다: 한 번도 못 본 선수로 평가한다
        train = train[~train["player_id"].isin(test["player_id"])]
        yield year, train, test


def split_last_year(train):
    """학습 데이터의 마지막 해를 떼어 낸다 (언제 학습을 멈출지 정하는 용도)."""
    last = train["snapshot_year"].max()
    return train[train["snapshot_year"] < last], train[train["snapshot_year"] == last]


# ---------------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------------
def top_hit_rate(score, truth):
    """score 상위 5%로 뽑은 선수 중, truth 기준 실제 상위 5%인 비율."""
    if score.nunique() <= 1:      # 모든 선수에게 같은 점수를 주면 순위를 매길 수 없다
        return np.nan
    k = int(np.ceil(len(truth) * TOP_FRACTION))
    picked = set(score.nlargest(k).index)
    actual = set(truth.nlargest(k).index)
    return len(picked & actual) / k


def rank_corr(score, truth):
    """순위 상관(Spearman). 1이면 순위가 완전히 같고, 0이면 무관하다."""
    if score.nunique() <= 1:
        return np.nan
    return score.corr(truth, method="spearman")


def evaluate(df, pred):
    """한 해의 예측을 채점한다.

    df: 같은 기준일 연도의 행들
    pred: 같은 index를 가진 예측값 (같은 해 평균 대비 성장, 로그)
    """
    err = (pred - df[TARGET]).abs().mean()
    pred_future = df["log_value_now"] + pred     # 같은 해 안에서는 순위 비교에 이걸로 충분하다
    return {
        "선수 수": len(df),
        "평균 몇 배 틀림": 10 ** err,
        "순위상관(성장)": rank_corr(pred, df[TARGET]),
        "순위상관(몸값)": rank_corr(pred_future, df["log_value_future"]),
        "적중률(급성장)": top_hit_rate(pred, df[TARGET]),
        "적중률(미래 스타)": top_hit_rate(pred_future, df["log_value_future"]),
    }


def summarize(results):
    """연도별 채점 결과(dict)를 표로 만들고 개발/시험 평균을 덧붙인다."""
    table = pd.DataFrame(results).T
    table.loc["개발 평균"] = table.loc[[y for y in DEV_YEARS if y in table.index]].mean()
    table.loc["시험 평균"] = table.loc[[y for y in TEST_YEARS if y in table.index]].mean()
    table["선수 수"] = table["선수 수"].round().astype(int)
    return table.round(3)
