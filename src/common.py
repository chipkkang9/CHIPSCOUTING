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

    # 같은 해 기준 상대값: 그 해 대상 선수들의 중앙값을 뺀다 (기준일 당시 정보만 쓴다)
    by_year = df.groupby("snapshot_year")
    for raw, rel in YEAR_RELATIVE.items():
        df[rel] = df[raw] - by_year[raw].transform("median")
    # 정답도 같은 해 평균을 뺀다. "시장 평균보다 얼마나 더 컸는가"를 맞히는 문제가 된다.
    df[TARGET] = df["log_growth"] - by_year["log_growth"].transform("mean")

    # 키와 주발의 빈 값 채우기. 프로필이 비어 있다는 것 자체가 "지금 무명이다"라는
    # 미래 정보라서, 모델이 그걸 신호로 쓰지 못하게 평범한 값으로 메운다.
    df["height"] = (df["height"]
                    .fillna(df.groupby("position")["height"].transform("median"))
                    .fillna(df["height"].median()))
    df["foot"] = df["foot"].fillna("right")
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def load_dataset():
    """학습 테이블을 읽어 feature를 붙인다."""
    return add_features(pd.read_parquet(DATASET))


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
