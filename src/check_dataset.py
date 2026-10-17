"""학습 테이블 품질 검사.

모델을 학습시키기 전에 "미래 정보가 새는 곳"과 "시기에 따라 기록 방식이
달라지는 곳"이 없는지 살핀다. 이전 데이터에서 실제로 겪은 문제들을 검사 항목으로 만들었다.

실행:  python src/check_dataset.py
"""
import numpy as np
import pandas as pd

from common import DATASET

pd.set_option("display.width", 220)
pd.set_option("display.unicode.east_asian_width", True)

# 시기에 따라 채워진 비율이 달라지면 안 되는 열들
COVERAGE_COLS = ["club_id", "league_level", "club_top11_value", "value_1y_ago", "value_2y_ago",
                 "max_transfer_fee", "years_at_club", "level_played_1s", "height", "foot"]


def check_coverage_over_time(ds):
    """검사 1: 열마다 값이 채워진 비율이 연도에 따라 급변하지 않는가."""
    print("=== 검사 1. 연도별로 값이 채워진 비율 ===")
    print("특정 연도부터 갑자기 오르거나 내리면, 과거 선수와 최근 선수의 기록 방식이 다르다는 신호다.\n")
    cov = ds.groupby("snapshot_year")[COVERAGE_COLS].agg(lambda s: s.notna().mean()).round(2)
    print(cov.T.to_string())
    swing = (cov.max() - cov.min()).round(2)
    print("\n최대-최소 차이:", swing.to_dict())
    print("0.15 넘는 열:", list(swing[swing > 0.15].index) or "없음", "\n")


def check_labels_over_time(ds):
    """검사 2: 정답이 있는 비율이 특정 시기에 몰려 떨어지지 않는가."""
    print("=== 검사 2. 연도별 정답(3년 뒤 몸값) 있는 비율 ===")
    past = ds[ds["value_future"].notna() | (ds["snapshot_year"] <= ds.loc[ds["has_label"] == 1, "snapshot_year"].max())]
    table = past.groupby("snapshot_year").agg(
        선수수=("has_label", "size"), 정답있음=("has_label", "mean"),
        평균성장=("log_growth", "mean"), 성장_표준편차=("log_growth", "std")).round(3)
    print(table.to_string(), "\n")


def check_direction(ds):
    """검사 3: 수준 높은 곳에 있는 선수가 더 잘 살아남는가 (상식과 같은 방향인가)."""
    print("=== 검사 3. 소속 리그 수준별 생존율과 성장 ===")
    print("성공한 선수만 골라 담긴 데이터라면, 수준 낮은 리그 선수의 생존율이 오히려 높게 나온다.\n")
    past = ds[ds["snapshot_year"] <= ds.loc[ds["has_label"] == 1, "snapshot_year"].max()].copy()
    past["리그수준"] = pd.cut(past["league_level"], [0, 4.5, 5.2, 5.8, 6.4, 9],
                          labels=["매우 낮음", "낮음", "중간", "높음", "매우 높음"]).astype(object)
    past["리그수준"] = past["리그수준"].fillna("모름")
    table = past.groupby("리그수준").agg(
        선수수=("has_label", "size"), 정답있음=("has_label", "mean"), 평균성장=("log_growth", "mean")).round(3)
    print(table.reindex(["매우 높음", "높음", "중간", "낮음", "매우 낮음", "모름"]).to_string(), "\n")


def check_missingness_leak(ds):
    """검사 4: "값이 비어 있다"는 사실이 미래를 알려주지는 않는가."""
    print("=== 검사 4. 값이 비어 있는 선수와 채워진 선수의 차이 ===")
    print("비어 있는 쪽의 정답 비율이나 성장이 크게 다르면, 빈 값 자체가 미래 정보일 수 있다.\n")
    past = ds[ds["snapshot_year"] <= ds.loc[ds["has_label"] == 1, "snapshot_year"].max()]
    rows = {}
    for col in COVERAGE_COLS:
        missing = past[col].isna()
        if missing.mean() < 0.02:
            continue
        rows[col] = {
            "빈 비율": missing.mean(),
            "정답있음(빈 값)": past.loc[missing, "has_label"].mean(),
            "정답있음(채워짐)": past.loc[~missing, "has_label"].mean(),
            "평균성장(빈 값)": past.loc[missing, "log_growth"].mean(),
            "평균성장(채워짐)": past.loc[~missing, "log_growth"].mean(),
        }
    print(pd.DataFrame(rows).T.round(3).to_string(), "\n")


def check_known_players(ds):
    """검사 5: 아는 선수의 행이 실제 커리어와 맞는가."""
    print("=== 검사 5. 아는 선수로 눈으로 확인 ===")
    cols = ["snapshot_year", "age", "value_now", "club_name", "league", "games_1s", "goals_1s", "value_future"]
    for name in ["Erling Haaland", "Kang-in Lee", "Jude Bellingham"]:
        print(name)
        print(ds.loc[ds["name"] == name, cols].round(1).to_string(index=False), "\n")


def main():
    ds = pd.read_parquet(DATASET)
    print(f"{DATASET.name}: {len(ds):,}행, {ds.shape[1]}열 (그중 유망주 {ds['is_prospect'].sum():,}행만 검사)\n")
    ds = ds[ds["is_prospect"] == 1]
    check_coverage_over_time(ds)
    check_labels_over_time(ds)
    check_direction(ds)
    check_missingness_leak(ds)
    check_known_players(ds)


if __name__ == "__main__":
    main()
