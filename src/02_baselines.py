"""3단계: 기준선 재기.

학습이 필요 없는(또는 평균만 내는) 단순한 방법 세 가지의 성적을 잰다.
앞으로 만들 모델은 이 성적을 넘어야 "뭔가 배웠다"고 말할 수 있다.

모든 함수는 (train, df)를 받아 df 각 행의 "같은 해 평균 대비 성장" 예측을 돌려준다.

실행:  python src/02_baselines.py
"""
import numpy as np
import pandas as pd

from common import TARGET, evaluate, load_dataset, summarize, walk_forward


def predict_market_average(train, df):
    """기준선 1: 모든 선수가 그 해 평균만큼만 성장한다 (상대 성장 0)."""
    return pd.Series(0.0, index=df.index)


def predict_age_growth(train, df):
    """기준선 2: 같은 나이 선수들의 평균 상대 성장만큼 성장한다."""
    growth_by_age = train.groupby(np.floor(train["age"]))[TARGET].mean()
    return np.floor(df["age"]).map(growth_by_age).fillna(0.0)


def predict_age_value_growth(train, df, n_bins=10):
    """기준선 3: 나이와 현재 몸값 수준이 같은 선수들의 평균 상대 성장만큼 성장한다."""
    # 현재 몸값(같은 해 중앙값 대비)을 학습 데이터 기준 10개 구간으로 자른다
    edges = np.unique(train["rel_value_now"].quantile(np.linspace(0, 1, n_bins + 1)).values)
    edges[0], edges[-1] = -np.inf, np.inf

    def cell(d):
        return pd.DataFrame({"age": np.floor(d["age"]),
                             "bin": pd.cut(d["rel_value_now"], edges, labels=False)}, index=d.index)

    c_train = cell(train)
    table = train.groupby([c_train["age"], c_train["bin"]])[TARGET].mean()
    growth = pd.Series(table.reindex(pd.MultiIndex.from_frame(cell(df))).values, index=df.index)
    return growth.fillna(0.0)


BASELINES = {
    "1. 모두 평균만큼 성장": predict_market_average,
    "2. 나이별 평균 성장": predict_age_growth,
    "3. 나이x몸값 구간별 평균 성장": predict_age_value_growth,
}


def main():
    ds = load_dataset()
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)

    results = {name: {} for name in BASELINES}
    for year, train, test in walk_forward(ds):
        print(f"{year}년 평가: 학습 {train['snapshot_year'].min()}~{train['snapshot_year'].max()}년 "
              f"{len(train):,}행 -> 평가 {len(test):,}행")
        for name, fn in BASELINES.items():
            results[name][year] = evaluate(test, fn(train, test))

    for name in BASELINES:
        print(f"\n=== {name} ===")
        print(summarize(results[name]).to_string())


if __name__ == "__main__":
    main()
