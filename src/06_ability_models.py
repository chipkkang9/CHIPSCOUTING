"""6단계: 기량 모델. "지금 얼마나 잘하나"와 "1~5년 뒤 얼마나 잘할까"를 1~99 점수로 예측한다.

몸값 대신 EA FC 평점(overall)을 기량의 기준으로 쓴다 (01c_add_ea.py).

  현재 기량 모델  입력: 기준일(7월 1일)까지의 기록            정답: 그해 9월 EA 평점
  성장 모델       입력: 같은 기록 + 현재 기량 추정 + 예측 기간   정답: N년 뒤 9월 EA 평점 (N = 1~5)

현재 기량 모델이 먼저 "지금 기량"을 추정하고, 성장 모델은 거기서 N년 동안 몇 점 오르내릴지를
맞힌다 (기량 수준이 아니라 변화폭을 맞힌다). 트리 모델은 학습 데이터의 최고점 근처에서 값을
깎는 성질이 있어서, 수준을 바로 맞히게 하면 최상위 유망주의 성장을 과소평가하기 때문이다.

예측 기간(1~5년)은 모델마다 따로 두지 않고 입력 하나("몇 년 뒤인가")로 넣어서, 모델 하나가
다섯 기간을 함께 배운다. 기간별로 따로 학습시키면 1년 뒤 +3, 2년 뒤 +1, 3년 뒤 +4처럼
들쭉날쭉해지기 쉽다.

성장 모델의 정답에는 구멍이 있다. N년 뒤 EA가 평가하지 않는 리그로 내려간 선수는 정답이 없다.
그대로 두면 "살아남은 선수"만 배우게 되므로, 그런 선수는 N년 뒤 시점에 현재 기량 모델이 추정한
값을 정답 대신 쓴다 (아예 기록이 사라진 선수는 여전히 빠진다).

실행:  python src/06_ability_models.py            (전체, 약 10분)
       python src/06_ability_models.py --quick    (1-1 비교를 건너뛴다)
출력:  models/ability_current.txt, models/ability_growth.txt    최종 모델
       models/ability_interval.json                             예측 기간·나이대별 80% 범위
       data/processed/scout_2025.parquet                        2025년 7월 기준 전 선수 점수 (검색용)
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (EA_DATASET, EA_FEATURES, FEATURES, PROSPECT_AGE, ROOT, attach_ea,
                    load_dataset, rank_corr)

PARAMS = {
    "objective": "regression", "metric": "l1", "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "cat_smooth": 20, "seed": 0, "verbose": -1,
}
MAX_ROUNDS, PATIENCE = 3000, 100

VALUE_ONLY = ["rel_value_now", "age", "position", "main_position"]
CURRENT_FEATURES = FEATURES + EA_FEATURES                  # 현재 기량 모델의 입력
GROWTH_FEATURES = FEATURES + ["current_est", "horizon"]    # 성장 모델의 입력 (이전 EA 평점은 current_est 안에 녹아 있다)
FEATURE_SETS = {
    "몸값만": VALUE_ONLY,
    "경기 기록 + 몸값": FEATURES,
    "경기 기록 + 몸값 + 이전 EA 평점": CURRENT_FEATURES,
}
HORIZONS = [1, 2, 3, 4, 5]
FIRST_ESTIMATE_YEAR = 2015          # EA 평점이 2014년부터라 그 다음 해부터 "과거로 배운" 추정이 가능하다
EVAL_YEARS = [2019, 2020, 2021, 2022]
SCOUT_YEAR = 2025
AGE_BANDS = [0, 19, 21, 23, 26, 29, 99]               # 예측 범위를 나이대별로 따로 잰다
AGE_LABELS = ["~19세", "20~21세", "22~23세", "24~26세", "27~29세", "30세~"]


# ---------------------------------------------------------------------------
# 학습 도구
# ---------------------------------------------------------------------------
def fit(train, feats, target):
    """LightGBM 학습. 학습 데이터 안에서 떼어 낸 부분으로 나무 개수를 정하고, 전체로 다시 학습한다."""
    years = sorted(train["snapshot_year"].unique())
    if len(years) >= 2:                               # 마지막 해를 떼어 낸다 (과거로 배워 최근을 맞히기)
        hold = train["snapshot_year"] == years[-1]
    else:                                             # 한 해뿐이면 선수 10%를 떼어 낸다
        rng = np.random.default_rng(0)
        players = train["player_id"].unique()
        hold = train["player_id"].isin(rng.choice(players, size=max(1, len(players) // 10), replace=False))
    d_fit = lgb.Dataset(train.loc[~hold, feats], train.loc[~hold, target])
    d_hold = lgb.Dataset(train.loc[hold, feats], train.loc[hold, target], reference=d_fit)
    probe = lgb.train(PARAMS, d_fit, MAX_ROUNDS, valid_sets=[d_hold],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return lgb.train(PARAMS, lgb.Dataset(train[feats], train[target]), probe.best_iteration)


def metrics(pred, truth):
    """평균 오차(점), 3점 이내로 맞힌 비율, 순위상관."""
    err = (pred - truth).abs()
    return {"평균 오차(점)": err.mean(), "3점 이내": (err <= 3).mean(), "순위상관": rank_corr(pred, truth),
            "선수 수": len(truth)}


def show(title, rows):
    print(f"\n{title}")
    table = pd.DataFrame(rows).T
    table["선수 수"] = table["선수 수"].astype(int)
    print(table.round(3).to_string())


def age_band(age):
    return pd.cut(age, AGE_BANDS, labels=AGE_LABELS, right=True).astype(str)


# ---------------------------------------------------------------------------
# 1. 현재 기량 모델
# ---------------------------------------------------------------------------
def compare_current_inputs(ds):
    """입력 묶음별로 현재 기량을 얼마나 맞히나. 평가 연도의 선수는 학습에서 뺀다 (처음 보는 선수)."""
    print("=" * 70)
    print("1-1. 현재 기량: 어떤 정보가 얼마나 도움이 되나 (처음 보는 선수로 평가)")
    print("=" * 70)
    labeled = ds[ds["ability_now"].notna()]
    results = {name: [] for name in FEATURE_SETS}
    for year in [2021, 2022, 2023, 2025]:
        test = labeled[labeled["snapshot_year"] == year]
        train = labeled[(labeled["snapshot_year"] < year) & ~labeled["player_id"].isin(test["player_id"])]
        for name, feats in FEATURE_SETS.items():
            pred = pd.Series(fit(train, feats, "ability_now").predict(test[feats]), index=test.index)
            results[name].append(test.assign(pred=pred)[["age", "ability_now", "pred"]])
    for subset, cond in [("전체 선수", lambda r: r["age"] > 0), ("21세 이하", lambda r: r["age"] <= PROSPECT_AGE)]:
        rows = {}
        for name, parts in results.items():
            r = pd.concat(parts)
            r = r[cond(r)]
            rows[name] = metrics(r["pred"], r["ability_now"])
        show(f"[{subset}, 평가 연도 2021·2022·2023·2025 합산]", rows)


def current_estimates(ds):
    """해마다 그 전 해까지의 EA 정답으로만 학습한 모델로 그해 모든 선수의 현재 기량을 추정한다.

    이렇게 만든 current_est는 기준일에 실제로 알 수 있었던 정보만으로 만든 값이라,
    성장 모델의 입력과 "추정 정답"에 그대로 쓸 수 있다.
    """
    print("\n" + "=" * 70)
    print("1-2. 현재 기량: 해마다 과거 데이터로만 학습해 추정 (실제 쓰는 방식 그대로 평가)")
    print("=" * 70)
    est = pd.Series(np.nan, index=ds.index)
    labeled = ds[ds["ability_now"].notna()]
    rows, model = {}, None
    for year in range(FIRST_ESTIMATE_YEAR, SCOUT_YEAR + 1):
        model = fit(labeled[labeled["snapshot_year"] < year], CURRENT_FEATURES, "ability_now")
        mask = ds["snapshot_year"] == year
        est[mask] = model.predict(ds.loc[mask, CURRENT_FEATURES])
        truth = ds.loc[mask, "ability_now"]
        if truth.notna().any():
            rows[year] = metrics(est[mask][truth.notna()], truth.dropna())
    show("[연도별: 그해 9월 EA 평점과 비교]", rows)
    return est, model              # model = 2025년 추정에 쓴 모델 (2023년까지의 정답으로 학습)


# ---------------------------------------------------------------------------
# 2. 성장 모델 (1~5년 뒤 기량)
# ---------------------------------------------------------------------------
def ea_label_by_year():
    """(선수, 연도) -> 그해 7월 2일~12월 31일에 공개된 EA 평점 = "그해 여름의 기량"."""
    ea = pd.read_parquet(EA_DATASET).sort_values("published")
    ea["year"] = ea["published"].dt.year
    summer = ea[ea["published"] > pd.to_datetime(ea["year"].astype(str) + "-07-01")]
    return summer.groupby(["player_id", "year"])["overall"].last()


def growth_table(ds):
    """예측 기간마다 (선수, 기준일, 기간) 한 행씩 쌓은 학습 표를 만든다.

    정답: N년 뒤 EA 평점이 있으면 그것, 없으면 N년 뒤 시점의 현재 기량 추정값.
    변화폭(delta) = 정답 - 지금의 현재 기량 추정.
    """
    ea_label = ea_label_by_year()
    est = ds.set_index(["player_id", "snapshot_year"])["current_est"]
    base = ds[ds["current_est"].notna()]
    cols = list(dict.fromkeys(FEATURES + ["current_est", "player_id", "snapshot_year", "age", "ea_overall_prev"]))
    parts = []
    for h in HORIZONS:
        keys = pd.MultiIndex.from_arrays([base["player_id"], base["snapshot_year"] + h])
        ea_h = pd.Series(ea_label.reindex(keys).values, index=base.index)
        est_h = pd.Series(est.reindex(keys).values, index=base.index)
        part = base[cols].assign(horizon=h, ea_label=ea_h, label=ea_h.fillna(est_h))
        parts.append(part[part["label"].notna()])
    long = pd.concat(parts, ignore_index=True)
    long["delta"] = long["label"] - long["current_est"]
    long["label_year"] = long["snapshot_year"] + long["horizon"]
    return long


def baseline_change(train, test):
    """기준선: 현재 추정 + 같은 나이·같은 기간 선수들의 평균 변화."""
    key = lambda d: np.floor(d["age"]).astype(int).astype(str) + "_" + d["horizon"].astype(str)
    mean_change = train.groupby(key(train))["delta"].mean()
    return test["current_est"] + key(test).map(mean_change).fillna(0).values


def evaluate_growth(long):
    """평가 연도마다 그 시점에 정답이 나와 있던 행으로만 학습해서, 그해 선수들의 1~5년 뒤를 맞힌다.

    N년 뒤 정답(9월 평점)이 평가 연도 7월 1일 전에 공개돼 있어야 하므로, 학습에는
    "기준일 + 기간 <= 평가 연도 - 1"인 행만 쓴다.
    5년 뒤는 이 조건을 지키면 평가할 수 있는 해가 없어서, 2020년 평가에서만 조건을 1년 늦춘다
    (정답이 평가 시점보다 두 달 늦게 나온 행까지 학습에 쓴다).
    """
    print("\n" + "=" * 70)
    print("2. 성장 모델: 7월 1일까지의 기록으로 1~5년 뒤 9월 EA 평점 맞히기")
    print("=" * 70)
    collected = []
    runs = [(y, 1) for y in EVAL_YEARS] + [(2020, 0)]           # (평가 연도, 정답 공개를 기다리는 해 수)
    for year, gap in runs:
        # 평가 대상: 그해 선수 전원 (정답이 EA 평점이든 추정값이든). 채점은 두 가지로 따로 한다.
        test = long[long["snapshot_year"] == year]
        if gap == 0:
            test = test[test["horizon"] == 5]                      # 조건을 늦춘 평가는 5년 뒤만 본다
        else:
            test = test[test["horizon"] < 5]
        test = test[test["label_year"] <= SCOUT_YEAR]
        # 평가 선수의 과거 행도 학습에 쓴다. 그 행의 정답은 평가 시점 전에 이미 나와 있던 값이라
        # 실제로 쓸 수 있는 정보다. (평가 선수를 통째로 빼면 그해 뛰는 선수 대부분이 빠져 학습이 크게 줄어든다)
        train = long[long["label_year"] <= year - gap]
        if test.empty or train.empty:
            continue
        model = fit(train, GROWTH_FEATURES, "delta")
        out = test[["snapshot_year", "horizon", "age", "ea_label", "label", "current_est", "ea_overall_prev"]].copy()
        out["모델"] = test["current_est"] + model.predict(test[GROWTH_FEATURES])
        out["기준선: 현재 추정 그대로"] = test["current_est"]
        out["기준선: 현재 추정 + 나이·기간별 평균 변화"] = baseline_change(train, test)
        collected.append(out)
        print(f"  {year}년 평가 완료 ({'정답 공개 조건 늦춤, 5년 뒤만' if gap == 0 else '1~4년 뒤'}): "
              f"학습 {len(train):,}행, 평가 {len(test):,}행")

    every = pd.concat(collected)
    models = ["모델", "기준선: 현재 추정 그대로", "기준선: 현재 추정 + 나이·기간별 평균 변화"]
    # 채점 1: N년 뒤 EA 평점이 있는 선수만 (진짜 정답, 하지만 EA 평가를 계속 받은 선수만 남는다)
    # 채점 2: 하부 리그로 내려가 추정 정답을 쓰는 선수까지 전원 (모든 선수를 보지만 정답에 추정 오차가 섞인다)
    for truth, title in [("ea_label", "채점 1: EA 평점이 있는 선수만"), ("label", "채점 2: 추정 정답까지 포함한 전원")]:
        r = every[every[truth].notna()]
        for subset, sub in [("전체", r), ("21세 이하", r[r["age"] <= PROSPECT_AGE])]:
            rows = {}
            for h in HORIZONS:
                part = sub[sub["horizon"] == h]
                if part.empty:
                    continue
                years = sorted(part["snapshot_year"].unique())
                for m in models:
                    rows[(f"{h}년 뒤 (평가 {','.join(map(str, years))})", m)] = metrics(part[m], part[truth])
            show(f"[{title} | {subset}] 예측 기간별 성적", rows)

    # 80% 예측 범위: 예측 기간·나이대마다 (실제 EA 평점 - 예측)의 10%, 90% 지점
    interval = {}
    r = every[every["ea_label"].notna()]
    resid = r.assign(err=r["ea_label"] - r["모델"], band=age_band(r["age"]))
    print("\n[80% 예측 범위: 기간·나이대별 (실제 - 예측)의 10%~90% 지점, 점]")
    table = {}
    for h in HORIZONS:
        interval[str(h)] = {}
        for band in AGE_LABELS:
            q = resid[(resid["horizon"] == h) & (resid["band"] == band)]["err"]
            if len(q) < 30:                                        # 사례가 너무 적으면 바로 아래 기간 값을 넓혀 쓴다
                lo, hi = interval[str(h - 1)][band]
                lo, hi = lo * h / (h - 1), hi * h / (h - 1)
            else:
                lo, hi = float(q.quantile(0.1)), float(q.quantile(0.9))
            interval[str(h)][band] = [lo, hi]
            table.setdefault(band, {})[f"{h}년"] = f"{lo:+.1f}~{hi:+.1f}"
    print(pd.DataFrame(table).T.to_string())
    return interval


# ---------------------------------------------------------------------------
# 3. 최종 모델과 검색용 표
# ---------------------------------------------------------------------------
def train_final_and_score(ds, long, current_model, interval):
    print("\n" + "=" * 70)
    print(f"3. 최종 모델 학습과 {SCOUT_YEAR}년 7월 기준 전 선수 점수")
    print("=" * 70)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    growth = fit(long, GROWTH_FEATURES, "delta")
    current_model.save_model(models_dir / "ability_current.txt")
    growth.save_model(models_dir / "ability_growth.txt")
    (models_dir / "ability_interval.json").write_text(json.dumps(interval, ensure_ascii=False, indent=2))
    print(f"  성장 모델 학습 {len(long):,}행 (기준일 {long['snapshot_year'].min()}~{long['snapshot_year'].max()}년, "
          f"그중 추정 정답 {long['ea_label'].isna().sum():,}행)")

    now = ds[ds["snapshot_year"] == SCOUT_YEAR].copy()
    now["current_ability"] = np.clip(now["current_est"], 1, 99)
    bands = age_band(now["age"])
    for h in HORIZONS:
        future = np.clip(now["current_est"] + growth.predict(now.assign(horizon=h)[GROWTH_FEATURES]), 1, 99)
        now[f"future_{h}"] = future
        now[f"low_{h}"] = np.clip(future + bands.map(lambda b: interval[str(h)][b][0]), 1, 99)
        now[f"high_{h}"] = np.clip(future + bands.map(lambda b: interval[str(h)][b][1]), 1, 99)
        now[f"future_pct_age_{h}"] = now.groupby(np.floor(now["age"]))[f"future_{h}"].rank(pct=True) * 100
    now["current_pct"] = now["current_ability"].rank(pct=True) * 100                  # 전체 선수 중 백분위
    # 이전 판과 같은 이름도 남긴다 (3년 뒤 = 기본값)
    now["future_ability"], now["future_low"], now["future_high"] = now["future_3"], now["low_3"], now["high_3"]
    now["future_pct_age"] = now["future_pct_age_3"]
    now["growth"] = now["future_3"] - now["current_ability"]
    keep = (["player_id", "name", "age", "position", "main_position", "club_name", "league", "citizenship",
             "value_now", "games_1s", "starts_1s", "goals_1s", "assists_1s", "level_played_1s",
             "ea_overall_prev", "ea_potential_prev", "ability_now", "current_ability", "current_pct",
             "future_ability", "future_low", "future_high", "future_pct_age", "growth"]
            + [f"{c}_{h}" for h in HORIZONS for c in ["future", "low", "high", "future_pct_age"]])
    out_path = ROOT / "data" / "processed" / f"scout_{SCOUT_YEAR}.parquet"
    now[keep].to_parquet(out_path, index=False)
    print(f"  저장: {out_path} ({len(now):,}명)")

    rated = now[now["ability_now"].notna()]
    print(f"  확인: FC 26(2025년 9월) 평점이 있는 {len(rated):,}명에서 현재 기량 추정의 평균 오차 "
          f"{(rated['current_ability'] - rated['ability_now']).abs().mean():.2f}점")
    print("\n  21세 이하, 3년 뒤 예상 기량 상위 10명 (1~5년 뒤):")
    top = now[now["age"] <= PROSPECT_AGE].nlargest(10, "future_3")
    print(top[["name", "age", "club_name", "current_ability"] + [f"future_{h}" for h in HORIZONS]]
          .round(1).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1-1 입력 비교를 건너뛴다")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    ds = attach_ea(load_dataset(max_age=40))
    print(f"학습 테이블 {len(ds):,}행 (15~40세, 2012~2025년 기준일)\n")
    if not args.quick:
        compare_current_inputs(ds)
    ds["current_est"], current_model = current_estimates(ds)
    long = growth_table(ds)
    interval = evaluate_growth(long)
    train_final_and_score(ds, long, current_model, interval)


if __name__ == "__main__":
    main()
