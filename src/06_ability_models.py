"""6단계: 기량 모델. "지금 얼마나 잘하나"와 "3년 뒤 얼마나 잘할까"를 1~99 점수로 예측한다.

몸값 대신 EA FC 평점(overall)을 기량의 기준으로 쓴다 (01c_add_ea.py).

  현재 기량 모델  입력: 기준일(7월 1일)까지의 기록   정답: 그해 9월 EA 평점 (ability_now)
  성장 모델       입력: 같은 기록 + 현재 기량 추정   정답: 3년 뒤 9월 EA 평점 (ability_future)

현재 기량 모델이 먼저 "지금 기량"을 추정하고, 성장 모델은 거기서 3년 동안 몇 점 오르내릴지를
맞힌다 (기량 수준이 아니라 변화폭을 맞힌다). 트리 모델은 학습 데이터의 최고점 근처에서 값을
깎는 성질이 있어서, 수준을 바로 맞히게 하면 최상위 유망주의 성장을 과소평가하기 때문이다.

성장 모델의 정답에는 구멍이 있다. 3년 뒤 EA가 평가하지 않는 리그로 내려간 선수는 정답이 없다.
그대로 두면 "살아남은 선수"만 배우게 되므로, 그런 선수는 3년 뒤 시점에 현재 기량 모델이 추정한
값을 정답 대신 쓴다 (아예 기록이 사라진 선수는 여전히 빠진다).

실행:  python src/06_ability_models.py
출력:  models/ability_current.txt, models/ability_growth.txt    최종 모델
       models/ability_interval.json                             3년 뒤 기량의 80% 범위 (나이대별 오차 분포)
       data/processed/scout_2025.parquet                        2025년 7월 기준 전 선수 점수 (검색용)
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (EA_FEATURES, FEATURES, HORIZON_YEARS, PROSPECT_AGE, ROOT, attach_ea,
                    load_dataset, rank_corr)

PARAMS = {
    "objective": "regression", "metric": "l1", "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "cat_smooth": 20, "seed": 0, "verbose": -1,
}
MAX_ROUNDS, PATIENCE = 3000, 100

VALUE_ONLY = ["rel_value_now", "age", "position", "main_position"]
CURRENT_FEATURES = FEATURES + EA_FEATURES            # 현재 기량 모델의 입력
GROWTH_FEATURES = FEATURES + ["current_est"]         # 성장 모델의 입력 (이전 EA 평점은 current_est 안에 녹아 있다)
FEATURE_SETS = {
    "몸값만": VALUE_ONLY,
    "경기 기록 + 몸값": FEATURES,
    "경기 기록 + 몸값 + 이전 EA 평점": CURRENT_FEATURES,
}
FIRST_ESTIMATE_YEAR = 2015          # EA 평점이 2014년부터라 그 다음 해부터 "과거로 배운" 추정이 가능하다
FUTURE_EVAL_YEARS = [2019, 2020, 2022]   # 현재 기량 추정이 2015년부터라 2019년부터 평가 가능. 2021은 3년 뒤(2024) 정답이 없다
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


def top_hit(df, pred_col, truth_col, frac=0.05):
    """해마다 예측 상위 5% 중 실제 상위 5%였던 비율의 평균."""
    hits = []
    for _, g in df.groupby("snapshot_year"):
        k = max(1, int(np.ceil(len(g) * frac)))
        hits.append(len(set(g[pred_col].nlargest(k).index) & set(g[truth_col].nlargest(k).index)) / k)
    return float(np.mean(hits))


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
# 2. 성장 모델 (3년 뒤 기량)
# ---------------------------------------------------------------------------
def future_labels(ds):
    """3년 뒤 기량 정답: EA 평점이 있으면 그것, 없으면 3년 뒤 시점의 현재 기량 추정값."""
    later = ds[["player_id", "snapshot_year", "current_est"]].copy()
    later["snapshot_year"] -= HORIZON_YEARS                    # 3년 뒤 행을 지금 행에 맞춘다
    merged = ds[["player_id", "snapshot_year", "ability_future"]].merge(
        later, on=["player_id", "snapshot_year"], how="left")
    merged.index = ds.index
    label = merged["ability_future"].fillna(merged["current_est"])
    source = np.where(merged["ability_future"].notna(), "EA", np.where(merged["current_est"].notna(), "추정", ""))
    return label, pd.Series(source, index=ds.index)


def evaluate_future(ds):
    print("\n" + "=" * 70)
    print("2. 성장 모델: 7월 1일까지의 기록으로 3년 뒤 9월 EA 평점 맞히기")
    print("=" * 70)
    collected = []
    for year in FUTURE_EVAL_YEARS:
        test = ds[(ds["snapshot_year"] == year) & ds["ability_future"].notna() & ds["current_est"].notna()]
        # 3년 뒤 정답이 year년 7월 1일 전에 나와 있어야 한다 -> 기준일이 year-4년 이하인 행만 학습
        train = ds[(ds["snapshot_year"] <= year - HORIZON_YEARS - 1) & ds["label"].notna()
                   & ds["current_est"].notna() & ~ds["player_id"].isin(test["player_id"])]
        out = test[["snapshot_year", "age", "ability_future", "current_est", "ea_overall_prev", "ea_potential_prev"]].copy()

        level = fit(train, GROWTH_FEATURES, "label")                                 # 수준을 바로 맞히기
        out["모델: 3년 뒤 수준을 바로 예측"] = level.predict(test[GROWTH_FEATURES])
        growth = fit(train.assign(delta=train["label"] - train["current_est"]), GROWTH_FEATURES, "delta")
        out["모델: 현재 추정 + 변화폭 예측"] = test["current_est"] + growth.predict(test[GROWTH_FEATURES])

        change = (train["label"] - train["current_est"]).groupby(np.floor(train["age"])).mean()
        out["기준선: 현재 추정 그대로"] = test["current_est"]
        out["기준선: 현재 추정 + 나이별 평균 변화"] = test["current_est"] + np.floor(test["age"]).map(change).fillna(0)
        out["기준선: 이전 EA 평점 그대로"] = test["ea_overall_prev"]
        out["기준선: EA 잠재력"] = test["ea_potential_prev"]
        collected.append(out)
        print(f"  {year}년 평가 완료 (학습 {len(train):,}행, 평가 {len(test):,}행)")

    r = pd.concat(collected)
    models = [c for c in r.columns if c.startswith(("모델", "기준선"))]
    with_ea = r[r["ea_overall_prev"].notna()]
    for title, part in [("이전 EA 평점이 있는 선수", with_ea), ("이전 EA 평점이 없는 선수", r[r["ea_overall_prev"].isna()])]:
        for subset, sub in [("전체", part), ("21세 이하", part[part["age"] <= PROSPECT_AGE])]:
            rows = {m: metrics(sub[m], sub["ability_future"]) for m in models if sub[m].notna().all()}
            show(f"[{title} | {subset}]", rows)
    young = with_ea[with_ea["age"] <= PROSPECT_AGE]
    print("\n[21세 이하 (이전 EA 평점 있음), 3년 뒤 기량 상위 5% 적중률 (연도별 평균)]")
    for m in models:
        print(f"  {m}: {top_hit(young, m, 'ability_future'):.3f}")

    # 최상위권에서 깎는지 확인: 실제 3년 뒤 기량이 85 이상인 선수의 평균 오차(예측 - 실제)
    top = r[r["ability_future"] >= 85]
    print(f"\n[실제 3년 뒤 기량 85 이상인 {len(top)}명: 평균 (예측 - 실제), 음수면 과소평가]")
    for m in ["모델: 3년 뒤 수준을 바로 예측", "모델: 현재 추정 + 변화폭 예측"]:
        print(f"  {m}: {(top[m] - top['ability_future']).mean():+.2f}점")

    # 80% 예측 범위: 나이대별로 (실제 - 예측)의 10%, 90% 지점을 잰다
    main = "모델: 현재 추정 + 변화폭 예측"
    resid = (r["ability_future"] - r[main]).groupby(age_band(r["age"]))
    interval = {band: [float(q.quantile(0.1)), float(q.quantile(0.9))] for band, q in resid}
    print("\n[80% 예측 범위: 나이대별 (실제 - 예측)의 10%~90% 지점]")
    for band in AGE_LABELS:
        lo, hi = interval[band]
        print(f"  {band}: {lo:+.1f} ~ {hi:+.1f}점")
    return interval


# ---------------------------------------------------------------------------
# 3. 최종 모델과 검색용 표
# ---------------------------------------------------------------------------
def train_final_and_score(ds, current_model, interval):
    print("\n" + "=" * 70)
    print(f"3. 최종 모델 학습과 {SCOUT_YEAR}년 7월 기준 전 선수 점수")
    print("=" * 70)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    train = ds[ds["label"].notna() & ds["current_est"].notna()]
    growth = fit(train.assign(delta=train["label"] - train["current_est"]), GROWTH_FEATURES, "delta")
    current_model.save_model(models_dir / "ability_current.txt")
    growth.save_model(models_dir / "ability_growth.txt")
    (models_dir / "ability_interval.json").write_text(json.dumps(interval, ensure_ascii=False, indent=2))
    print(f"  성장 모델 학습 {len(train):,}행 (기준일 {train['snapshot_year'].min()}~{train['snapshot_year'].max()}년, "
          f"그중 추정 정답 {(train['label_source'] == '추정').sum():,}행)")

    now = ds[ds["snapshot_year"] == SCOUT_YEAR].copy()
    now["current_ability"] = np.clip(now["current_est"], 1, 99)
    now["future_ability"] = np.clip(now["current_est"] + growth.predict(now[GROWTH_FEATURES]), 1, 99)
    bands = age_band(now["age"])
    now["future_low"] = np.clip(now["future_ability"] + bands.map(lambda b: interval[b][0]), 1, 99)
    now["future_high"] = np.clip(now["future_ability"] + bands.map(lambda b: interval[b][1]), 1, 99)
    now["growth"] = now["future_ability"] - now["current_ability"]
    now["current_pct"] = now["current_ability"].rank(pct=True) * 100                  # 전체 선수 중 백분위
    now["future_pct_age"] = now.groupby(np.floor(now["age"]))["future_ability"].rank(pct=True) * 100
    keep = ["player_id", "name", "age", "position", "main_position", "club_name", "league", "citizenship",
            "value_now", "games_1s", "starts_1s", "goals_1s", "assists_1s", "level_played_1s",
            "ea_overall_prev", "ea_potential_prev", "ability_now",
            "current_ability", "future_ability", "future_low", "future_high", "growth",
            "current_pct", "future_pct_age"]
    out_path = ROOT / "data" / "processed" / f"scout_{SCOUT_YEAR}.parquet"
    now[keep].to_parquet(out_path, index=False)
    print(f"  저장: {out_path} ({len(now):,}명)")

    rated = now[now["ability_now"].notna()]
    print(f"  확인: FC 26(2025년 9월) 평점이 있는 {len(rated):,}명에서 현재 기량 추정의 평균 오차 "
          f"{(rated['current_ability'] - rated['ability_now']).abs().mean():.2f}점")
    print("\n  21세 이하, 3년 뒤 예상 기량 상위 10명:")
    top = now[now["age"] <= PROSPECT_AGE].nlargest(10, "future_ability")
    print(top[["name", "age", "club_name", "current_ability", "future_ability", "future_low", "future_high"]]
          .round(1).to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    ds = attach_ea(load_dataset(max_age=40))
    print(f"학습 테이블 {len(ds):,}행 (15~40세, 2012~2025년 기준일)\n")
    compare_current_inputs(ds)
    ds["current_est"], current_model = current_estimates(ds)
    ds["label"], ds["label_source"] = future_labels(ds)
    interval = evaluate_future(ds)
    train_final_and_score(ds, current_model, interval)


if __name__ == "__main__":
    main()
