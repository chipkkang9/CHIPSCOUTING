"""4단계: 트리 모델(LightGBM) 학습.

선수의 feature를 보고 "같은 해 유망주 평균 대비 3년간 성장"을 예측하도록 학습시킨다.
작은 결정 나무를 하나씩 추가하면서, 앞의 나무들이 틀린 만큼을 다음 나무가
메우는 방식(gradient boosting)이다.

평가 연도(2016~2022)마다 그 시점까지 정답이 나온 과거 데이터로만 새로 학습시키고,
그 해를 맞혀 본다 (walk-forward).

실행:  python src/03_train_gbdt.py
출력:  models/gbdt.txt                  마지막 평가 연도(2022)에 쓴 모델
       outputs/gbdt_learning_curve.png  학습 곡선 (2022년 평가분)
       outputs/gbdt_importance.png      feature 중요도 (평가 연도 평균)
"""
import argparse
import importlib
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")   # 창을 띄우지 않고 그림을 파일로만 저장
import matplotlib.pyplot as plt
import pandas as pd

from common import (FEATURES, ROOT, TARGET, evaluate, load_dataset, split_last_year,
                    summarize, walk_forward)

baselines = importlib.import_module("02_baselines")   # 파일 이름이 숫자로 시작해서 이렇게 불러온다

PARAMS = {
    "objective": "regression",   # 오차 제곱을 줄이는 방향으로 학습
    "metric": "l1",              # 화면에 찍는 점수는 평균 절대 오차 (로그 단위)
    "learning_rate": 0.03,       # 나무 하나가 고치는 폭. 작을수록 천천히, 꼼꼼히 배운다
    "num_leaves": 31,            # 나무 하나의 잎(선수를 나눈 칸) 최대 개수
    "min_data_in_leaf": 40,      # 한 칸에 최소 40명. 소수 사례에 끼워 맞추는 것을 막는다
    "feature_fraction": 0.8,     # 나무마다 feature의 80%만 무작위로 사용
    "bagging_fraction": 0.8,     # 나무마다 학습 행의 80%만 무작위로 사용
    "bagging_freq": 1,
    "lambda_l2": 1.0,            # 예측값이 극단으로 가지 않게 누르는 정도
    "cat_smooth": 20,            # 범주형(국적, 리그 등)에서 사례가 적은 값을 덜 믿게 한다
    "seed": 0,
    "verbose": -1,
}
MAX_ROUNDS = 3000    # 나무를 최대 몇 개까지 추가할지
PATIENCE = 100       # 멈춤 판단용 점수가 이만큼 연속으로 안 좋아지면 멈춘다
LOG_EVERY = 50


def fit(train):
    """한 평가 연도에 쓸 모델을 학습시킨다. (모델, 학습 곡선 기록, 나무 개수)를 돌려준다."""
    # 1) 학습 데이터의 마지막 해를 떼어, 나무를 몇 개까지 쌓을지 정한다
    fit_part, stop_part = split_last_year(train)
    d_fit = lgb.Dataset(fit_part[FEATURES], fit_part[TARGET])
    d_stop = lgb.Dataset(stop_part[FEATURES], stop_part[TARGET], reference=d_fit)
    history = {}
    probe = lgb.train(
        PARAMS, d_fit, num_boost_round=MAX_ROUNDS,
        valid_sets=[d_fit, d_stop], valid_names=["train", "val"],
        callbacks=[
            lgb.log_evaluation(LOG_EVERY),          # 50개마다 점수 출력
            lgb.record_evaluation(history),         # 그래프용으로 점수 기록
            lgb.early_stopping(PATIENCE, first_metric_only=True, verbose=False),
        ],
    )
    n_trees = probe.best_iteration
    # 2) 그 개수로, 떼어 놨던 마지막 해까지 포함해 다시 학습시킨다
    model = lgb.train(PARAMS, lgb.Dataset(train[FEATURES], train[TARGET]), num_boost_round=n_trees)
    return model, history, n_trees


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT))
    out_dir = Path(ap.parse_args().out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    print(f"대상 선수 {len(ds):,}행 | feature {len(FEATURES)}개\n")

    results, base_results, importances = {}, {}, []
    for year, train, test in walk_forward(ds):
        print(f"--- {year}년 평가: 학습 {train['snapshot_year'].min()}~{train['snapshot_year'].max()}년 "
              f"{len(train):,}행 -> 평가 {len(test):,}행 ---")
        model, history, n_trees = fit(train)
        print(f"나무 {n_trees}개에서 멈춤\n")

        pred = pd.Series(model.predict(test[FEATURES]), index=test.index)
        results[year] = evaluate(test, pred)
        base_results[year] = evaluate(test, baselines.predict_age_value_growth(train, test))
        importances.append(pd.Series(model.feature_importance(importance_type="gain"), index=FEATURES))

    # --- 채점: 기준선 3과 나란히 ---
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print("=== 기준선 3 (나이x몸값 구간별 평균 성장) ===")
    print(summarize(base_results).to_string())
    print("\n=== 트리 모델 ===")
    print(summarize(results).to_string())

    # --- feature 중요도 (연도별 비율의 평균) ---
    importance = pd.concat([imp / imp.sum() * 100 for imp in importances], axis=1).mean(axis=1)
    importance = importance.sort_values(ascending=False)
    print("\n=== feature 중요도 (전체 기여의 %, 평가 연도 평균) ===")
    print(importance.head(15).round(1).to_string())

    # --- 저장: 마지막 평가 연도의 모델과 그림 ---
    model.save_model(out_dir / "models" / "gbdt.txt")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train"]["l1"], label="train")
    ax.plot(history["val"]["l1"], label="val (last training year)")
    ax.axvline(n_trees, color="gray", linestyle="--", label=f"best ({n_trees})")
    ax.set_xlabel("number of trees")
    ax.set_ylabel("mean absolute error (log10)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "outputs" / "gbdt_learning_curve.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 6))
    importance.head(20)[::-1].plot.barh(ax=ax)
    ax.set_xlabel("share of total gain (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "outputs" / "gbdt_importance.png", dpi=150)
    print(f"\n저장: {out_dir / 'models' / 'gbdt.txt'}, {out_dir / 'outputs'}/gbdt_*.png")


if __name__ == "__main__":
    main()
