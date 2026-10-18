"""4단계: FBref 세부 지표가 예측을 얼마나 낫게 하는지 비교한다.

두 모델을 똑같은 방식으로 학습시킨다.
  A: 기본 feature만 (03_train_gbdt.py와 같은 44개)
  B: 기본 feature + FBref 세부 지표 40개

그리고 학습에 쓰는 선수의 범위를 두 가지로 해 본다.
  21세 이하로 학습: 유망주만으로 배운다. 세부 지표가 있는 행이 약 430개뿐이다.
  25세 이하로 학습: 22~25세 선수까지 배운다. 세부 지표 행이 약 3,000개로 는다.
                    (나이를 입력으로 주므로, 모델은 나이에 따른 차이도 같이 배운다)

채점은 항상 같은 선수들로 한다: 세부 지표가 있는 21세 이하 유망주 (2018~2022년 기준일).
인원이 적어서 검증을 세 겹으로 한다.
  1. 같은 선수들로 네 모델을 채점해 차이를 본다.
  2. 선수를 다시 뽑아 계산하는 bootstrap으로 차이의 오차 범위를 낸다.
  3. 순위가 크게 바뀐 선수를 한 명씩 보여준다. 어떤 지표 때문이었고 실제로 어떻게 됐나.

실행:  python src/04_compare_fbref.py
"""
import importlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import FBREF_FEATURES, FEATURES, PROSPECT_AGE, TARGET, load_dataset, rank_corr

PARAMS = importlib.import_module("03_train_gbdt").PARAMS   # 03과 같은 설정을 쓴다
N_TREES = 400            # 03에서 나무 276~624개에서 멈췄다. 비교를 단순하게 하려고 고정한다
N_FOLDS = 5
N_BOOTSTRAP = 1000
SEED = 0
WIDE_AGE = 25
MODELS = {"A": FEATURES, "B": FEATURES + FBREF_FEATURES}


def cross_validate(labeled, tag):
    """선수 단위로 5등분해서, 한 조각씩 빼고 학습하고 뺀 조각을 예측한다 (out-of-fold 예측).

    같은 선수의 다른 해 행이 학습과 예측에 동시에 들어가지 않게 선수 단위로 나눈다.
    모델 B는 행마다 "각 세부 지표가 예측을 얼마나 움직였나"(기여도)도 함께 돌려준다.
    """
    oof = pd.DataFrame(index=labeled.index, dtype=float)
    contrib = pd.DataFrame(index=labeled.index, columns=FBREF_FEATURES, dtype=float)
    folds = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(folds.split(labeled, groups=labeled["player_id"]), start=1):
        train, test = labeled.iloc[tr], labeled.iloc[te]
        for name, feats in MODELS.items():
            model = lgb.train(PARAMS, lgb.Dataset(train[feats], train[TARGET]), num_boost_round=N_TREES)
            oof.loc[test.index, f"{name}_{tag}"] = model.predict(test[feats])
            if name == "B":
                parts = model.predict(test[feats], pred_contrib=True)[:, :-1]   # 마지막 열은 기본값
                contrib.loc[test.index] = parts[:, len(FEATURES):]
        print(f"  [{tag}] {k}/{N_FOLDS}번째 조각 완료")
    return oof, contrib


def scores(df, pred):
    """한 묶음의 점수: 순위상관, 평균 오차, 상위 10% 적중률."""
    k = max(1, int(round(len(df) * 0.10)))
    picked = set(pred.nlargest(k).index)
    actual = set(df[TARGET].nlargest(k).index)
    return {
        "순위상관(성장)": rank_corr(pred, df[TARGET]),
        "평균 몇 배 틀림": 10 ** (pred - df[TARGET]).abs().mean(),
        "상위10% 적중률": len(picked & actual) / k,
    }


def bootstrap_difference(df, col_new, col_old, rng):
    """선수를 복원 추출로 다시 뽑아 (새 모델 - 옛 모델) 순위상관 차이를 1,000번 계산한다."""
    players = df["player_id"].unique()
    rows_by_player = df.groupby("player_id").indices
    diffs = []
    for _ in range(N_BOOTSTRAP):
        sample = rng.choice(players, size=len(players), replace=True)
        part = df.iloc[np.concatenate([rows_by_player[p] for p in sample])]
        diffs.append(rank_corr(part[col_new], part[TARGET]) - rank_corr(part[col_old], part[TARGET]))
    diffs = np.array(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), (diffs > 0).mean()


def show_movers(detail, contrib, col_new, col_old):
    """새 모델에서 순위가 크게 바뀐 선수를 보여준다."""
    d = detail.copy()
    by_year = d.groupby("snapshot_year")
    d["rank_old"] = by_year[col_old].rank(ascending=False)
    d["rank_new"] = by_year[col_new].rank(ascending=False)
    d["rank_actual"] = by_year[TARGET].rank(ascending=False)
    d["n_year"] = by_year[col_old].transform("size")
    d["moved"] = d["rank_old"] - d["rank_new"]          # 양수면 새 모델에서 순위가 올라감

    def top_reasons(i, n=3):
        c = contrib.loc[i].astype(float)
        c = c[c.abs() > 0.005].sort_values(key=abs, ascending=False).head(n)
        return ", ".join(f"{name.removeprefix('fb_')} {v:+.2f}" for name, v in c.items())

    cols = ["snapshot_year", "name", "main_position", "club_name", "value_now", "value_future",
            "rank_old", "rank_new", "rank_actual", "n_year"]
    for title, part in [("크게 오른 선수", d.nlargest(8, "moved")), ("크게 내린 선수", d.nsmallest(8, "moved"))]:
        print(f"\n[{title}]  순위는 그 해 채점 대상 안에서의 등수 (1이 성장 예측 최고)")
        table = part[cols].copy()
        table["세부 지표 기여 상위 3개"] = [top_reasons(i) for i in part.index]
        table[["value_now", "value_future"]] = (table[["value_now", "value_future"]] / 1e6).round(1)
        print(table.rename(columns={"value_now": "몸값(백만)", "value_future": "3년뒤(백만)"}).to_string(index=False))

    closer = (d["rank_new"] - d["rank_actual"]).abs() < (d["rank_old"] - d["rank_actual"]).abs()
    moved = d["moved"].abs() >= 5
    print(f"\n순위가 5계단 이상 바뀐 {moved.sum()}명 중 새 모델이 실제 순위에 더 가까워진 선수: "
          f"{(closer & moved).sum()}명 ({(closer & moved).sum() / max(moved.sum(), 1):.0%})")


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)

    wide = load_dataset(with_fbref=True, max_age=WIDE_AGE)
    wide = wide[wide["has_label"] == 1]
    young = wide[wide["age"] <= PROSPECT_AGE]
    has_fb = wide["fb_touches_p90"].notna()
    print(f"정답 있는 행: 21세 이하 {len(young):,}개 (세부 지표 {young['fb_touches_p90'].notna().sum()}개) | "
          f"25세 이하 {len(wide):,}개 (세부 지표 {has_fb.sum():,}개)\n")

    oof21, _ = cross_validate(young, "21")
    oof25, contrib25 = cross_validate(wide, "25")

    # 채점 대상: 세부 지표가 있는 21세 이하 유망주
    detail = young[young["fb_touches_p90"].notna()].join(oof21).join(oof25)
    print(f"\n채점 대상: 세부 지표가 있는 유망주 {len(detail)}행 ({detail['player_id'].nunique()}명)")

    # --- 검증 1 ---
    print("\n=== 검증 1. 같은 선수로 채점 ===")
    names = {"A_21": "A, 21세 이하로 학습", "B_21": "B, 21세 이하로 학습",
             "A_25": "A, 25세 이하로 학습", "B_25": "B, 25세 이하로 학습"}
    print(pd.DataFrame({label: scores(detail, detail[col]) for col, label in names.items()}).T.round(3).to_string())
    print("\n포지션별 순위상관(성장):")
    by_pos = {pos: {"선수 수": len(g), **{c: rank_corr(g[c], g[TARGET]) for c in names}}
              for pos, g in detail.groupby("main_position", observed=True) if len(g) >= 20}
    print(pd.DataFrame(by_pos).T.round(3).to_string())

    # --- 검증 2 ---
    print("\n=== 검증 2. 순위상관 차이의 95% 범위 (bootstrap 1,000번) ===")
    flat = detail.reset_index(drop=True)
    for new, old, meaning in [("B_21", "A_21", "21세 이하로 학습할 때 세부 지표의 효과"),
                              ("B_25", "A_25", "25세 이하로 학습할 때 세부 지표의 효과"),
                              ("A_25", "A_21", "학습 범위를 넓힌 것만의 효과 (세부 지표 없이)")]:
        low, high, share = bootstrap_difference(flat, new, old, np.random.default_rng(SEED))
        print(f"{new} - {old}: {low:+.3f} ~ {high:+.3f} | 새 쪽이 나은 비율 {share:.1%}  ({meaning})")
    print("범위가 0을 포함하면 '낫다'고 말할 수 없다.")

    # --- 검증 3 ---
    print("\n=== 검증 3. 25세 이하로 학습할 때, B에서 순위가 크게 바뀐 선수 (A_25 대비) ===")
    show_movers(detail, contrib25, "B_25", "A_25")

    print("\n=== 25세 이하로 학습한 B에서 예측을 가장 크게 움직인 세부 지표 (채점 대상 기준) ===")
    print(contrib25.loc[detail.index].astype(float).abs().mean().sort_values(ascending=False).head(12).round(4).to_string())


if __name__ == "__main__":
    main()
