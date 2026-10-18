"""5단계: 포지션별 모델에 포지션별 세부 지표를 넣으면 나아지는지 비교한다.

04에서는 모든 포지션을 한 모델에 넣고 세부 지표 40개를 전부 줬다. 그랬더니 공격수가
"공중볼을 못 이긴다"는 이유로 순위가 떨어지는 일이 생겼다. 그래서 이번에는
  - 골키퍼를 빼고,
  - 공격수, 미드필더, 수비수마다 따로 모델을 학습시키고,
  - 포지션마다 의미 있는 세부 지표 14개씩만 준다.

비교하는 모델 세 가지 (전부 25세 이하로 학습, 선수 단위 5등분 교차검증):
  통합 A    : 포지션 구분 없이 하나의 모델, 기본 feature만 (04의 A_25와 같은 방식)
  포지션별 A: 포지션마다 따로 학습, 기본 feature만
  포지션별 B: 포지션마다 따로 학습, 기본 feature + 그 포지션의 세부 지표

채점은 세부 지표가 있는 21세 이하 유망주(골키퍼 제외)로만 한다.

실행:  python src/05_position_models.py
"""
import importlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import FEATURES, PROSPECT_AGE, TARGET, load_dataset, rank_corr

compare = importlib.import_module("04_compare_fbref")    # 채점, bootstrap 도구를 같이 쓴다
PARAMS = compare.PARAMS
N_TREES = compare.N_TREES
N_FOLDS = compare.N_FOLDS
SEED = compare.SEED
WIDE_AGE = compare.WIDE_AGE

# 포지션마다 주는 세부 지표 (FBref, 90분당 값과 성공률). 처음 요청한 지표 목록을 포지션에 맞게 나눴다.
POSITION_FEATURES = {
    "Attack": [       # 공격수: 골, 도움, 유효슈팅, xG에 무게
        "fb_goals_p90", "fb_assists_p90", "fb_xg_p90", "fb_npxg_p90", "fb_xag_p90",
        "fb_shots_on_target_p90", "fb_shot_on_target_pct", "fb_goals_minus_xg_p90",
        "fb_touches_p90", "fb_touches_att_pen_p90", "fb_take_ons_p90", "fb_take_on_pct",
        "fb_dispossessed_p90", "fb_fouled_p90",
    ],
    "Midfield": [     # 미드필더: 공격과 수비 둘 다, 기대도움에 무게
        "fb_xag_p90", "fb_key_passes_p90", "fb_passes_final_third_p90", "fb_progressive_passes_p90",
        "fb_long_completed_p90", "fb_long_pass_pct", "fb_touches_p90", "fb_xg_p90",
        "fb_tackles_p90", "fb_tackle_win_pct", "fb_interceptions_p90", "fb_recoveries_p90",
        "fb_dispossessed_p90", "fb_fouls_p90",
    ],
    "Defender": [     # 수비수: 태클, 가로채기, 공중볼, 클리어, 리커버리에 무게
        "fb_tackles_p90", "fb_tackle_win_pct", "fb_dribbler_tackle_pct", "fb_interceptions_p90",
        "fb_clearances_p90", "fb_blocks_p90", "fb_recoveries_p90", "fb_aerials_won_p90",
        "fb_aerial_win_pct", "fb_long_completed_p90", "fb_long_pass_pct", "fb_progressive_passes_p90",
        "fb_errors_p90", "fb_fouls_p90",
    ],
}
POSITIONS = list(POSITION_FEATURES)
POSITION_KO = {"Attack": "공격수", "Midfield": "미드필더", "Defender": "수비수"}


def train_predict(train, test, feats):
    model = lgb.train(PARAMS, lgb.Dataset(train[feats], train[TARGET]), num_boost_round=N_TREES)
    return model, model.predict(test[feats])


def cross_validate(labeled):
    """선수 단위 5등분 교차검증으로 세 모델의 out-of-fold 예측과 B의 기여도를 만든다."""
    oof = pd.DataFrame(index=labeled.index, columns=["pooled_A", "pos_A", "pos_B"], dtype=float)
    contrib = {}                     # 행 번호 -> {세부 지표: 기여도}
    folds = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(folds.split(labeled, groups=labeled["player_id"]), start=1):
        train, test = labeled.iloc[tr], labeled.iloc[te]
        # 통합 A: 포지션 구분 없이 한 모델
        _, pred = train_predict(train, test, FEATURES)
        oof.loc[test.index, "pooled_A"] = pred
        # 포지션별 A, B: 같은 포지션 선수끼리만 학습하고 예측한다
        for pos, detail_feats in POSITION_FEATURES.items():
            tr_pos = train[train["main_position"] == pos]
            te_pos = test[test["main_position"] == pos]
            _, pred = train_predict(tr_pos, te_pos, FEATURES)
            oof.loc[te_pos.index, "pos_A"] = pred
            feats_b = FEATURES + detail_feats
            model, pred = train_predict(tr_pos, te_pos, feats_b)
            oof.loc[te_pos.index, "pos_B"] = pred
            parts = model.predict(te_pos[feats_b], pred_contrib=True)[:, len(FEATURES):-1]
            for i, row in zip(te_pos.index, parts):
                contrib[i] = dict(zip(detail_feats, row))
        print(f"  {k}/{N_FOLDS}번째 조각 완료")
    return oof, contrib


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)

    wide = load_dataset(with_fbref=True, max_age=WIDE_AGE)
    wide = wide[(wide["has_label"] == 1) & wide["main_position"].isin(POSITIONS)]
    has_fb = wide["fb_touches_p90"].notna()
    print(f"정답 있는 25세 이하 필드 플레이어 {len(wide):,}행 (세부 지표 {has_fb.sum():,}행)")
    print(wide[has_fb].groupby("main_position", observed=True).size().rename("포지션별 세부 지표 행").to_string(), "\n")

    oof, contrib = cross_validate(wide)
    wide = wide.join(oof)
    detail = wide[(wide["age"] <= PROSPECT_AGE) & wide["fb_touches_p90"].notna()]
    print(f"\n채점 대상: 세부 지표가 있는 21세 이하 필드 플레이어 {len(detail)}행 ({detail['player_id'].nunique()}명)")

    # --- 검증 1 ---
    print("\n=== 검증 1. 같은 선수로 채점 ===")
    names = {"pooled_A": "통합 A (기본 feature)", "pos_A": "포지션별 A (기본 feature)",
             "pos_B": "포지션별 B (기본 + 포지션 세부 지표)"}
    print("[전체]")
    print(pd.DataFrame({label: compare.scores(detail, detail[c]) for c, label in names.items()}).T.round(3).to_string())
    print("\n[포지션별 순위상관(성장)]")
    by_pos = {POSITION_KO[pos]: {"선수 수": len(g), **{c: rank_corr(g[c], g[TARGET]) for c in names}}
              for pos, g in detail.groupby("main_position", observed=True)}
    print(pd.DataFrame(by_pos).T.round(3).to_string())

    # --- 검증 2 ---
    print("\n=== 검증 2. 순위상관 차이의 95% 범위 (bootstrap 1,000번) ===")
    for part_name, part in [("전체", detail)] + [(POSITION_KO[p], g) for p, g in detail.groupby("main_position", observed=True)]:
        flat = part.reset_index(drop=True)
        for new, old, meaning in [("pos_B", "pos_A", "세부 지표의 효과"), ("pos_A", "pooled_A", "포지션을 나눈 효과")]:
            low, high, share = compare.bootstrap_difference(flat, new, old, np.random.default_rng(SEED))
            print(f"{part_name:5s} {new} - {old}: {low:+.3f} ~ {high:+.3f} | 새 쪽이 나은 비율 {share:5.1%}  ({meaning})")

    # --- 검증 3 ---
    print("\n=== 검증 3. 포지션별 B에서 순위가 크게 바뀐 선수 (포지션별 A 대비) ===")
    d = detail.copy()
    by_year = d.groupby("snapshot_year")
    d["rank_A"] = by_year["pos_A"].rank(ascending=False)
    d["rank_B"] = by_year["pos_B"].rank(ascending=False)
    d["rank_actual"] = by_year[TARGET].rank(ascending=False)
    d["n_year"] = by_year["pos_A"].transform("size")
    d["moved"] = d["rank_A"] - d["rank_B"]

    def top_reasons(i, n=3):
        c = pd.Series(contrib[i])
        c = c[c.abs() > 0.005].sort_values(key=abs, ascending=False).head(n)
        return ", ".join(f"{k.removeprefix('fb_')} {v:+.2f}" for k, v in c.items())

    cols = ["snapshot_year", "name", "main_position", "club_name", "value_now", "value_future",
            "rank_A", "rank_B", "rank_actual", "n_year"]
    for title, part in [("크게 오른 선수", d.nlargest(8, "moved")), ("크게 내린 선수", d.nsmallest(8, "moved"))]:
        print(f"\n[{title}]  순위는 그 해 채점 대상 안에서의 등수 (1이 성장 예측 최고)")
        table = part[cols].copy()
        table["세부 지표 기여 상위 3개"] = [top_reasons(i) for i in part.index]
        table[["value_now", "value_future"]] = (table[["value_now", "value_future"]] / 1e6).round(1)
        print(table.rename(columns={"value_now": "몸값(백만)", "value_future": "3년뒤(백만)"}).to_string(index=False))
    closer = (d["rank_B"] - d["rank_actual"]).abs() < (d["rank_A"] - d["rank_actual"]).abs()
    moved = d["moved"].abs() >= 5
    print(f"\n순위가 5계단 이상 바뀐 {moved.sum()}명 중 B가 실제 순위에 더 가까워진 선수: "
          f"{(closer & moved).sum()}명 ({(closer & moved).sum() / max(moved.sum(), 1):.0%})")

    print("\n=== 포지션별로 예측을 가장 크게 움직인 세부 지표 (채점 대상 기준, 기여 절댓값 평균) ===")
    for pos, g in detail.groupby("main_position", observed=True):
        c = pd.DataFrame([contrib[i] for i in g.index]).abs().mean().sort_values(ascending=False).head(6)
        print(f"{POSITION_KO[pos]}: " + ", ".join(f"{k.removeprefix('fb_')} {v:.4f}" for k, v in c.items()))


if __name__ == "__main__":
    main()
