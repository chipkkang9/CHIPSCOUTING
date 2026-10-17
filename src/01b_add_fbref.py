"""1b단계: FBref 세부 지표를 학습 테이블의 선수에게 연결한다.

FBref(5대 리그, 2017/18~2024/25 시즌)의 선수·시즌 기록을 90분당 지표로 바꾸고,
이름과 태어난 해로 Transfermarkt 선수 ID(player_id)를 찾아 붙인다.

시즌 "17/18"의 기록은 2018년 7월 1일 기준일의 "지난 시즌" 기록으로 쓴다.
(01_build_dataset.py의 completed_season과 같은 규칙)

원본: data/raw_stats/emrey3lmaz_top-5-league-football-player-stats-2017-2025/
실행:  python src/01b_add_fbref.py
출력:  data/processed/fbref_features.parquet   (player_id, snapshot_year)마다 한 행
"""
import glob
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FBREF = glob.glob(str(ROOT / "data" / "raw_stats" / "emrey3lmaz_*" / "*.csv"))[0]
PROFILES = ROOT / "data" / "raw_salimt" / "player_profiles.csv"
OUT = ROOT / "data" / "processed" / "fbref_features.parquet"

MIN_MINUTES = 450   # 이보다 적게 뛰면 90분당 지표가 몇 경기로 크게 흔들려서 쓰지 않는다

# FBref 원래 열 이름 -> 우리가 쓸 짧은 이름. 전부 "시즌 합계" 값이다.
COUNTS = {
    # 출전
    "Playing Time_Min": "minutes",
    "Playing Time_Starts": "starts",
    # 공격
    "Performance_Gls": "goals",
    "Performance_Ast": "assists",
    "Expected_xG": "xg",
    "Expected_npxG": "npxg",
    "Expected_xAG": "xag",                  # 기대 도움
    "Standard_Sh": "shots",
    "Standard_SoT": "shots_on_target",
    "KP_": "key_passes",
    "SCA_SCA": "shot_creating_actions",
    "Touches_Touches": "touches",
    "Touches_Att 3rd": "touches_att_third",
    "Touches_Att Pen": "touches_att_pen",
    "Take-Ons_Att": "take_ons",
    "Take-Ons_Succ": "take_ons_won",
    "Carries_Dis": "dispossessed",          # 상대에게 빼앗김
    "Carries_Mis": "miscontrols",           # 컨트롤 실수
    # 패스
    "1/3_": "passes_final_third",           # final third로 들어간 패스
    "PPA_": "passes_penalty_area",
    "Progression_PrgP": "progressive_passes",
    "Pass Types_TB": "through_balls",
    "Progression_PrgC": "progressive_carries",
    "Receiving_PrgR": "progressive_received",
    "Total_Cmp": "passes_completed",
    "Total_Att": "passes_attempted",
    "Long_Cmp": "long_completed",
    "Long_Att": "long_attempted",
    # 수비
    "Tackles_Tkl": "tackles",
    "Tackles_TklW": "tackles_won",
    "Challenges_Tkl": "dribblers_tackled",
    "Challenges_Att": "dribblers_challenged",
    "Int_": "interceptions",
    "Clr_": "clearances",
    "Blocks_Blocks": "blocks",
    "Performance_Recov": "recoveries",
    "Err_": "errors",
    # 경합, 파울
    "Aerial Duels_Won": "aerials_won",
    "Aerial Duels_Lost": "aerials_lost",
    "Performance_Fls": "fouls",
    "Performance_Fld": "fouled",
}
# 90분당 값으로 바꾸지 않는 열 (출전 시간 자체)
NOT_PER90 = ["minutes", "starts"]


def normalize_name(name):
    """이름 비교용: 악센트를 떼고, 소문자로, 글자와 공백만 남긴다. 'Højbjerg' -> 'hojbjerg'"""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    for a, b in [("ø", "o"), ("đ", "d"), ("ł", "l"), ("ß", "ss"), ("æ", "ae"), ("ı", "i")]:
        s = s.replace(a, b)
    return re.sub(r"[^a-z ]", "", s.replace("-", " ")).strip()


def load_fbref():
    """FBref 표를 읽어 (선수, 시즌)마다 한 행으로 합친다."""
    # 이 파일은 세미콜론으로 구분되고, 소수점이 쉼표다 ("20,5")
    raw = pd.read_csv(FBREF, sep=";", decimal=",", low_memory=False)
    raw = raw.rename(columns=COUNTS)
    raw["born"] = raw["born_"]
    # 시즌 1718 -> 2018년 7월 1일 기준일
    raw["snapshot_year"] = 2000 + raw["season"] % 100
    # 시즌 중에 5대 리그 안에서 팀을 옮긴 선수는 팀마다 한 줄씩 있어서 합친다
    keys = ["player", "born", "snapshot_year"]
    fb = raw.groupby(keys, as_index=False)[list(COUNTS.values())].sum(min_count=1)
    fb["fbref_league"] = raw.groupby(keys)["league"].last().values
    fb["fbref_team"] = raw.groupby(keys)["team"].last().values
    return fb


def per90_features(fb):
    """시즌 합계를 90분당 값과 성공률로 바꾼다."""
    out = fb[["player", "born", "snapshot_year", "fbref_league", "fbref_team", "minutes", "starts"]].copy()
    nineties = (fb["minutes"] / 90).where(fb["minutes"] >= MIN_MINUTES)   # 450분 미만이면 빈 값
    for col in COUNTS.values():
        if col not in NOT_PER90:
            out[f"fb_{col}_p90"] = fb[col] / nineties

    def rate(won, total, min_total=5):
        """성공률. 시도가 너무 적으면(5번 미만) 한두 번으로 100%가 되므로 빈 값."""
        return (fb[won] / fb[total].where(fb[total] >= min_total)).where(nineties.notna())

    out["fb_pass_pct"] = rate("passes_completed", "passes_attempted")
    out["fb_long_pass_pct"] = rate("long_completed", "long_attempted")
    out["fb_tackle_win_pct"] = rate("tackles_won", "tackles")
    out["fb_dribbler_tackle_pct"] = rate("dribblers_tackled", "dribblers_challenged")
    fb["aerials_total"] = fb["aerials_won"] + fb["aerials_lost"]
    out["fb_aerial_win_pct"] = rate("aerials_won", "aerials_total")
    out["fb_take_on_pct"] = rate("take_ons_won", "take_ons")
    out["fb_shot_on_target_pct"] = rate("shots_on_target", "shots")
    # 기대값 대비 결정력: 실제 골 - 기대 골 (90분당)
    out["fb_goals_minus_xg_p90"] = out["fb_goals_p90"] - out["fb_xg_p90"]
    out = out.rename(columns={"minutes": "fb_minutes", "starts": "fb_starts"})
    return out


def link_players(fb):
    """이름 + 태어난 해로 Transfermarkt player_id를 찾는다.

    1차: 정규화한 전체 이름 + 태어난 해가 정확히 한 명과 맞으면 연결.
    2차: 1차에서 못 찾았으면, 성(마지막 단어) + 태어난 해가 정확히 한 명과 맞으면 연결.
    후보가 여러 명이면 잘못 연결할 위험이 있어 연결하지 않는다.
    """
    tm = pd.read_csv(PROFILES, low_memory=False, usecols=["player_id", "player_name", "date_of_birth"])
    tm["name_key"] = tm["player_name"].str.replace(r"\s*\(\d+\)$", "", regex=True).map(normalize_name)
    tm["born"] = pd.to_datetime(tm["date_of_birth"], errors="coerce").dt.year
    tm["last_key"] = tm["name_key"].str.split().str[-1]

    fb = fb.copy()
    fb["name_key"] = fb["player"].map(normalize_name)
    fb["last_key"] = fb["name_key"].str.split().str[-1]

    def unique_lookup(key_cols):
        """key_cols가 같은 Transfermarkt 선수가 딱 한 명인 경우만 남긴 표."""
        g = tm.dropna(subset=["born"]).groupby(key_cols)["player_id"]
        counts = g.transform("size")
        return tm.loc[counts.index[counts == 1], key_cols + ["player_id"]]

    full = unique_lookup(["name_key", "born"])
    fb = fb.merge(full, on=["name_key", "born"], how="left")
    fb["link_method"] = np.where(fb["player_id"].notna(), "full_name", None)

    last = unique_lookup(["last_key", "born"]).rename(columns={"player_id": "pid_by_last"})
    fb = fb.merge(last, on=["last_key", "born"], how="left")
    rescue = fb["player_id"].isna() & fb["pid_by_last"].notna()
    fb.loc[rescue, "player_id"] = fb.loc[rescue, "pid_by_last"]
    fb.loc[rescue, "link_method"] = "last_name"
    return fb.drop(columns=["name_key", "last_key", "pid_by_last"])


def main():
    fb = per90_features(load_fbref())
    fb = link_players(fb)

    # 연결 결과 보고
    young = fb[(fb["snapshot_year"] - fb["born"] <= 22) & (fb["fb_minutes"] >= MIN_MINUTES)]
    print(f"FBref 선수·시즌 {len(fb):,}행")
    for label, part in [("전체", fb), ("21세 안팎·450분 이상", young)]:
        share = part["link_method"].value_counts(dropna=False, normalize=True).round(3).to_dict()
        print(f"  {label}: {len(part):,}행 | 연결 방법 비율 {share}")

    linked = fb.dropna(subset=["player_id"]).copy()
    linked["player_id"] = linked["player_id"].astype(int)
    # 한 Transfermarkt 선수에 FBref 두 명이 붙은 경우(동명이인 등)는 둘 다 버린다
    dup = linked.duplicated(["player_id", "snapshot_year"], keep=False)
    print(f"  같은 선수에 두 명이 연결돼 버린 행: {dup.sum()}")
    linked = linked[~dup]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    linked.to_parquet(OUT, index=False)
    print(f"\n저장: {OUT}  ({len(linked):,}행, {linked.shape[1]}열)")
    print(linked.groupby("snapshot_year").size().rename("연결된 선수·시즌").to_string())


if __name__ == "__main__":
    main()
