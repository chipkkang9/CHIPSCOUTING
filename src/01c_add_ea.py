"""1c단계: EA FC(FIFA) 평점을 Transfermarkt 선수에게 연결한다.

EA FC 평점(overall)은 1~99 척도로 "지금 이 선수가 얼마나 잘하는가"를 매긴 값이다.
몸값과 달리 나이, 계약 기간, 시장 분위기가 직접 들어가지 않아서, 이 프로젝트에서
"기량"의 기준(정답)으로 쓴다.

원본 (data/raw_ea/):
  fc24_all/male_players.csv          FIFA 15 ~ FC 24 출시 시점 평점 (2014년 9월 ~ 2023년 9월)
  fc25_sofifa/player-data-full-...   FC 25 시즌 막바지 평점 (2025년 6월 3일)
  fc26_sofifa/FC26_20250921.csv      FC 26 출시 시점 평점 (2025년 9월 19일)

EA 쪽 선수 ID(sofifa id)는 판이 바뀌어도 같다. 그래서 EA 선수 한 명을 Transfermarkt 선수
한 명에 한 번만 연결하면, 모든 판의 평점이 따라온다.

실행:  python src/01c_add_ea.py
출력:  data/processed/ea_ratings.parquet   (Transfermarkt 선수, 평점 공개일)마다 한 행
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EA = ROOT / "data" / "raw_ea"
PROFILES = ROOT / "data" / "raw_salimt" / "player_profiles.csv"
OUT = ROOT / "data" / "processed" / "ea_ratings.parquet"

normalize_name = importlib.import_module("01b_add_fbref").normalize_name   # FBref 연결 때 만든 이름 정리 함수


def load_ea():
    """EA 평점 세 파일을 같은 모양으로 합친다: (sofifa_id, 공개일, 판, overall, potential, 이름, 생일)."""
    hist = pd.read_csv(EA / "fc24_all" / "male_players.csv", low_memory=False, usecols=[
        "player_id", "fifa_version", "update_as_of", "long_name", "short_name", "dob", "overall", "potential"])
    hist["edition"] = "FIFA " + hist["fifa_version"].astype(int).astype(str)
    hist.loc[hist["fifa_version"] >= 24, "edition"] = "FC 24"
    hist = hist.rename(columns={"update_as_of": "published"})

    fc25 = pd.read_csv(EA / "fc25_sofifa" / "player-data-full-2025-june.csv", low_memory=False, usecols=[
        "player_id", "version", "full_name", "name", "dob", "overall_rating", "potential"])
    fc25 = fc25.rename(columns={"version": "published", "full_name": "long_name", "overall_rating": "overall"})
    fc25["short_name"] = fc25.pop("name").str.rstrip(" -")
    fc25["edition"] = "FC 25 (6월)"

    fc26 = pd.read_csv(EA / "fc26_sofifa" / "FC26_20250921.csv", low_memory=False, usecols=[
        "player_id", "fifa_update_date", "long_name", "short_name", "dob", "overall", "potential"])
    fc26 = fc26.rename(columns={"fifa_update_date": "published"})
    fc26["edition"] = "FC 26"

    cols = ["player_id", "published", "edition", "overall", "potential", "long_name", "short_name", "dob"]
    ea = pd.concat([hist[cols], fc25[cols], fc26[cols]], ignore_index=True)
    ea = ea.rename(columns={"player_id": "sofifa_id"})
    ea["published"] = pd.to_datetime(ea["published"])
    ea["dob"] = pd.to_datetime(ea["dob"], errors="coerce")
    return ea.dropna(subset=["overall"])


def link_to_transfermarkt(ea):
    """EA 선수(sofifa_id)마다 생년월일이 같은 Transfermarkt 선수 중 이름이 맞는 한 명을 찾는다.

    생년월일은 정확히 같아야 한다. 같은 생일인 선수가 평균 6명쯤 있으므로, 그중 이름으로 고른다.
    아래 규칙을 순서대로 적용해서, 처음으로 후보가 딱 한 명 남는 규칙에서 연결한다.
      1. 정리한 이름이 EA 전체 이름이나 짧은 이름과 똑같다
      2. Transfermarkt 이름의 단어가 전부 EA 전체 이름 안에 있다   (Erling Haaland ⊂ Erling Braut Haaland)
      3. 성이 같고 이름 첫 글자가 같다                               (Erling Haaland ~ E. Haaland)
      4. Transfermarkt 이름의 마지막 단어가 EA 전체 이름 안에 있다
    """
    people = (ea.sort_values("published").groupby("sofifa_id")
              [["long_name", "short_name", "dob"]].last().reset_index().dropna(subset=["dob"]))
    people["long_tokens"] = people["long_name"].map(lambda s: set(normalize_name(s).split()))
    people["short_norm"] = people["short_name"].map(normalize_name)

    tm = pd.read_csv(PROFILES, low_memory=False, usecols=["player_id", "player_name", "date_of_birth"])
    tm["name"] = tm["player_name"].str.replace(r"\s*\(\d+\)$", "", regex=True)
    tm["norm"] = tm["name"].map(normalize_name)
    tm["dob"] = pd.to_datetime(tm["date_of_birth"], errors="coerce")
    tm = tm.dropna(subset=["dob"])

    pairs = people.merge(tm[["player_id", "norm", "dob"]], on="dob")      # 생일이 같은 모든 짝
    tokens = pairs["norm"].str.split()
    first, last = tokens.str[0].fillna(""), tokens.str[-1].fillna("")
    short_tokens = pairs["short_norm"].str.split()
    rules = {
        1: (pairs["norm"] == pairs["long_tokens"].map(" ".join)) | (pairs["norm"] == pairs["short_norm"]),
        2: [set(t) <= lt for t, lt in zip(tokens, pairs["long_tokens"])],
        3: (last == short_tokens.str[-1].fillna("#")) & (first.str[:1] == pairs["short_norm"].str[:1]),
        4: [l in lt for l, lt in zip(last, pairs["long_tokens"])],
    }
    linked = []
    remaining = set(pairs["sofifa_id"])
    for rule, ok in rules.items():
        cand = pairs[pd.Series(ok, index=pairs.index) & pairs["sofifa_id"].isin(remaining)]
        n = cand.groupby("sofifa_id")["player_id"].transform("size")
        unique = cand[n == 1][["sofifa_id", "player_id"]].assign(rule=rule)
        linked.append(unique)
        remaining -= set(unique["sofifa_id"])
    links = pd.concat(linked, ignore_index=True)
    # 한 Transfermarkt 선수에 EA 선수 두 명이 붙으면 어느 쪽이 맞는지 모르므로 둘 다 버린다
    links = links[~links.duplicated("player_id", keep=False)]
    return links, len(people)


def main():
    ea = load_ea()
    print("EA 평점 판별 선수 수와 공개일:")
    print(ea.groupby("edition", sort=False).agg(선수=("sofifa_id", "nunique"), 공개일=("published", "max")).to_string())

    links, n_people = link_to_transfermarkt(ea)
    print(f"\nEA 선수 {n_people:,}명 중 Transfermarkt 선수와 연결: {len(links):,}명 ({len(links) / n_people:.1%})")
    print("연결된 규칙별 인원:", links["rule"].value_counts().sort_index().to_dict())

    out = ea.merge(links[["sofifa_id", "player_id"]], on="sofifa_id")
    out = out[["player_id", "sofifa_id", "published", "edition", "overall", "potential"]]
    print("\n판별로 연결된 비율:")
    print((ea.merge(links, on="sofifa_id", how="left").groupby("edition", sort=False)["player_id"]
           .apply(lambda s: f"{s.notna().mean():.1%}")).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\n저장: {OUT}  ({len(out):,}행)")


if __name__ == "__main__":
    main()
