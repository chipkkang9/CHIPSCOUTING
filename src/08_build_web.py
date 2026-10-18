"""8단계: 선수 검색 웹 페이지 만들기 (한국어 판, 영어 판).

06_ability_models.py가 만든 scout_2025.parquet의 점수를 페이지 안에 넣어서, 서버 없이 브라우저만으로
검색되는 HTML을 만든다. 07_scout.py의 터미널 검색과 같은 기능을 화면으로 옮긴 것이다.

페이지에 들어가는 것
  - 현재 기량, 1~5년 뒤 예상 기량 (기간을 골라 볼 수 있다)과 80% 범위
  - 전체 기량 랭킹 (리그, 포지션, 나이대로 거를 수 있다)과 유망주 순위
  - 선수 사진 (위키미디어 공용의 자유 라이선스 사진)
      브라우저로 바로 여는 판: 위키미디어 주소에서 사진을 바로 불러온다. 위키데이터에 사진이 있는 선수 전부
      claude.ai 판: 외부 이미지가 막혀 있어서, 페이지와 함께 올린 사본(약 230명)만 쓴다
  - 몸값 (유로, 파운드, 원), 나이와 태어난 해
  - 포지션 숙련도 (EA FC 26의 포지션별 평점, 축구장 그림)

실행:  python src/08_build_web.py
출력:  web/index.html, web/en.html                   브라우저로 바로 여는 파일 (한국어, 영어)
       web/chipscouting_ko.html, web/chipscouting_en.html   claude.ai 아티팩트로 올리는 파일
"""
import json
import math
from pathlib import Path

import pandas as pd

from common import EA_DATASET, ROOT

SCOUT_TABLE = ROOT / "data" / "processed" / "scout_2025.parquet"
PROFILES = ROOT / "data" / "raw_salimt" / "player_profiles.csv"
PERFORMANCES = ROOT / "data" / "raw_salimt" / "player_performances.csv"
FC26 = ROOT / "data" / "raw_ea" / "fc26_sofifa" / "FC26_20250921.csv"
FC25 = ROOT / "data" / "raw_ea" / "fc25_sofifa" / "player-data-full-2025-june.csv"
PHOTO_META = ROOT / "data" / "processed" / "photos.json"                 # 08a_fetch_media.py가 만든 사진 정보
ARTIFACT_PHOTOS = ROOT / "data" / "processed" / "photos_artifact.json"   # claude.ai 페이지에 올려 둔 사진의 선수 ID
FX_FILE = ROOT / "data" / "processed" / "fx_rates.json"
WEB = ROOT / "web"
HORIZONS = [1, 2, 3, 4, 5]

# 아티팩트 주소 (한국어 판, 영어 판). 서로 링크한다.
ARTIFACT_URL = {"ko": "https://claude.ai/artifact/PBQm92L2Qnr2p5cvyUXaNZ", "en": "https://claude.ai/artifact/7ivVuDbXFRfmxCYTogLKSu"}

# 축구장 그림의 칸: (칸 이름, EA FC 26 평점 열, EA 포지션 표기들)
SLOTS = [
    ("ST", "st", ["ST", "CF"]),
    ("LW", "lw", ["LW"]), ("CAM", "cam", ["CAM"]), ("RW", "rw", ["RW"]),
    ("LM", "lm", ["LM"]), ("CM", "cm", ["CM"]), ("RM", "rm", ["RM"]),
    ("LWB", "lwb", ["LWB"]), ("CDM", "cdm", ["CDM"]), ("RWB", "rwb", ["RWB"]),
    ("LB", "lb", ["LB"]), ("CB", "cb", ["CB"]), ("RB", "rb", ["RB"]),
    ("GK", "gk", ["GK"]),
]
EA_TO_SLOT = {code: i for i, (_, _, codes) in enumerate(SLOTS) for code in codes}
# Transfermarkt 세부 포지션 -> 칸 (EA 평점이 없는 선수에게 주 포지션 하나만 표시할 때)
TM_TO_SLOT = {
    "Centre-Forward": "ST", "Second Striker": "ST", "Attack": "ST", "Left Winger": "LW", "Right Winger": "RW",
    "Attacking Midfield": "CAM", "Left Midfield": "LM", "Right Midfield": "RM", "Central Midfield": "CM",
    "Midfield": "CM", "Defensive Midfield": "CDM", "Left-Back": "LB", "Right-Back": "RB", "Centre-Back": "CB",
    "Sweeper": "CB", "Defender": "CB", "Goalkeeper": "GK",
}
SLOT_INDEX = {name: i for i, (name, _, _) in enumerate(SLOTS)}


# ---------------------------------------------------------------------------
# 데이터 모으기
# ---------------------------------------------------------------------------
def league_names():
    """대회 ID -> 대회 이름. 출전 기록에 가장 많이 적힌 이름을 쓴다. ('GB1' -> 'Premier League')"""
    perf = pd.read_csv(PERFORMANCES, usecols=["competition_id", "competition_name"], low_memory=False)
    return perf.groupby("competition_id")["competition_name"].agg(lambda s: s.value_counts().index[0])


def position_ratings():
    """Transfermarkt 선수 ID -> (칸별 EA 평점 14개, 주 포지션 칸 번호들).

    EA FC 26 평점이 있으면 포지션별 평점과 EA가 정한 주 포지션을 쓴다.
    없으면 FC 25 (6월)의 포지션 목록만 쓴다 (평점 숫자 없음).
    """
    links = pd.read_parquet(EA_DATASET, columns=["player_id", "sofifa_id", "edition"])
    out = {}

    fc25 = pd.read_csv(FC25, low_memory=False, usecols=["player_id", "positions"])
    ids25 = links[links["edition"] == "FC 25 (6월)"].drop_duplicates("sofifa_id").set_index("sofifa_id")["player_id"]
    for sofifa, positions in zip(fc25["player_id"], fc25["positions"]):
        if sofifa in ids25.index and isinstance(positions, str):
            natural = [EA_TO_SLOT[p.strip()] for p in positions.split(",") if p.strip() in EA_TO_SLOT]
            out[int(ids25[sofifa])] = (None, list(dict.fromkeys(natural)))

    cols = ["player_id", "player_positions"] + [c for _, c, _ in SLOTS]
    fc26 = pd.read_csv(FC26, low_memory=False, usecols=cols)
    ids26 = links[links["edition"] == "FC 26"].drop_duplicates("sofifa_id").set_index("sofifa_id")["player_id"]
    for _, row in fc26.iterrows():
        if row["player_id"] not in ids26.index:
            continue
        # "86+3"처럼 적혀 있다. 앞의 숫자가 그 포지션의 기본 평점이다
        ratings = [int(str(row[c]).split("+")[0].split("-")[0]) for _, c, _ in SLOTS]
        natural = [EA_TO_SLOT[p.strip()] for p in str(row["player_positions"]).split(",") if p.strip() in EA_TO_SLOT]
        out[int(ids26[row["player_id"]])] = (ratings, list(dict.fromkeys(natural)))
    return out


def encode(values):
    """같은 글자가 반복되는 열을 (고유값 목록, 번호 배열)로 바꿔 파일 크기를 줄인다."""
    codes, uniques = pd.factorize(values.fillna(""), sort=True)
    return list(uniques), codes.tolist()


def ints(series, scale=1):
    """숫자 열을 정수 배열로. 빈 값은 null. scale=10이면 소수 첫째 자리까지 보존한다."""
    return [None if pd.isna(v) else int(round(v * scale)) for v in series]


def ages(series):
    """나이를 10배 정수로. 반올림하지 않고 내린다.
    24.97세를 반올림해 250으로 적으면 페이지가 만 나이를 25세로 보여준다."""
    return [None if pd.isna(v) else int(math.floor(v * 10)) for v in series]


def scores(series):
    """점수를 10배 정수로. 페이지가 이 값을 다시 정수로 반올림해도 원래 점수의 반올림과 같게 적는다.
    89.46점을 그냥 반올림하면 895가 되고, 페이지에서 89.5를 다시 반올림해 90점이 된다. 이때는 894로 적는다."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(None)
            continue
        k = int(round(v * 10))
        if k % 10 == 5 and v < k / 10:
            k -= 1
        out.append(k)
    return out


def build_data():
    t = pd.read_parquet(SCOUT_TABLE).dropna(subset=["name"])
    t = t.sort_values("current_ability", ascending=False).reset_index(drop=True)   # 기본 순서 = 현재 기량순
    names = league_names()
    t["league_label"] = t["league"].map(lambda c: f"{names.get(c, c)} · {c}" if isinstance(c, str) and c else "")
    t["detail"] = t["position"].astype(str).str.split(" - ").str[-1]
    t["nation"] = t["citizenship"].fillna("").str.replace(r"\s{2,}", " / ", regex=True)
    dob = pd.read_csv(PROFILES, usecols=["player_id", "date_of_birth"], low_memory=False)
    born = pd.to_datetime(dob.set_index("player_id")["date_of_birth"], errors="coerce").dt.year
    t["born"] = t["player_id"].map(born)

    data = {"name": t["name"].tolist(), "age": ages(t["age"]), "born": ints(t["born"])}
    data["mainList"], data["main"] = encode(t["main_position"].astype(str))
    for key, col in [("detail", "detail"), ("club", "club_name"), ("league", "league_label"), ("nation", "nation")]:
        data[key + "List"], data[key] = encode(t[col])
    data.update({
        "value": ints(t["value_now"] / 1000),            # 천 유로 단위
        "games": ints(t["games_1s"]), "starts": ints(t["starts_1s"]),
        "goals": ints(t["goals_1s"]), "assists": ints(t["assists_1s"]),
        "eaPrev": ints(t["ea_overall_prev"]), "eaPot": ints(t["ea_potential_prev"]), "eaNow": ints(t["ability_now"]),
        "cur": scores(t["current_ability"]), "curPct": ints(t["current_pct"], 10),
    })
    for h in HORIZONS:
        data[f"fut{h}"] = scores(t[f"future_{h}"])
        data[f"lo{h}"] = scores(t[f"low_{h}"])
        data[f"hi{h}"] = scores(t[f"high_{h}"])
        data[f"pct{h}"] = ints(t[f"future_pct_age_{h}"], 10)

    # 포지션 숙련도: 평점이 있는 선수는 14칸 평점, 없는 선수는 주 포지션만
    ratings = position_ratings()
    data["slots"] = [name for name, _, _ in SLOTS]
    pos_rows, pos_index, natural = [], [], []
    for pid, detail in zip(t["player_id"], t["detail"]):
        r, nat = ratings.get(int(pid), (None, []))
        if not nat and detail in TM_TO_SLOT:
            nat = [SLOT_INDEX[TM_TO_SLOT[detail]]]
        natural.append(nat)
        if r is None:
            pos_index.append(-1)
        else:
            pos_index.append(len(pos_rows))
            pos_rows.append(r)
    data.update({"posRows": pos_rows, "posIndex": pos_index, "natural": natural})

    data["fx"] = json.loads(FX_FILE.read_text()) if FX_FILE.exists() else {"date": None, "GBP": None, "KRW": None}
    return data, t["player_id"].tolist()


def photo_arrays(player_ids, mode):
    """선수별 사진 번호와 사진 목록 [주소, 촬영자, 라이선스, 위키미디어 파일 이름].

    mode="local": 위키미디어 주소에서 바로 불러온다 (사진 정보가 있는 선수 전부).
    mode="artifact": claude.ai 페이지에 함께 올려 둔 사본 photos/<ID>.jpg를 쓴다 (올려 둔 선수만).
    """
    meta = json.loads(PHOTO_META.read_text()) if PHOTO_META.exists() else {}
    if mode == "artifact":
        uploaded = set(json.loads(ARTIFACT_PHOTOS.read_text())["player_ids"]) if ARTIFACT_PHOTOS.exists() else set()
        usable = {str(i) for i in uploaded} & set(meta)
        src = lambda pid: f"photos/{pid}.jpg"
    else:
        usable = set(meta)
        # 뒤에 붙은 추적용 꼬리(?utm_...)는 떼어 크기를 줄이고, 오래 써 온 사진 서버 주소(upload.wikimedia.org)로 바꾼다.
        # 위키미디어 공용이 최근 돌려주는 thumb.wikimedia.org도 같은 사진이지만, 시험해 보니 응답이 들쭉날쭉했다.
        src = lambda pid: meta[pid]["thumb"].split("?")[0].replace("//thumb.wikimedia.org/", "//upload.wikimedia.org/")
    index, photos = [], []
    for pid in map(str, player_ids):
        if pid in usable:
            index.append(len(photos))
            m = meta[pid]
            photos.append([src(pid), m["author"], m["license"], m["name"]])
        else:
            index.append(-1)
    return index, photos


# ---------------------------------------------------------------------------
# 글자 (한국어 / 영어)
# ---------------------------------------------------------------------------
TEXT = {
    "ko": {
        "html_lang": "ko",
        "subtitle": "2025년 7월 1일 기준 · 현재 기량과 1~5년 뒤 예상 기량 (1~100)",
        "tab_search": "선수 검색", "tab_all": "전체 랭킹", "tab_rank": "유망주 순위", "tabs_label": "보기 선택",
        "search_label": "선수 이름을 영문으로 입력하세요. 일부만 써도 됩니다.",
        "search_placeholder": "예: Yamal, Son Heung, Bellingham",
        "examples": "예시", "results_label": "검색 결과",
        "f_age": "나이 상한", "f_pos": "포지션", "f_sort": "정렬", "f_years": "예측 기간",
        "pos_all": "전체", "sort_fut": "N년 뒤 예상 기량", "sort_cur": "현재 기량", "sort_gain": "N년 동안 오를 점수",
        "rank_note": "행을 누르면 선수 카드가 열립니다.",
        "th": ["순위", "이름", "나이", "포지션", "소속", "현재", "예상", "80% 범위", "몸값"],
        "th_all": ["순위", "이름", "나이", "포지션", "소속", "리그", "현재", "예상", "몸값"],
        "f_league": "리그", "league_all": "전체 리그", "f_ageband": "나이", "more": "100명 더 보기",
        "other_lang": "English",
        "footer": [
            "점수는 EA FC(FIFA) 평점과 같은 1~99 척도입니다. 현재 기량은 7월 1일까지의 출전 기록, 뛴 대회와 소속 팀의 수준, 몸값, 작년 EA 평점으로 추정했고, 예상 기량은 과거 선수들의 사례로 배운 1~5년 뒤 기량입니다.",
            "80% 범위는 과거 예측에서 실제 결과가 갈린 폭입니다. 10명 중 8명은 이 범위 안에 들었습니다. 등급 이름(월드클래스, 정상급 등)은 점수대를 읽기 쉽게 붙인 대략적인 기준입니다.",
            "포지션 숙련도는 EA FC 26의 포지션별 평점입니다.{photo_note}",
            "데이터: Transfermarkt 기반 공개 데이터셋, EA FC 평점 공개 데이터셋(Kaggle){photo_src}. 2025년 7월 기준 최근 1년 안에 몸값이 매겨진 15~40세 선수만 들어 있습니다.",
        ],
        "photo_note": " 선수 사진은 위키미디어 공용의 자유 라이선스 사진이며, 위키데이터에 사진이 등록된 선수만 나옵니다.",
        "photo_src": ", 위키데이터",
    },
    "en": {
        "html_lang": "en",
        "subtitle": "As of 1 July 2025 · current ability and projected ability 1–5 years ahead (1–100)",
        "tab_search": "Player search", "tab_all": "Overall ranking", "tab_rank": "Prospect ranking", "tabs_label": "Choose view",
        "search_label": "Type a player's name. Part of the name is enough.",
        "search_placeholder": "e.g. Yamal, Son Heung, Bellingham",
        "examples": "Try", "results_label": "Search results",
        "f_age": "Max age", "f_pos": "Position", "f_sort": "Sort by", "f_years": "Projection",
        "pos_all": "All", "sort_fut": "Ability in N years", "sort_cur": "Current ability", "sort_gain": "Points gained in N years",
        "rank_note": "Select a row to open the player card.",
        "th": ["Rank", "Name", "Age", "Position", "Club", "Now", "Projected", "80% range", "Value"],
        "th_all": ["Rank", "Name", "Age", "Position", "Club", "League", "Now", "Projected", "Value"],
        "f_league": "League", "league_all": "All leagues", "f_ageband": "Age", "more": "Show 100 more",
        "other_lang": "한국어",
        "footer": [
            "Scores use the same 1–99 scale as EA FC (FIFA) ratings. Current ability is estimated from appearances up to 1 July, the level of the competitions and club, market value and last year's EA rating. Projected ability is learned from how past players developed over 1–5 years.",
            "The 80% range shows how far real outcomes landed from past projections: 8 in 10 players ended up inside it. Tier names such as World class and Elite are rough guides to the score bands.",
            "Position ratings are EA FC 26's per-position ratings.{photo_note}",
            "Data: Transfermarkt-based open dataset, EA FC rating datasets (Kaggle){photo_src}. Covers players aged 15–40 with a market value in the year before July 2025.",
        ],
        "photo_note": " Player photos are freely licensed images from Wikimedia Commons, shown only for players with a photo on Wikidata.",
        "photo_src": ", Wikidata",
    },
}

JS_TEXT = {
    "ko": {
        "pos": {"Attack": "공격수", "Midfield": "미드필더", "Defender": "수비수", "Goalkeeper": "골키퍼"},
        "tiers": ["월드클래스", "정상급", "빅리그 주전급", "1부 리그 주전급", "1부 리그 로테이션급", "성장 단계"],
        "now": "현재 기량", "future": "{n}년 뒤 예상 기량", "of": "/ 100",
        "topAll": "전체 선수 중 상위 {p}%", "range": "80% 범위 {lo}~{hi}", "topAge": "같은 나이 선수 중 상위 {p}%",
        "up": "상승 예상", "down": "하락 예상", "flat": "유지 예상",
        "age": "만 {a}세", "born": "{y}년생", "noClub": "소속 정보 없음",
        "value": "몸값", "valueNote": "2025년 7월 기준 · 환율 {d} 유럽중앙은행",
        "season": "지난 시즌 {g}경기 (선발 {s}) · {go}골 {as}도움",
        "years": "{n}년", "nowShort": "지금", "inYears": "{n}년 뒤",
        "trend": "기량 흐름", "trendNote": "띠는 80% 범위",
        "pitch": "포지션 숙련도", "pitchNote": "EA FC 26 포지션별 평점 · 금색 테두리 = 주 포지션",
        "pitchNoteNoEA": "EA 평점이 없어 주 포지션만 표시",
        "photoBy": "사진", "noPhoto": "사진 없음",
        "ea25": "EA FC 25 (2025년 6월)", "eaPot": "EA 잠재력", "ea26": "EA FC 26 (2025년 9월)",
        "listHint": "이름을 입력하면 여기 목록이 나옵니다. 지금은 예시 선수를 보여 드립니다.",
        "noMatch": "\"{q}\"과 맞는 선수가 없습니다. 영문 이름의 일부만 넣어 보세요. (예: Son, Yamal)",
        "count": "{n} · 현재 기량순", "countMany": "50명 이상", "countN": "{n}명",
        "yearsLabel": "예측 기간", "ageCol": "{a}",
        "ageBands": {"": "전체 나이", "0-21": "21세 이하", "22-25": "22~25세", "26-29": "26~29세", "30-99": "30세 이상"},
        "shown": "{total}명 중 {k}명 표시 · 행을 누르면 선수 카드가 열립니다.",
        "krw": "ko",
    },
    "en": {
        "pos": {"Attack": "Forward", "Midfield": "Midfielder", "Defender": "Defender", "Goalkeeper": "Goalkeeper"},
        "tiers": ["World class", "Elite", "Big-league starter", "Top-flight starter", "Top-flight squad player", "Developing"],
        "now": "Current ability", "future": "Projected ability in {n} yr", "of": "/ 100",
        "topAll": "Top {p}% of all players", "range": "80% range {lo}–{hi}", "topAge": "Top {p}% of players the same age",
        "up": "Rising", "down": "Declining", "flat": "Steady",
        "age": "Age {a}", "born": "Born {y}", "noClub": "Club unknown",
        "value": "Market value", "valueNote": "As of July 2025 · exchange rates {d}, European Central Bank",
        "season": "Last season {g} apps ({s} starts) · {go} goals {as} assists",
        "years": "{n} yr", "nowShort": "Now", "inYears": "In {n} yr",
        "trend": "Ability path", "trendNote": "Band shows the 80% range",
        "pitch": "Position ratings", "pitchNote": "EA FC 26 per-position ratings · gold ring = natural position",
        "pitchNoteNoEA": "No EA rating, natural position only",
        "photoBy": "Photo", "noPhoto": "No photo",
        "ea25": "EA FC 25 (June 2025)", "eaPot": "EA potential", "ea26": "EA FC 26 (Sept 2025)",
        "listHint": "Matching players appear here as you type. An example player is shown for now.",
        "noMatch": "No player matches \"{q}\". Try part of the name, e.g. Son or Yamal.",
        "count": "{n} · by current ability", "countMany": "50+ players", "countN": "{n} players",
        "yearsLabel": "Projection", "ageCol": "{a}",
        "ageBands": {"": "All ages", "0-21": "21 and under", "22-25": "22–25", "26-29": "26–29", "30-99": "30 and over"},
        "shown": "Showing {k} of {total} players · select a row to open the player card.",
        "krw": "en",
    },
}


PAGE = r"""<title>CHIPSCOUTING</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap">
<style>
/* 레이아웃: 스카우트 리포트. 왼쪽은 검색 결과, 오른쪽은 선수 카드(사진과 포지션 그림이 카드 왼쪽).
   숫자는 경기장 전광판처럼 좁고 큰 글자. 초록 = 현재 기량, 금색 = 예상 기량. */
:root {
  --bg: #f2f4ef;
  --surface: #ffffff;
  --ink: #16221b;
  --muted: #5b685f;
  --line: #d6ddd5;
  --accent: #1c6a44;
  --gold: #a96c12;
  --band: rgba(169, 108, 18, 0.16);
  --soft: #e7ece5;
  --up: #1c6a44;
  --down: #a33a2b;
  --t1: #9c5a06; --t2: #1c6a44; --t3: #22707e; --t4: #45618a; --t5: #6a6b7c; --t6: #7d847b;
  /* 포지션 그림은 두 테마 모두 어두운 잔디 위에 그린다 */
  --pitch: #14261c; --pitch-line: rgba(232, 240, 233, 0.32); --pitch-text: #eef4ef;
  --p4: #7ee36a; --p3: #4fbf55; --p2: #2f8f45; --p1: #1f5e33; --p0: #2a3a30; --ring: #f2c14e;
  --display: "Barlow Condensed", "Arial Narrow", "Helvetica Neue", Arial, sans-serif;
  --body: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1511; --surface: #151f19; --ink: #e4ebe5; --muted: #98a69d; --line: #26332b;
    --accent: #4dbb86; --gold: #e3a84c; --band: rgba(227, 168, 76, 0.20); --soft: #1c2821;
    --up: #4dbb86; --down: #ec7b68;
    --t1: #e8a64a; --t2: #4dbb86; --t3: #4fb6c6; --t4: #8aa7d6; --t5: #a5a7ba; --t6: #9aa39a;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0e1511; --surface: #151f19; --ink: #e4ebe5; --muted: #98a69d; --line: #26332b;
  --accent: #4dbb86; --gold: #e3a84c; --band: rgba(227, 168, 76, 0.20); --soft: #1c2821;
  --up: #4dbb86; --down: #ec7b68;
  --t1: #e8a64a; --t2: #4dbb86; --t3: #4fb6c6; --t4: #8aa7d6; --t5: #a5a7ba; --t6: #9aa39a;
  color-scheme: dark;
}
body { background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 15px; line-height: 1.55; }
.wrap { max-width: 1240px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }

.top { border-bottom: 1px solid var(--line); background: var(--surface); }
.top .wrap { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px 24px; padding-block: 18px 0; }
.brand h1 { margin: 0; font-family: var(--display); font-weight: 700; font-size: 34px; letter-spacing: 0.04em; line-height: 1; }
.brand p { margin: 6px 0 14px; color: var(--muted); font-size: 13px; }
.navside { display: flex; align-items: flex-end; gap: 14px; flex-wrap: wrap; }
.lang { font-size: 13px; color: var(--muted); padding-bottom: 14px; text-decoration: none; border-bottom: 1px solid transparent; }
.lang:hover { color: var(--ink); border-bottom-color: var(--line); }
.tabs { display: flex; gap: 4px; }
.tab { appearance: none; border: 0; background: none; color: var(--muted); font: 600 14px/1 var(--body); padding: 12px 14px 13px; border-bottom: 3px solid transparent; cursor: pointer; }
.tab[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); }
.tab:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible, .result:focus-visible, tr:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

main { padding-block: 22px 40px; }
.searchbar { display: grid; gap: 8px; margin-bottom: 18px; }
.searchbar label { font-size: 13px; color: var(--muted); }
#q { width: 100%; box-sizing: border-box; font: 500 18px/1.3 var(--body); color: var(--ink); background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.examples { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 13px; color: var(--muted); }
.chip { appearance: none; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 999px; padding: 4px 10px; font: 500 13px/1.4 var(--body); cursor: pointer; }
.chip:hover { border-color: var(--accent); }

.split { display: grid; grid-template-columns: minmax(0, 280px) minmax(0, 1fr); gap: 20px; align-items: start; }
@media (max-width: 900px) { .split { grid-template-columns: minmax(0, 1fr); } .results { max-height: 240px; } }
.results { list-style: none; margin: 0; padding: 0; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); max-height: 640px; overflow-y: auto; }
.results .count { padding: 10px 12px; font-size: 12px; color: var(--muted); border-bottom: 1px solid var(--line); }
.result { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2px 10px; padding: 9px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }
.result:last-child { border-bottom: 0; }
.result:hover { background: var(--soft); }
.result[aria-current="true"] { background: var(--soft); box-shadow: inset 3px 0 0 var(--accent); }
.result .nm { font-weight: 600; min-width: 0; overflow-wrap: anywhere; }
.result .sub { grid-column: 1; font-size: 12px; color: var(--muted); min-width: 0; overflow-wrap: anywhere; }
.result .mini { grid-row: 1 / span 2; grid-column: 2; align-self: center; font-family: var(--display); font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.mini .arrow { color: var(--muted); font-size: 16px; margin: 0 2px; }
.mini .f { color: var(--gold); }
.empty { padding: 16px 12px; color: var(--muted); font-size: 14px; }

.card { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 22px; min-width: 0; }
.card-grid { display: grid; grid-template-columns: minmax(0, 240px) minmax(0, 1fr); gap: 24px; align-items: start; }
@media (max-width: 640px) { .card-grid { grid-template-columns: minmax(0, 1fr); } .side { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); } }
.side { display: grid; gap: 16px; align-content: start; }
.photo { margin: 0; display: grid; gap: 6px; }
.photo .frame { aspect-ratio: 3 / 4; width: 100%; max-width: 100%; border-radius: 10px; overflow: hidden; background: var(--soft); display: grid; place-items: center; }
.photo img { width: 100%; height: 100%; object-fit: cover; object-position: 50% 20%; display: block; }
.photo .initials { font-family: var(--display); font-weight: 700; font-size: 64px; color: var(--muted); letter-spacing: 0.04em; }
.photo figcaption { font-size: 11px; color: var(--muted); line-height: 1.4; overflow-wrap: anywhere; }
.photo figcaption a { color: inherit; }
.pitchbox h3, .trend h3 { margin: 0 0 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.06em; color: var(--muted); }
.pitchbox svg { display: block; width: 100%; height: auto; border-radius: 10px; }
.pitchbox .note, .trend .note { margin: 6px 0 0; font-size: 11px; color: var(--muted); line-height: 1.4; }

.mainbox { min-width: 0; display: grid; gap: 4px; align-content: start; }
.card h2 { margin: 0; font-family: var(--display); font-weight: 700; font-size: 42px; line-height: 1.05; letter-spacing: 0.01em; text-wrap: balance; overflow-wrap: anywhere; }
.meta { margin: 4px 0 0; color: var(--muted); font-size: 14px; overflow-wrap: anywhere; }
.meta strong { color: var(--ink); font-weight: 600; }
.money { margin: 12px 0 2px; border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; display: grid; gap: 4px; }
.money .lbl { font-size: 12px; letter-spacing: 0.06em; color: var(--muted); }
.money .vals { display: flex; flex-wrap: wrap; gap: 4px 22px; align-items: baseline; }
.money .v { font-family: var(--display); font-weight: 600; font-size: 24px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.money .note { font-size: 11px; color: var(--muted); }

.horizon { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; margin: 16px 0 0; }
.horizon .lbl { font-size: 12px; letter-spacing: 0.06em; color: var(--muted); }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; }
.seg button { appearance: none; border: 0; background: var(--surface); color: var(--muted); font: 600 13px/1 var(--body); padding: 8px 13px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--line); }
.seg button[aria-checked="true"] { background: var(--gold); color: var(--surface); }

.scores { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 14px 0 6px; }
@media (max-width: 520px) { .scores { grid-template-columns: minmax(0, 1fr); } }
.score { border-top: 3px solid var(--accent); padding-top: 10px; display: grid; gap: 4px; }
.score.future { border-top-color: var(--gold); }
.score .label { font-size: 12px; letter-spacing: 0.06em; color: var(--muted); }
.score .big { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.score .num { font-family: var(--display); font-weight: 700; font-size: 64px; line-height: 0.95; font-variant-numeric: tabular-nums; }
.score.now .num { color: var(--accent); }
.score.future .num { color: var(--gold); }
.score .of { font-family: var(--display); font-size: 22px; color: var(--muted); }
.score .sub { font-size: 13px; color: var(--muted); }
.tier { display: inline-block; font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 999px; border: 1px solid currentColor; }
.delta { font-family: var(--display); font-weight: 600; font-size: 22px; font-variant-numeric: tabular-nums; }
.delta.up { color: var(--up); } .delta.down { color: var(--down); } .delta.flat { color: var(--muted); }

.scale { margin: 14px 0 0; }
.scale svg, .trend svg { display: block; width: 100%; height: auto; overflow: visible; }
.scale text, .trend text { font-family: var(--display); font-size: 13px; fill: var(--muted); }
.scale .mk text { font-family: var(--body); font-size: 12px; font-weight: 600; }
.trend { margin: 16px 0 0; }
.ref { display: flex; flex-wrap: wrap; gap: 6px 22px; margin: 14px 0 0; padding-top: 12px; border-top: 1px dashed var(--line); font-size: 13px; color: var(--muted); }
.ref b { color: var(--ink); font-family: var(--display); font-size: 17px; font-weight: 600; margin-left: 4px; font-variant-numeric: tabular-nums; }

.filters { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: end; margin-bottom: 14px; }
.filters label { display: grid; gap: 4px; font-size: 12px; color: var(--muted); }
.filters select { font: 500 14px/1.3 var(--body); color: var(--ink); background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 7px 10px; }
.filters .note { font-size: 13px; color: var(--muted); }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { font-size: 12px; color: var(--muted); font-weight: 500; letter-spacing: 0.04em; position: sticky; top: 0; background: var(--surface); }
td.n { font-family: var(--display); font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums; }
td.n.f { color: var(--gold); }
td.n.c { color: var(--accent); }
td.rk { color: var(--muted); font-variant-numeric: tabular-nums; }
tbody tr { cursor: pointer; }
.more { appearance: none; display: block; margin: 12px auto 0; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 999px; padding: 8px 18px; font: 600 13px/1 var(--body); cursor: pointer; }
.more:hover { border-color: var(--accent); }
td.lg { color: var(--muted); max-width: 22ch; overflow: hidden; text-overflow: ellipsis; }
tbody tr:hover { background: var(--soft); }

/* 아래 설명 글: 위 내용과 같은 폭을 두 단으로 나눠 채운다 (좁은 화면에서는 한 단) */
footer { margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--line); font-size: 12.5px; color: var(--muted);
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 32px; }
@media (max-width: 760px) { footer { grid-template-columns: minmax(0, 1fr); } }
footer p { margin: 0; }
@media (prefers-reduced-motion: no-preference) { .card { animation: rise 0.25s ease-out; } @keyframes rise { from { transform: translateY(4px); } to { transform: none; } } }
</style>

<header class="top">
  <div class="wrap">
    <div class="brand">
      <h1>CHIPSCOUTING</h1>
      <p id="stamp">{{subtitle}}</p>
    </div>
    <div class="navside">
      <nav class="tabs" role="tablist" aria-label="{{tabs_label}}">
        <button class="tab" id="tab-search" role="tab" aria-selected="true" aria-controls="view-search">{{tab_search}}</button>
        <button class="tab" id="tab-all" role="tab" aria-selected="false" aria-controls="view-all">{{tab_all}}</button>
        <button class="tab" id="tab-rank" role="tab" aria-selected="false" aria-controls="view-rank">{{tab_rank}}</button>
      </nav>
      <a class="lang" href="{{other_href}}" lang="{{other_code}}">{{other_lang}}</a>
    </div>
  </div>
</header>

<main class="wrap">
  <section id="view-search" role="tabpanel" aria-labelledby="tab-search">
    <div class="searchbar">
      <label for="q">{{search_label}}</label>
      <input id="q" type="search" placeholder="{{search_placeholder}}" autocomplete="off" spellcheck="false">
      <div class="examples" id="examples">{{examples}}</div>
    </div>
    <div class="split">
      <ol class="results" id="results" aria-label="{{results_label}}"></ol>
      <article class="card" id="card" aria-live="polite"></article>
    </div>
  </section>

  <section id="view-all" role="tabpanel" aria-labelledby="tab-all" hidden>
    <div class="filters">
      <label for="a-sort">{{f_sort}}
        <select id="a-sort"></select>
      </label>
      <label for="a-years">{{f_years}}
        <select id="a-years"></select>
      </label>
      <label for="a-league">{{f_league}}
        <select id="a-league"><option value="">{{league_all}}</option></select>
      </label>
      <label for="a-pos">{{f_pos}}
        <select id="a-pos"><option value="">{{pos_all}}</option></select>
      </label>
      <label for="a-age">{{f_ageband}}
        <select id="a-age"></select>
      </label>
      <span class="note" id="a-count"></span>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr>{{th_all}}</tr></thead>
        <tbody id="all-body"></tbody>
      </table>
    </div>
    <button class="more" id="a-more" type="button">{{more}}</button>
  </section>

  <section id="view-rank" role="tabpanel" aria-labelledby="tab-rank" hidden>
    <div class="filters">
      <label for="f-years">{{f_years}}
        <select id="f-years"></select>
      </label>
      <label for="f-age">{{f_age}}
        <select id="f-age"></select>
      </label>
      <label for="f-pos">{{f_pos}}
        <select id="f-pos"><option value="">{{pos_all}}</option></select>
      </label>
      <label for="f-sort">{{f_sort}}
        <select id="f-sort"></select>
      </label>
      <span class="note">{{rank_note}}</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr>{{th}}</tr></thead>
        <tbody id="rank-body"></tbody>
      </table>
    </div>
  </section>

  <footer>{{footer}}</footer>
</main>

<script>
const D = __DATA__;
const T = __TEXT__;
const SORT_LABELS = __SORTS__;
const N = D.name.length;
const EXAMPLES = ["Lamine Yamal", "Heung-min Son", "Kang-in Lee", "Min-hyeok Yang", "Jude Bellingham", "Erling Haaland"];
const TIER_CUTS = [[85, "--t1"], [80, "--t2"], [75, "--t3"], [70, "--t4"], [65, "--t5"], [0, "--t6"]];
const SLOT_XY = {ST: [100, 34], LW: [34, 80], CAM: [100, 88], RW: [166, 80], LM: [30, 138], CM: [100, 142], RM: [170, 138],
  LWB: [30, 192], CDM: [100, 190], RWB: [170, 192], LB: [36, 240], CB: [100, 246], RB: [164, 240], GK: [100, 288]};
let H = 3;            // 고른 예측 기간 (년)
let current = null;   // 지금 카드에 보이는 선수 번호
try { const saved = +localStorage.getItem("chipscouting-years"); if (saved >= 1 && saved <= 5) H = saved; } catch (e) {}

function fmt(s, vars) { return s.replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? ""); }
function norm(s) {
  return (s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase()
    .replace(/ø/g, "o").replace(/đ/g, "d").replace(/ł/g, "l").replace(/ß/g, "ss").replace(/æ/g, "ae").replace(/ı/g, "i")
    .replace(/-/g, " ").replace(/[^a-z ]/g, "").trim();
}
const TOKENS = D.name.map(n => norm(n).split(/\s+/).filter(Boolean));
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function topShare(pct10) { return Math.max(100 - pct10 / 10, 0.1).toFixed(1); }
function tierIndex(score) { return TIER_CUTS.findIndex(t => score >= t[0]); }

// 몸값: 천 유로 단위 숫자를 유로, 파운드, 원으로
function euro(k) { if (k == null) return "-"; return k >= 1000 ? "€" + (k / 1000).toFixed(1) + "M" : "€" + k + "K"; }
function pound(k) { if (k == null || !D.fx.GBP) return "-"; const v = k * D.fx.GBP; return v >= 1000 ? "£" + (v / 1000).toFixed(1) + "M" : "£" + Math.round(v) + "K"; }
function won(k) {
  if (k == null || !D.fx.KRW) return "-";
  const w = k * 1000 * D.fx.KRW;
  if (T.krw === "en") return w >= 1e9 ? "KRW " + (w / 1e9).toFixed(1) + "B" : "KRW " + (w / 1e6).toFixed(1) + "M";
  if (w >= 1e12) return (w / 1e12).toFixed(1) + "조 원";
  if (w >= 1e8) return Math.round(w / 1e8).toLocaleString("ko-KR") + "억 원";
  return Math.round(w / 1e4).toLocaleString("ko-KR") + "만 원";
}

function player(i) {
  const p = {
    i, name: D.name[i], age: D.age[i] / 10, born: D.born[i], main: D.mainList[D.main[i]], detail: D.detailList[D.detail[i]],
    club: D.clubList[D.club[i]], league: D.leagueList[D.league[i]], nation: D.nationList[D.nation[i]],
    value: D.value[i], games: D.games[i], starts: D.starts[i], goals: D.goals[i], assists: D.assists[i],
    eaPrev: D.eaPrev[i], eaPot: D.eaPot[i], eaNow: D.eaNow[i], cur: D.cur[i] / 10, curPct: D.curPct[i],
    fut: {}, lo: {}, hi: {}, pct: {},
  };
  for (let h = 1; h <= 5; h++) { p.fut[h] = D["fut" + h][i] / 10; p.lo[h] = D["lo" + h][i] / 10; p.hi[h] = D["hi" + h][i] / 10; p.pct[h] = D["pct" + h][i]; }
  return p;
}

// 검색어의 단어 하나하나가 이름 속 어떤 단어의 앞부분과 맞으면 찾는다 ("son heung" -> Heung-min Son)
function search(query, limit) {
  const words = norm(query).split(/\s+/).filter(Boolean);
  const out = [];
  if (!words.length) return out;
  for (let i = 0; i < N && out.length < limit; i++) {
    const toks = TOKENS[i];
    if (words.every(w => toks.some(t => t.startsWith(w)))) out.push(i);
  }
  return out;  // 데이터가 현재 기량순으로 정렬돼 있어서 결과도 그 순서다
}

function photoHtml(p) {
  const k = D.photo[p.i];
  if (k < 0) {
    return `<figure class="photo"><div class="frame" role="img" aria-label="${esc(T.noPhoto)}"><span class="initials">${esc(initials(p.name))}</span></div></figure>`;
  }
  const [src, author, license, file] = D.photos[k];
  const page = "https://commons.wikimedia.org/wiki/File:" + encodeURIComponent(file.replace(/ /g, "_"));
  // 인터넷이 끊겼거나 사진이 지워져서 못 불러오면 머리글자로 바꾼다
  return `<figure class="photo"><div class="frame"><img src="${esc(src)}" alt="${esc(p.name)}" loading="lazy" onerror="photoFailed(this)"></div>
    <figcaption>${esc(T.photoBy)}: ${esc(author || "Wikimedia Commons")} · ${esc(license)} · <a href="${esc(page)}" target="_blank" rel="noopener">Wikimedia Commons</a></figcaption></figure>`;
}

function initials(name) { return name.split(/[\s-]+/).filter(Boolean).slice(0, 2).map(w => w[0]).join("").toUpperCase(); }
function photoFailed(img) {
  const fig = img.closest(".photo");
  fig.innerHTML = `<div class="frame"><span class="initials">${esc(initials(D.name[current] || ""))}</span></div>`;
}

// 포지션 숙련도: 세로 축구장 위 14칸. 가장 높은 평점과의 차이로 색을 정한다
function pitchHtml(p) {
  const k = D.posIndex[p.i], ratings = k >= 0 ? D.posRows[k] : null, natural = D.natural[p.i] || [];
  const best = ratings ? Math.max(...ratings) : 0;
  const level = r => { const d = best - r; return d <= 1 ? "--p4" : d <= 4 ? "--p3" : d <= 8 ? "--p2" : d <= 15 ? "--p1" : "--p0"; };
  const dots = D.slots.map((slot, s) => {
    const [x, y] = SLOT_XY[slot];
    const isNat = natural.includes(s), primary = natural[0] === s;
    const fill = ratings ? `var(${level(ratings[s])})` : isNat ? "var(--p3)" : "var(--p0)";
    const ring = isNat ? `<circle cx="${x}" cy="${y}" r="11" fill="none" stroke="var(--ring)" stroke-width="${primary ? 2.5 : 1.2}"/>` : "";
    const label = ratings ? `<text x="${x}" y="${y - 15}" text-anchor="middle" fill="var(--pitch-text)" font-family="var(--display)" font-size="14" font-weight="600">${ratings[s]}</text>` : "";
    return `${ring}<circle cx="${x}" cy="${y}" r="8" fill="${fill}"/>${label}`;
  }).join("");
  const lines = `<rect x="4" y="4" width="192" height="304" rx="6" fill="var(--pitch)"/>
    <g fill="none" stroke="var(--pitch-line)" stroke-width="1.2">
      <rect x="10" y="10" width="180" height="292" rx="2"/><line x1="10" y1="156" x2="190" y2="156"/><circle cx="100" cy="156" r="24"/>
      <rect x="55" y="10" width="90" height="38"/><rect x="78" y="10" width="44" height="14"/>
      <rect x="55" y="264" width="90" height="38"/><rect x="78" y="288" width="44" height="14"/></g>`;
  return `<section class="pitchbox"><h3>${esc(T.pitch)}</h3>
    <svg viewBox="0 0 200 312" role="img" aria-label="${esc(T.pitch)}">${lines}${dots}</svg>
    <p class="note">${esc(ratings ? T.pitchNote : T.pitchNoteNoEA)}</p></section>`;
}

function scaleSvg(p) {
  const W = 640, x = v => 20 + (Math.min(Math.max(v, 40), 100) - 40) / 60 * (W - 40);
  const cur = Math.round(p.cur), fut = Math.round(p.fut[H]);
  const ticks = [40, 50, 60, 70, 80, 90, 100].map(t =>
    `<line x1="${x(t)}" y1="30" x2="${x(t)}" y2="36" stroke="var(--line)"/><text x="${x(t)}" y="52" text-anchor="middle">${t}</text>`).join("");
  const bandW = Math.max(x(p.hi[H]) - x(p.lo[H]), 2);
  const close = Math.abs(x(fut) - x(cur)) < 90;
  return `<svg viewBox="0 0 ${W} 60" role="img" aria-label="${esc(T.nowShort)} ${cur}, ${esc(fmt(T.inYears, {n: H}))} ${fut}">
    <rect x="20" y="30" width="${W - 40}" height="6" rx="3" fill="var(--soft)"/>
    <rect x="${x(p.lo[H])}" y="26" width="${bandW}" height="14" rx="4" fill="var(--band)"/>
    ${ticks}
    <g class="mk"><circle cx="${x(cur)}" cy="33" r="7" fill="var(--accent)"/><text x="${x(cur)}" y="${close ? 0 : 16}" text-anchor="middle" fill="var(--accent)">${esc(T.nowShort)} ${cur}</text></g>
    <g class="mk"><circle cx="${x(fut)}" cy="33" r="7" fill="var(--surface)" stroke="var(--gold)" stroke-width="3"/><text x="${x(fut)}" y="16" text-anchor="middle" fill="var(--gold)">${esc(fmt(T.inYears, {n: H}))} ${fut}</text></g>
  </svg>`;
}

// 기량 흐름: 지금부터 5년 뒤까지 예상 기량과 80% 범위
function trendSvg(p) {
  const W = 640, Hh = 150, left = 34, right = 14, top = 14, bottom = 30;
  const pts = [[0, p.cur, p.cur, p.cur]].concat([1, 2, 3, 4, 5].map(h => [h, p.fut[h], p.lo[h], p.hi[h]]));
  const lo = Math.floor(Math.min(...pts.map(q => q[2])) / 5) * 5, hi = Math.ceil(Math.max(...pts.map(q => q[3])) / 5) * 5;
  const x = h => left + h / 5 * (W - left - right), y = v => top + (hi - v) / Math.max(hi - lo, 5) * (Hh - top - bottom);
  const band = pts.map(q => `${x(q[0])},${y(q[3])}`).join(" ") + " " + pts.slice().reverse().map(q => `${x(q[0])},${y(q[2])}`).join(" ");
  const line = pts.map(q => `${x(q[0])},${y(q[1])}`).join(" ");
  const grid = [];
  for (let v = lo; v <= hi; v += 5) grid.push(`<line x1="${left}" y1="${y(v)}" x2="${W - right}" y2="${y(v)}" stroke="var(--line)" stroke-width="1"/><text x="${left - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`);
  const labels = pts.map(q => `<text x="${x(q[0])}" y="${Hh - 8}" text-anchor="middle">${q[0] === 0 ? esc(T.nowShort) : esc(fmt(T.years, {n: q[0]}))}</text>`).join("");
  const dots = pts.map(q => q[0] === 0
    ? `<circle cx="${x(0)}" cy="${y(q[1])}" r="5" fill="var(--accent)"/>`
    : `<circle cx="${x(q[0])}" cy="${y(q[1])}" r="${q[0] === H ? 6 : 3.5}" fill="${q[0] === H ? "var(--surface)" : "var(--gold)"}" stroke="var(--gold)" stroke-width="${q[0] === H ? 3 : 0}"/>`).join("");
  const sel = pts[H];
  return `<section class="trend"><h3>${esc(T.trend)}</h3>
    <svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="${esc(T.trend)}">${grid.join("")}
      <polygon points="${band}" fill="var(--band)"/>
      <polyline points="${line}" fill="none" stroke="var(--gold)" stroke-width="2.5" stroke-linejoin="round"/>
      ${dots}${labels}
      <text x="${x(H)}" y="${y(sel[1]) - 12}" text-anchor="middle" fill="var(--gold)" font-weight="600">${Math.round(sel[1])}</text>
    </svg><p class="note">${esc(T.trendNote)}</p></section>`;
}

function horizonHtml() {
  const buttons = [1, 2, 3, 4, 5].map(h => `<button type="button" role="radio" aria-checked="${h === H}" data-h="${h}">${esc(fmt(T.years, {n: h}))}</button>`).join("");
  return `<div class="horizon"><span class="lbl" id="h-label">${esc(T.yearsLabel)}</span><div class="seg" role="radiogroup" aria-labelledby="h-label">${buttons}</div></div>`;
}

function renderCard(i) {
  const p = player(i);
  const cur = Math.round(p.cur), fut = Math.round(p.fut[H]);
  const g = fut - cur;                                   // 화면에 보이는 두 숫자의 차이로 계산한다
  const gCls = g >= 1 ? "up" : g <= -1 ? "down" : "flat";
  const tNow = tierIndex(cur);
  const pos = T.pos[p.main] || p.main;
  const ageTxt = fmt(T.age, {a: Math.floor(p.age)}) + (p.born ? " · " + fmt(T.born, {y: p.born}) : "");
  const season = p.games == null ? "" : fmt(T.season, {g: p.games, s: p.starts, go: p.goals, as: p.assists});
  document.getElementById("card").innerHTML = `<div class="card-grid">
    <aside class="side">${photoHtml(p)}${pitchHtml(p)}</aside>
    <div class="mainbox">
      <h2>${esc(p.name)}</h2>
      <p class="meta"><strong>${esc(ageTxt)}</strong> · ${esc(pos)}${p.detail && p.detail !== p.main ? " (" + esc(p.detail) + ")" : ""}</p>
      <p class="meta">${esc(p.club || T.noClub)}${p.league ? " · " + esc(p.league) : ""}${p.nation ? " · " + esc(p.nation) : ""}</p>
      ${season ? `<p class="meta">${esc(season)}</p>` : ""}
      <div class="money"><span class="lbl">${esc(T.value)}</span>
        <div class="vals"><span class="v">${euro(p.value)}</span><span class="v">${pound(p.value)}</span><span class="v">${won(p.value)}</span></div>
        <span class="note">${esc(fmt(T.valueNote, {d: D.fx.date || "-"}))}</span></div>
      ${horizonHtml()}
      <div class="scores">
        <div class="score now">
          <span class="label">${esc(T.now)}</span>
          <span class="big"><span class="num">${cur}</span><span class="of">${esc(T.of)}</span><span class="tier" style="color: var(${TIER_CUTS[tNow][1]})">${esc(T.tiers[tNow])}</span></span>
          <span class="sub">${esc(fmt(T.topAll, {p: topShare(p.curPct)}))}</span>
        </div>
        <div class="score future">
          <span class="label">${esc(fmt(T.future, {n: H}))}</span>
          <span class="big"><span class="num">${fut}</span><span class="of">${esc(T.of)}</span><span class="delta ${gCls}">${g > 0 ? "+" : ""}${g}</span><span class="tier" style="color: var(--${gCls === "flat" ? "t6" : gCls})">${esc(T[gCls])}</span></span>
          <span class="sub">${esc(fmt(T.range, {lo: Math.round(p.lo[H]), hi: Math.round(p.hi[H])}))} · ${esc(fmt(T.topAge, {p: topShare(p.pct[H])}))}</span>
        </div>
      </div>
      <div class="scale">${scaleSvg(p)}</div>
      ${trendSvg(p)}
      <div class="ref">
        <span>${esc(T.ea25)}<b>${p.eaPrev ?? "-"}</b></span>
        <span>${esc(T.eaPot)}<b>${p.eaPot ?? "-"}</b></span>
        <span>${esc(T.ea26)}<b>${p.eaNow ?? "-"}</b></span>
      </div>
    </div></div>`;
  current = i;
  document.querySelectorAll(".result").forEach(el => el.setAttribute("aria-current", String(+el.dataset.i === i)));
}

function renderResults(list, query) {
  const ol = document.getElementById("results");
  if (!query) { ol.innerHTML = `<li class="count">${esc(T.listHint)}</li>`; return; }
  if (!list.length) { ol.innerHTML = `<li class="empty">${esc(fmt(T.noMatch, {q: query}))}</li>`; return; }
  const n = list.length >= 50 ? T.countMany : fmt(T.countN, {n: list.length});
  ol.innerHTML = `<li class="count">${esc(fmt(T.count, {n}))}</li>` + list.map(i => {
    const p = player(i);
    return `<li class="result" tabindex="0" data-i="${i}" aria-current="${i === current}">
      <span class="nm">${esc(p.name)}</span>
      <span class="mini"><span>${Math.round(p.cur)}</span><span class="arrow">→</span><span class="f">${Math.round(p.fut[H])}</span></span>
      <span class="sub">${esc(fmt(T.age, {a: Math.floor(p.age)}))} · ${esc(p.club || "-")}</span>
    </li>`;
  }).join("");
}

let lastList = [];
function onSearch() {
  const q = document.getElementById("q").value;
  lastList = search(q, 50);
  renderResults(lastList, q.trim());
  if (lastList.length) renderCard(lastList[0]);
}

// 유망주 순위
function renderRank() {
  const maxAge = +document.getElementById("f-age").value;
  const pos = document.getElementById("f-pos").value;
  const sort = document.getElementById("f-sort").value;
  const f = D["fut" + H];
  const idx = [];
  for (let i = 0; i < N; i++) {
    if (D.age[i] / 10 > maxAge) continue;
    if (pos && D.mainList[D.main[i]] !== pos) continue;
    idx.push(i);
  }
  const key = sort === "cur" ? i => D.cur[i] : sort === "gain" ? i => f[i] - D.cur[i] : i => f[i];
  idx.sort((a, b) => key(b) - key(a) || f[b] - f[a]);
  document.getElementById("rank-body").innerHTML = idx.slice(0, 100).map((i, r) => {
    const p = player(i);
    return `<tr tabindex="0" data-i="${i}">
      <td class="rk">${r + 1}</td><td>${esc(p.name)}</td><td>${p.age.toFixed(1)}</td><td>${esc(T.pos[p.main] || "")}</td>
      <td>${esc(p.club || "-")}</td><td class="n c">${Math.round(p.cur)}</td><td class="n f">${Math.round(p.fut[H])}</td>
      <td>${Math.round(p.lo[H])}~${Math.round(p.hi[H])}</td><td>${euro(p.value)}</td></tr>`;
  }).join("");
}

// 전체 랭킹: 리그·포지션·나이대로 거르고 100명씩 보여준다
const ALL_PAGE = 100;
let allShown = ALL_PAGE;
function renderAll(keepCount) {
  if (!keepCount) allShown = ALL_PAGE;
  const league = document.getElementById("a-league").value;
  const pos = document.getElementById("a-pos").value;
  const band = document.getElementById("a-age").value;
  const sort = document.getElementById("a-sort").value;
  const [amin, amax] = band ? band.split("-").map(Number) : [0, 99];
  const f = D["fut" + H];
  const idx = [];
  for (let i = 0; i < N; i++) {
    const age = Math.floor(D.age[i] / 10);
    if (age < amin || age > amax) continue;
    if (league !== "" && D.league[i] !== +league) continue;
    if (pos && D.mainList[D.main[i]] !== pos) continue;
    idx.push(i);
  }
  const key = sort === "fut" ? i => f[i] : sort === "gain" ? i => f[i] - D.cur[i] : i => D.cur[i];
  idx.sort((a, b) => key(b) - key(a) || D.cur[b] - D.cur[a]);
  const shown = idx.slice(0, allShown);
  document.getElementById("all-body").innerHTML = shown.map((i, r) => {
    const p = player(i);
    return `<tr tabindex="0" data-i="${i}">
      <td class="rk">${r + 1}</td><td>${esc(p.name)}</td><td>${Math.floor(p.age)}</td><td>${esc(T.pos[p.main] || "")}</td>
      <td>${esc(p.club || "-")}</td><td class="lg">${esc((p.league || "-").split(" · ")[0])}</td>
      <td class="n c">${Math.round(p.cur)}</td><td class="n f">${Math.round(p.fut[H])}</td><td>${euro(p.value)}</td></tr>`;
  }).join("");
  document.getElementById("a-count").textContent = fmt(T.shown, {total: idx.length.toLocaleString(), k: shown.length.toLocaleString()});
  document.getElementById("a-more").hidden = shown.length >= idx.length;
}

function fillSortLabels() {
  for (const [id, fallback] of [["f-sort", "fut"], ["a-sort", "cur"]]) {
    const sel = document.getElementById(id), v = sel.value || fallback;
    sel.innerHTML = Object.entries(SORT_LABELS).map(([k, label]) => `<option value="${k}">${esc(label.replace("N", H))}</option>`).join("");
    sel.value = v;
  }
}

function setYears(h) {
  H = h;
  try { localStorage.setItem("chipscouting-years", String(h)); } catch (e) {}
  document.getElementById("f-years").value = String(h);
  document.getElementById("a-years").value = String(h);
  fillSortLabels();
  if (current != null) renderCard(current);
  renderResults(lastList, document.getElementById("q").value.trim());
  if (!document.getElementById("view-rank").hidden) renderRank();
  if (!document.getElementById("view-all").hidden) renderAll(true);
}

function showTab(name) {
  for (const v of ["search", "all", "rank"]) {
    document.getElementById("view-" + v).hidden = v !== name;
    document.getElementById("tab-" + v).setAttribute("aria-selected", String(v === name));
  }
  if (name === "rank") renderRank();
  if (name === "all") renderAll();
}

function openPlayer(i) {
  document.getElementById("q").value = D.name[i];
  showTab("search");
  onSearch();
  renderCard(i);
}

// 화면 연결
document.getElementById("examples").insertAdjacentHTML("beforeend",
  EXAMPLES.map(n => `<button class="chip" type="button" data-q="${esc(n)}">${esc(n)}</button>`).join(""));
document.getElementById("examples").addEventListener("click", e => {
  const b = e.target.closest(".chip"); if (!b) return;
  document.getElementById("q").value = b.dataset.q; onSearch();
});
document.getElementById("q").addEventListener("input", onSearch);
document.getElementById("results").addEventListener("click", e => { const li = e.target.closest(".result"); if (li) renderCard(+li.dataset.i); });
document.getElementById("results").addEventListener("keydown", e => { const li = e.target.closest(".result"); if (li && e.key === "Enter") renderCard(+li.dataset.i); });
document.getElementById("card").addEventListener("click", e => { const b = e.target.closest(".seg button"); if (b) setYears(+b.dataset.h); });
document.getElementById("rank-body").addEventListener("click", e => { const tr = e.target.closest("tr"); if (tr) openPlayer(+tr.dataset.i); });
document.getElementById("rank-body").addEventListener("keydown", e => { const tr = e.target.closest("tr"); if (tr && e.key === "Enter") openPlayer(+tr.dataset.i); });
document.getElementById("tab-search").addEventListener("click", () => showTab("search"));
document.getElementById("tab-all").addEventListener("click", () => showTab("all"));
document.getElementById("tab-rank").addEventListener("click", () => showTab("rank"));
document.getElementById("all-body").addEventListener("click", e => { const tr = e.target.closest("tr"); if (tr) openPlayer(+tr.dataset.i); });
document.getElementById("all-body").addEventListener("keydown", e => { const tr = e.target.closest("tr"); if (tr && e.key === "Enter") openPlayer(+tr.dataset.i); });
document.getElementById("a-more").addEventListener("click", () => { allShown += ALL_PAGE; renderAll(true); });
for (const id of ["f-years", "a-years"]) {
  const sel = document.getElementById(id);
  for (let h = 1; h <= 5; h++) sel.insertAdjacentHTML("beforeend", `<option value="${h}">${esc(fmt(T.years, {n: h}))}</option>`);
  sel.addEventListener("change", () => setYears(+sel.value));
}
// 리그 목록: 선수 20명 이상인 리그를, 상위 20명의 평균 현재 기량이 높은 순으로
// (전체 평균은 데이터에 일부 선수만 들어 있는 리그가 부풀려져서 상위 20명을 쓴다)
const leagueCur = new Map();
for (let i = 0; i < N; i++) {
  if (!D.leagueList[D.league[i]]) continue;
  if (!leagueCur.has(D.league[i])) leagueCur.set(D.league[i], []);
  leagueCur.get(D.league[i]).push(D.cur[i]);
}
const top20 = v => v.slice().sort((a, b) => b - a).slice(0, 20).reduce((s, x) => s + x, 0);
const leagueSel = document.getElementById("a-league");
[...leagueCur.entries()].filter(([, v]) => v.length >= 20).sort((a, b) => top20(b[1]) - top20(a[1]))
  .forEach(([k, v]) => leagueSel.insertAdjacentHTML("beforeend", `<option value="${k}">${esc(D.leagueList[k])} (${v.length})</option>`));
const bandSel = document.getElementById("a-age");
Object.entries(T.ageBands).forEach(([k, v]) => bandSel.insertAdjacentHTML("beforeend", `<option value="${k}">${esc(v)}</option>`));
const aPosSel = document.getElementById("a-pos");
Object.entries(T.pos).forEach(([k, v]) => aPosSel.insertAdjacentHTML("beforeend", `<option value="${k}">${esc(v)}</option>`));
["a-league", "a-pos", "a-age", "a-sort"].forEach(id => document.getElementById(id).addEventListener("change", () => renderAll()));
const ageSel = document.getElementById("f-age");
for (let a = 17; a <= 25; a++) ageSel.insertAdjacentHTML("beforeend", `<option value="${a}"${a === 21 ? " selected" : ""}>${a}</option>`);
const posSel = document.getElementById("f-pos");
Object.entries(T.pos).forEach(([k, v]) => posSel.insertAdjacentHTML("beforeend", `<option value="${k}">${esc(v)}</option>`));
["f-age", "f-pos", "f-sort"].forEach(id => document.getElementById(id).addEventListener("change", renderRank));
document.getElementById("f-years").value = String(H);
document.getElementById("a-years").value = String(H);
fillSortLabels();

// 처음 화면: 예시 선수 한 명의 카드
renderResults([], "");
const first = search("Lamine Yamal", 1)[0];
renderCard(first ?? 0);
if (location.hash === "#rank") showTab("rank");
if (location.hash === "#all") showTab("all");
</script>
"""

LOCAL_HEAD = """<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}body{{margin:0;font:14px system-ui,sans-serif}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
"""


def render(lang, payload, other_href, has_photos):
    """has_photos: 사진이 한 장도 없으면 아래쪽 설명에서 사진 이야기를 뺀다."""
    text = TEXT[lang]
    other = "en" if lang == "ko" else "ko"
    page = PAGE
    for key, value in text.items():
        if key in ("th", "th_all"):
            value = "".join(f"<th>{v}</th>" for v in value)
        elif key == "footer":
            note, src = (text["photo_note"], text["photo_src"]) if has_photos else ("", "")
            value = "".join(f"<p>{v.format(photo_note=note, photo_src=src)}</p>" for v in value)
        page = page.replace("{{" + key + "}}", value)
    page = page.replace("{{other_href}}", other_href).replace("{{other_code}}", other)
    sorts = {"fut": text["sort_fut"], "cur": text["sort_cur"], "gain": text["sort_gain"]}
    page = page.replace("__TEXT__", json.dumps(JS_TEXT[lang], ensure_ascii=False))
    page = page.replace("__SORTS__", json.dumps(sorts, ensure_ascii=False))
    return page.replace("__DATA__", payload)


def local_file(page, lang):
    """아티팩트가 자동으로 붙여 주는 문서 머리 부분을 직접 붙여, 브라우저로 바로 여는 파일로 만든다."""
    head, body = page.split("<header", 1)
    return LOCAL_HEAD.format(lang=lang) + head + "</head>\n<body>\n<header" + body + "</body>\n</html>\n"


def payload_for(data, player_ids, mode):
    """사진 부분만 판에 맞게 바꿔서 페이지에 넣을 JSON 글자를 만든다."""
    data = dict(data)
    data["photo"], data["photos"] = photo_arrays(player_ids, mode)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return text, len(data["photos"])


def main():
    data, player_ids = build_data()
    WEB.mkdir(exist_ok=True)
    local_payload, n_local = payload_for(data, player_ids, "local")
    artifact_payload, n_artifact = payload_for(data, player_ids, "artifact")
    for lang, local_name in [("ko", "index.html"), ("en", "en.html")]:
        other = "en" if lang == "ko" else "ko"
        artifact_page = render(lang, artifact_payload, ARTIFACT_URL[other] or ("en.html" if lang == "ko" else "index.html"), n_artifact > 0)
        (WEB / f"chipscouting_{lang}.html").write_text(artifact_page, encoding="utf-8")
        local_page = render(lang, local_payload, "en.html" if lang == "ko" else "index.html", n_local > 0)
        (WEB / local_name).write_text(local_file(local_page, lang), encoding="utf-8")
    print(f"선수 {len(player_ids):,}명을 넣은 페이지를 만들었습니다")
    local_photos = f"사진 {n_local:,}명, 위키미디어에서 바로 불러옴" if n_local else "사진 없음 (이름 머리글자로 표시)"
    artifact_photos = f"사진 {n_artifact}명, 함께 올린 사본을 씀" if n_artifact else "사진 없음"
    print(f"  {WEB / 'index.html'} (한국어), {WEB / 'en.html'} (영어): 브라우저로 열기. "
          f"{local_photos} ({(WEB / 'index.html').stat().st_size / 1e6:.1f}MB)")
    print(f"  {WEB / 'chipscouting_ko.html'}, {WEB / 'chipscouting_en.html'}: 아티팩트용. "
          f"{artifact_photos} ({(WEB / 'chipscouting_ko.html').stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
