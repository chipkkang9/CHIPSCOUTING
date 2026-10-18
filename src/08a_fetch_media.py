"""8a단계: 웹 페이지에 넣을 환율과 (고르면) 선수 사진 정보를 가져온다.

사진
  보도 사진(경기 중 사진)은 대부분 사진 회사의 저작물이라 마음대로 쓸 수 없다. 그래서 누구나 출처만
  밝히면 쓸 수 있는 위키미디어 공용(Wikimedia Commons)의 자유 라이선스 사진을 쓴다.
  1. 위키데이터에서 "Transfermarkt 선수 ID"(P2446)와 대표 사진(P18)이 둘 다 있는 선수를 모두 찾는다.
  2. 위키미디어 공용에서 그 사진의 작은 판(가로 400px) 주소와 촬영자, 라이선스를 얻는다.
  사진 파일은 받지 않고 주소만 저장한다. 내 컴퓨터에서 여는 페이지는 이 주소에서 사진을 바로 불러온다.

  claude.ai 페이지는 외부 사이트의 이미지를 막아 두어서, 사진을 페이지와 함께 올려야 한다.
  그때만 --download로 사진 파일을 받는다 (올릴 수 있는 파일 수 한도 때문에 잘 알려진 선수 약 250명).

환율
  유럽중앙은행(ECB)이 매일 공개하는 기준 환율에서 유로 대비 파운드, 원 환율을 받는다.

실행:  python src/08a_fetch_media.py              환율만
       python src/08a_fetch_media.py --photos     환율과 사진 정보 (사진 파일은 받지 않는다)
       python src/08a_fetch_media.py --download   claude.ai에 올릴 사진 파일도 web/photos/에 받는다 (--photos 포함)
출력:  data/processed/photos.json      선수별 사진 주소, 촬영자, 라이선스
       data/processed/fx_rates.json    환율
"""
import argparse
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

from common import ROOT

SCOUT_TABLE = ROOT / "data" / "processed" / "scout_2025.parquet"
PHOTO_META = ROOT / "data" / "processed" / "photos.json"
ARTIFACT_PHOTOS = ROOT / "data" / "processed" / "photos_artifact.json"   # claude.ai에 올린 사진의 선수 ID
PHOTO_DIR = ROOT / "web" / "photos"
FX_FILE = ROOT / "data" / "processed" / "fx_rates.json"
USER_AGENT = "CHIPSCOUTING-personal-study/0.1 (football scouting study project)"
MAX_ARTIFACT_PHOTOS = 250     # 아티팩트 한 번에 올릴 수 있는 파일 수(255개) 안에 맞춘다
FREE_LICENSES = ("CC BY", "CC0", "Public domain", "PD")


def get(url, params=None, accept=None, post=False, retries=2):
    """웹에서 받아 온다. 요청이 너무 잦다는 응답(429)이면 1분 쉬고 다시 시도한다."""
    data = None
    if params and post:
        data = urllib.parse.urlencode(params).encode()
    elif params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": USER_AGENT, **({"Accept": accept} if accept else {})}
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=180) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries:
                raise
            print("  요청이 너무 잦다는 응답이라 65초 기다립니다...")
            time.sleep(65)


# ---------------------------------------------------------------------------
# 환율
# ---------------------------------------------------------------------------
def fetch_fx():
    xml = get("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml")
    root = ET.fromstring(xml)
    rates, date = {}, None
    for cube in root.iter():
        if cube.get("time"):
            date = cube.get("time")
        if cube.get("currency") in ("GBP", "KRW"):
            rates[cube.get("currency")] = float(cube.get("rate"))
    fx = {"date": date, "source": "European Central Bank euro foreign exchange reference rates", **rates}
    FX_FILE.write_text(json.dumps(fx, indent=2))
    print(f"환율 ({date}): 1유로 = {rates['GBP']}파운드 = {rates['KRW']}원")


# ---------------------------------------------------------------------------
# 사진 정보
# ---------------------------------------------------------------------------
def wikidata_images():
    """위키데이터에서 Transfermarkt ID와 대표 사진이 둘 다 있는 선수를 모두 가져온다: ID -> 사진 파일 이름.

    위키데이터 질의 서버가 1분에 한 번만 받아 주는 때가 있어서, 선수를 하나하나 묻지 않고
    "둘 다 있는 항목 전부"를 질의 한 번으로 받은 뒤 우리 선수만 고른다.
    """
    query = "SELECT ?tm ?image WHERE { ?item wdt:P2446 ?tm . ?item wdt:P18 ?image . }"
    raw = get("https://query.wikidata.org/sparql", {"query": query}, accept="text/csv", post=True)
    table = pd.read_csv(io.BytesIO(raw), dtype=str)
    table = table[table["tm"].str.fullmatch(r"\d+")]
    table["tm"] = table["tm"].astype(int)
    table["name"] = table["image"].str.split("Special:FilePath/").str[-1].map(urllib.parse.unquote)
    return table.drop_duplicates("tm").set_index("tm")["name"]     # 사진이 여러 장이면 첫 번째


def strip_html(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def commons_info(filenames):
    """파일 이름 -> 작은 판 주소, 촬영자, 라이선스. 50개씩 묶어서 묻는다."""
    out = {}
    names = sorted(filenames)
    for i in range(0, len(names), 50):
        titles = "|".join("File:" + n for n in names[i:i + 50])
        data = json.loads(get("https://commons.wikimedia.org/w/api.php", {
            "action": "query", "titles": titles, "prop": "imageinfo", "format": "json",
            "iiprop": "url|extmetadata", "iiurlwidth": 400,
            "iiextmetadatafilter": "Artist|LicenseShortName"}, post=True))
        normalized = {n["to"]: n["from"] for n in data["query"].get("normalized", [])}
        for page in data["query"]["pages"].values():
            if "imageinfo" not in page:
                continue
            info = page["imageinfo"][0]
            meta = info.get("extmetadata", {})
            title = normalized.get(page["title"], page["title"])
            out[title.removeprefix("File:")] = {
                "thumb": info.get("thumburl"),
                "author": strip_html(meta.get("Artist", {}).get("value", ""))[:80],
                "license": strip_html(meta.get("LicenseShortName", {}).get("value", "")),
            }
        if i // 50 % 20 == 0:
            print(f"  위키미디어 공용 정보 {min(i + 50, len(names)):,}/{len(names):,}")
        time.sleep(0.3)
    return out


def fetch_photo_meta():
    players = set(pd.read_parquet(SCOUT_TABLE, columns=["player_id"])["player_id"].astype(int))
    images = wikidata_images()
    images = images[images.index.isin(players)]
    print(f"검색 대상 선수 {len(players):,}명 중 위키데이터에 대표 사진이 있는 선수: {len(images):,}명")
    info = commons_info(set(images))
    meta, skipped = {}, 0
    for tm_id, name in images.items():
        m = info.get(name)
        if not m or not m["thumb"] or not m["license"].startswith(FREE_LICENSES):
            skipped += 1
            continue
        meta[str(tm_id)] = {"name": name, **m}
    PHOTO_META.write_text(json.dumps(meta, ensure_ascii=False, indent=0))
    print(f"사진 정보 저장: {len(meta):,}명 | 정보가 없거나 라이선스가 맞지 않아 건너뜀: {skipped}명")
    print("라이선스별:", pd.Series([m["license"] for m in meta.values()]).value_counts().head(8).to_dict())
    return meta


# ---------------------------------------------------------------------------
# claude.ai에 올릴 사진 파일 받기 (--download)
# ---------------------------------------------------------------------------
def pick_artifact_players():
    """claude.ai에 올릴 사진의 선수: 현재 기량 상위 170명 + 21세 이하 3년 뒤 상위 60명 + 한국 선수 상위 25명."""
    t = pd.read_parquet(SCOUT_TABLE)
    groups = [
        t.nlargest(170, "current_ability"),
        t[t["age"] <= 21].nlargest(60, "future_ability"),
        t[t["citizenship"].fillna("").str.contains("Korea, South")].nlargest(25, "current_ability"),
    ]
    return list(dict.fromkeys(int(i) for g in groups for i in g["player_id"]))   # 순서를 지키며 중복 제거


def download_artifact_photos(meta):
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    chosen = [i for i in pick_artifact_players() if str(i) in meta][:MAX_ARTIFACT_PHOTOS]
    for tm_id in chosen:
        path = PHOTO_DIR / f"{tm_id}.jpg"
        if not path.exists():
            path.write_bytes(get(meta[str(tm_id)]["thumb"]))
            time.sleep(0.2)
    ARTIFACT_PHOTOS.write_text(json.dumps({"player_ids": sorted(chosen)}, indent=1))
    size = sum(p.stat().st_size for p in PHOTO_DIR.iterdir()) / 1e6
    print(f"claude.ai에 올릴 사진 {len(chosen)}장을 {PHOTO_DIR}에 받았습니다 ({size:.1f}MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", action="store_true", help="선수 사진 주소와 출처도 받는다 (위키데이터, 위키미디어 공용)")
    ap.add_argument("--download", action="store_true", help="claude.ai에 올릴 사진 파일도 받는다 (--photos 포함)")
    args = ap.parse_args()
    fetch_fx()
    if args.photos or args.download:
        meta = fetch_photo_meta()
        if args.download:
            download_artifact_photos(meta)


if __name__ == "__main__":
    main()
