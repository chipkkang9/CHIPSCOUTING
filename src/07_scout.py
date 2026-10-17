"""7단계: 선수 검색. 이름을 넣으면 현재 기량과 성장 가능성을 1~100 점수로 보여준다.

06_ability_models.py가 만든 data/processed/scout_2025.parquet (2025년 7월 1일 기준 전 선수 점수)를 읽어서
찾기만 한다. 모델을 다시 돌리지 않으므로 바로 나온다.

  현재 기량    지금 얼마나 잘하는가. EA FC 평점과 같은 1~99 척도
  성장 가능성  3년 뒤(2028년 여름) 예상 기량. 같은 척도. 80% 범위를 함께 보여준다

실행 예:
  python src/07_scout.py "Lamine Yamal"
  python src/07_scout.py son heung
  python src/07_scout.py --top 20 --max-age 21          # 3년 뒤 예상 기량 상위 유망주
  python src/07_scout.py --top 20 --max-age 23 --position Defender
"""
import argparse
import importlib
import math

import pandas as pd

from common import ROOT

SCOUT_TABLE = ROOT / "data" / "processed" / "scout_2025.parquet"
normalize_name = importlib.import_module("01b_add_fbref").normalize_name
POSITION_KO = {"Attack": "공격수", "Midfield": "미드필더", "Defender": "수비수", "Goalkeeper": "골키퍼"}


def missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def money(v):
    if missing(v):
        return "-"
    return f"€{v / 1e6:.1f}M" if v >= 1e6 else f"€{v / 1e3:.0f}K"


def score(v):
    return "-" if missing(v) else f"{v:.0f}"


def search(table, query):
    """검색어의 모든 단어로 시작하는 단어가 이름에 들어 있는 선수를 찾는다. ('son heung' -> Heung-min Son)"""
    words = normalize_name(query).split()
    if not words:
        return table.iloc[0:0]
    name_words = table["name"].map(lambda n: normalize_name(n).split())
    hit = name_words.map(lambda ws: all(any(w.startswith(q) for w in ws) for q in words))
    return table[hit].sort_values("current_ability", ascending=False)


def card(p):
    """선수 한 명의 점수 카드."""
    pos = POSITION_KO.get(p["main_position"], p["main_position"])
    detail = p["position"].split(" - ")[-1] if isinstance(p["position"], str) else ""
    growth = round(p["growth"])                      # -0 같은 표시가 나오지 않게 정수로 반올림
    trend = "상승 예상" if growth >= 1 else ("하락 예상" if growth <= -1 else "유지 예상")
    top_share = max(100 - p["current_pct"], 0.1)
    lines = [
        "─" * 64,
        f" {p['name']}  ({p['age']:.0f}세, {pos} {detail})",
        f" 소속 {p['club_name'] or '-'} ({p['league'] or '-'}) | 국적 {p['citizenship'] or '-'} | 몸값 {money(p['value_now'])}",
        f" 지난 시즌 {p['games_1s']:.0f}경기 (선발 {p['starts_1s']:.0f}) {p['goals_1s']:.0f}골 {p['assists_1s']:.0f}도움",
        "─" * 64,
        f"  현재 기량    {score(p['current_ability']):>3} / 100   전체 선수 중 상위 {top_share:.1f}%",
        f"  성장 가능성  {score(p['future_ability']):>3} / 100   3년 뒤 예상 기량 ({growth:+d}점, {trend})",
        f"               80% 범위 {score(p['future_low'])} ~ {score(p['future_high'])}"
        f" | 같은 나이 선수 중 상위 {max(100 - p['future_pct_age'], 0.1):.1f}%",
        "─" * 64,
        f"  참고: EA FC 25 (2025년 6월) {score(p['ea_overall_prev'])}, EA 잠재력 {score(p['ea_potential_prev'])}"
        f" | EA FC 26 (2025년 9월) {score(p['ability_now'])}",
    ]
    return "\n".join(lines)


def top_list(table, n, max_age, position):
    t = table[table["age"] <= max_age]
    if position:
        t = t[t["main_position"] == position]
    t = t.nlargest(n, "future_ability")
    out = pd.DataFrame({
        "이름": t["name"], "나이": t["age"].round(1), "포지션": t["main_position"].map(POSITION_KO),
        "소속": t["club_name"], "현재 기량": t["current_ability"].round(0).astype(int),
        "성장 가능성": t["future_ability"].round(0).astype(int),
        "80% 범위": t["future_low"].round(0).astype(int).astype(str) + "~" + t["future_high"].round(0).astype(int).astype(str),
        "몸값": t["value_now"].map(money),
    })
    out.index = range(1, len(out) + 1)
    return out.to_string()


def main():
    ap = argparse.ArgumentParser(description="CHIPSCOUTING 선수 검색 (2025년 7월 기준)")
    ap.add_argument("query", nargs="*", help="선수 이름 (일부만 써도 된다)")
    ap.add_argument("--top", type=int, help="3년 뒤 예상 기량 상위 N명을 보여준다")
    ap.add_argument("--max-age", type=float, default=21, help="--top과 함께: 나이 상한 (기본 21)")
    ap.add_argument("--position", choices=list(POSITION_KO), help="--top과 함께: 포지션")
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    table = pd.read_parquet(SCOUT_TABLE)

    if args.top:
        label = POSITION_KO.get(args.position, "전체 포지션")
        print(f"{args.max_age:.0f}세 이하, {label}, 3년 뒤 예상 기량 상위 {args.top}명 (2025년 7월 기준)\n")
        print(top_list(table, args.top, args.max_age, args.position))
        return
    if not args.query:
        ap.print_help()
        return

    found = search(table, " ".join(args.query))
    if found.empty:
        print("찾는 선수가 없습니다. 이름 일부만 넣어 보세요. (2025년 7월 기준 최근 1년 안에 몸값이 매겨진 40세 이하 선수만 있습니다)")
        return
    for _, p in found.head(3).iterrows():
        print(card(p))
    if len(found) > 3:
        print(f"\n이름이 맞는 선수가 {len(found)}명 더 있습니다 (현재 기량 순):")
        rest = found.iloc[3:13]
        for _, p in rest.iterrows():
            print(f"  {p['name']} ({p['age']:.0f}세, {p['club_name']}) 현재 {score(p['current_ability'])}, 성장 {score(p['future_ability'])}")


if __name__ == "__main__":
    main()
