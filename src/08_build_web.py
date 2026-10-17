"""8단계: 선수 검색 웹 페이지 만들기.

06_ability_models.py가 만든 scout_2025.parquet의 점수를 페이지 안에 넣어서, 서버 없이 브라우저만으로
검색되는 HTML 파일 하나를 만든다. 07_scout.py의 터미널 검색과 같은 기능을 화면으로 옮긴 것이다.

실행:  python src/08_build_web.py
출력:  web/index.html          브라우저로 바로 여는 파일 (더블클릭)
       web/chipscouting.html   claude.ai 아티팩트로 올리는 파일 (문서 머리 부분이 없는 판)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT

SCOUT_TABLE = ROOT / "data" / "processed" / "scout_2025.parquet"
PERFORMANCES = ROOT / "data" / "raw_salimt" / "player_performances.csv"
WEB = ROOT / "web"
POSITION_ORDER = ["Attack", "Midfield", "Defender", "Goalkeeper"]


def league_names():
    """대회 ID -> 대회 이름. 출전 기록에 가장 많이 적힌 이름을 쓴다. ('GB1' -> 'Premier League')"""
    perf = pd.read_csv(PERFORMANCES, usecols=["competition_id", "competition_name"], low_memory=False)
    return perf.groupby("competition_id")["competition_name"].agg(lambda s: s.value_counts().index[0])


def encode(values):
    """같은 글자가 반복되는 열을 (고유값 목록, 번호 배열)로 바꿔 파일 크기를 줄인다."""
    codes, uniques = pd.factorize(values.fillna(""), sort=True)
    return list(uniques), codes.tolist()


def ints(series, scale=1):
    """숫자 열을 정수 배열로. 빈 값은 null. scale=10이면 소수 첫째 자리까지 보존한다."""
    return [None if pd.isna(v) else int(round(v * scale)) for v in series]


def build_data():
    t = pd.read_parquet(SCOUT_TABLE).dropna(subset=["name"])
    t = t.sort_values("current_ability", ascending=False).reset_index(drop=True)   # 기본 순서 = 현재 기량순
    names = league_names()
    t["league_label"] = t["league"].map(lambda c: f"{names.get(c, c)} · {c}" if isinstance(c, str) and c else "")
    t["detail"] = t["position"].astype(str).str.split(" - ").str[-1]
    t["nation"] = t["citizenship"].fillna("").str.replace(r"\s{2,}", " / ", regex=True)

    data = {"name": t["name"].tolist(), "age": ints(t["age"], 10)}
    data["mainList"], codes = encode(t["main_position"].astype(str))
    data["main"] = codes
    for key, col in [("detail", "detail"), ("club", "club_name"), ("league", "league_label"), ("nation", "nation")]:
        data[key + "List"], data[key] = encode(t[col])
    data.update({
        "value": ints(t["value_now"] / 1000),            # 천 유로 단위
        "games": ints(t["games_1s"]), "starts": ints(t["starts_1s"]),
        "goals": ints(t["goals_1s"]), "assists": ints(t["assists_1s"]),
        "eaPrev": ints(t["ea_overall_prev"]), "eaPot": ints(t["ea_potential_prev"]), "eaNow": ints(t["ability_now"]),
        "cur": ints(t["current_ability"], 10), "fut": ints(t["future_ability"], 10),
        "lo": ints(t["future_low"]), "hi": ints(t["future_high"]),
        "curPct": ints(t["current_pct"], 10), "futPctAge": ints(t["future_pct_age"], 10),
    })
    return data, len(t)


PAGE = r"""<title>CHIPSCOUTING</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap">
<style>
/* 레이아웃: 스카우트 리포트. 왼쪽은 검색 결과, 오른쪽은 선수 카드. 숫자는 경기장 전광판처럼 좁고 큰 글자. */
:root {
  --bg: #f2f4ef;
  --surface: #ffffff;
  --ink: #16221b;
  --muted: #5b685f;
  --line: #d6ddd5;
  --accent: #1c6a44;      /* 현재 기량 */
  --gold: #a96c12;        /* 성장 가능성 */
  --band: rgba(169, 108, 18, 0.16);
  --soft: #e7ece5;
  --up: #1c6a44;
  --down: #a33a2b;
  --t1: #9c5a06; --t2: #1c6a44; --t3: #22707e; --t4: #45618a; --t5: #6a6b7c; --t6: #7d847b;
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
.wrap { max-width: 1120px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }

.top { border-bottom: 1px solid var(--line); background: var(--surface); }
.top .wrap { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px 24px; padding-block: 18px 0; }
.brand h1 { margin: 0; font-family: var(--display); font-weight: 700; font-size: 34px; letter-spacing: 0.04em; line-height: 1; }
.brand p { margin: 6px 0 14px; color: var(--muted); font-size: 13px; }
.tabs { display: flex; gap: 4px; }
.tab { appearance: none; border: 0; background: none; color: var(--muted); font: 600 14px/1 var(--body); padding: 12px 14px 13px; border-bottom: 3px solid transparent; cursor: pointer; }
.tab[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); }
.tab:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible, .result:focus-visible, tr:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

main { padding-block: 22px 40px; }
.searchbar { display: grid; gap: 8px; margin-bottom: 18px; }
.searchbar label { font-size: 13px; color: var(--muted); }
#q { width: 100%; box-sizing: border-box; font: 500 18px/1.3 var(--body); color: var(--ink); background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.examples { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 13px; color: var(--muted); }
.chip { appearance: none; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 999px; padding: 4px 10px; font: 500 13px/1.4 var(--body); cursor: pointer; }
.chip:hover { border-color: var(--accent); }

.split { display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr); gap: 20px; align-items: start; }
@media (max-width: 760px) { .split { grid-template-columns: minmax(0, 1fr); } .results { max-height: 260px; } }
.results { list-style: none; margin: 0; padding: 0; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); max-height: 560px; overflow-y: auto; }
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

.card { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 22px 24px 20px; min-width: 0; }
.card h2 { margin: 0; font-family: var(--display); font-weight: 700; font-size: 40px; line-height: 1.05; letter-spacing: 0.01em; text-wrap: balance; overflow-wrap: anywhere; }
.meta { margin: 8px 0 0; color: var(--muted); font-size: 14px; overflow-wrap: anywhere; }
.meta strong { color: var(--ink); font-weight: 600; }
.scores { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 20px 0 6px; }
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

.scale { margin: 18px 0 4px; }
.scale svg { display: block; width: 100%; height: auto; overflow: visible; }
.scale text { font-family: var(--display); font-size: 13px; fill: var(--muted); }
.scale .mk text { font-family: var(--body); font-size: 12px; font-weight: 600; }
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
tbody tr:hover { background: var(--soft); }

footer { margin-top: 28px; font-size: 12.5px; color: var(--muted); display: grid; gap: 6px; max-width: 72ch; }
footer p { margin: 0; }
@media (prefers-reduced-motion: no-preference) { .card { animation: rise 0.25s ease-out; } @keyframes rise { from { transform: translateY(4px); } to { transform: none; } } }
</style>

<header class="top">
  <div class="wrap">
    <div class="brand">
      <h1>CHIPSCOUTING</h1>
      <p id="stamp">2025년 7월 1일 기준 · 현재 기량과 3년 뒤 예상 기량 (1~100)</p>
    </div>
    <nav class="tabs" role="tablist" aria-label="보기 선택">
      <button class="tab" id="tab-search" role="tab" aria-selected="true" aria-controls="view-search">선수 검색</button>
      <button class="tab" id="tab-rank" role="tab" aria-selected="false" aria-controls="view-rank">유망주 순위</button>
    </nav>
  </div>
</header>

<main class="wrap">
  <section id="view-search" role="tabpanel" aria-labelledby="tab-search">
    <div class="searchbar">
      <label for="q">선수 이름을 영문으로 입력하세요. 일부만 써도 됩니다.</label>
      <input id="q" type="search" placeholder="예: Yamal, Son Heung, Bellingham" autocomplete="off" spellcheck="false">
      <div class="examples" id="examples">예시</div>
    </div>
    <div class="split">
      <ol class="results" id="results" aria-label="검색 결과"></ol>
      <article class="card" id="card" aria-live="polite"></article>
    </div>
  </section>

  <section id="view-rank" role="tabpanel" aria-labelledby="tab-rank" hidden>
    <div class="filters">
      <label for="f-age">나이 상한
        <select id="f-age"></select>
      </label>
      <label for="f-pos">포지션
        <select id="f-pos">
          <option value="">전체</option>
          <option value="Attack">공격수</option>
          <option value="Midfield">미드필더</option>
          <option value="Defender">수비수</option>
          <option value="Goalkeeper">골키퍼</option>
        </select>
      </label>
      <label for="f-sort">정렬
        <select id="f-sort">
          <option value="fut">성장 가능성 (3년 뒤 기량)</option>
          <option value="cur">현재 기량</option>
          <option value="gain">3년 동안 오를 점수</option>
        </select>
      </label>
      <span class="note">행을 누르면 선수 카드가 열립니다.</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>순위</th><th>이름</th><th>나이</th><th>포지션</th><th>소속</th><th>현재</th><th>3년 뒤</th><th>80% 범위</th><th>몸값</th></tr></thead>
        <tbody id="rank-body"></tbody>
      </table>
    </div>
  </section>

  <footer>
    <p>점수는 EA FC(FIFA) 평점과 같은 1~99 척도입니다. 현재 기량은 7월 1일까지의 출전 기록, 뛴 대회와 소속 팀의 수준, 몸값, 작년 EA 평점으로 추정했고, 성장 가능성은 과거 선수들의 사례로 배운 3년 뒤 예상 기량입니다.</p>
    <p>80% 범위는 과거 예측에서 실제 결과가 갈린 폭입니다. 10명 중 8명은 이 범위 안에 들었습니다. 등급 이름(월드클래스, 정상급 등)은 점수대를 읽기 쉽게 붙인 대략적인 기준입니다.</p>
    <p>데이터: Transfermarkt 기반 공개 데이터셋, EA FC 평점 공개 데이터셋 (Kaggle). 2025년 7월 기준 최근 1년 안에 몸값이 매겨진 15~40세 선수만 들어 있습니다.</p>
  </footer>
</main>

<script>
const D = __DATA__;
const N = D.name.length;
const POS_KO = { Attack: "공격수", Midfield: "미드필더", Defender: "수비수", Goalkeeper: "골키퍼" };
const EXAMPLES = ["Lamine Yamal", "Heung-min Son", "Kang-in Lee", "Min-hyeok Yang", "Jude Bellingham", "Erling Haaland"];
const TIERS = [[85, "월드클래스", "--t1"], [80, "정상급", "--t2"], [75, "빅리그 주전급", "--t3"], [70, "1부 리그 주전급", "--t4"], [65, "1부 리그 로테이션급", "--t5"], [0, "성장 단계", "--t6"]];

function norm(s) {
  return (s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase()
    .replace(/ø/g, "o").replace(/đ/g, "d").replace(/ł/g, "l").replace(/ß/g, "ss").replace(/æ/g, "ae").replace(/ı/g, "i")
    .replace(/-/g, " ").replace(/[^a-z ]/g, "").trim();
}
const TOKENS = D.name.map(n => norm(n).split(/\s+/).filter(Boolean));

function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function money(k) { if (k == null) return "-"; return k >= 1000 ? "€" + (k / 1000).toFixed(1) + "M" : "€" + k + "K"; }
function num(v, scale) { return v == null ? null : v / (scale || 1); }
function tier(score) { return TIERS.find(t => score >= t[0]); }
function topShare(pct10) { return Math.max(100 - pct10 / 10, 0.1).toFixed(1); }

function player(i) {
  return {
    i, name: D.name[i], age: D.age[i] / 10, main: D.mainList[D.main[i]], detail: D.detailList[D.detail[i]],
    club: D.clubList[D.club[i]], league: D.leagueList[D.league[i]], nation: D.nationList[D.nation[i]],
    value: D.value[i], games: D.games[i], starts: D.starts[i], goals: D.goals[i], assists: D.assists[i],
    eaPrev: D.eaPrev[i], eaPot: D.eaPot[i], eaNow: D.eaNow[i],
    cur: D.cur[i] / 10, fut: D.fut[i] / 10, lo: D.lo[i], hi: D.hi[i], curPct: D.curPct[i], futPctAge: D.futPctAge[i],
  };
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

function scaleSvg(p) {
  const W = 640, x = v => 20 + (Math.min(Math.max(v, 40), 100) - 40) / 60 * (W - 40);
  const ticks = [40, 50, 60, 70, 80, 90, 100].map(t =>
    `<line x1="${x(t)}" y1="30" x2="${x(t)}" y2="36" stroke="var(--line)"/><text x="${x(t)}" y="52" text-anchor="middle">${t}</text>`).join("");
  const bandW = Math.max(x(p.hi) - x(p.lo), 2);
  return `<svg viewBox="0 0 ${W} 60" role="img" aria-label="현재 ${Math.round(p.cur)}점, 3년 뒤 ${Math.round(p.fut)}점, 80% 범위 ${p.lo}~${p.hi}점">
    <rect x="20" y="30" width="${W - 40}" height="6" rx="3" fill="var(--soft)"/>
    <rect x="${x(p.lo)}" y="26" width="${bandW}" height="14" rx="4" fill="var(--band)"/>
    ${ticks}
    <g class="mk"><circle cx="${x(p.cur)}" cy="33" r="7" fill="var(--accent)"/><text x="${x(p.cur)}" y="16" text-anchor="middle" fill="var(--accent)">지금 ${Math.round(p.cur)}</text></g>
    <g class="mk"><circle cx="${x(p.fut)}" cy="33" r="7" fill="var(--surface)" stroke="var(--gold)" stroke-width="3"/><text x="${x(p.fut)}" y="${Math.abs(x(p.fut) - x(p.cur)) < 70 ? 4 : 16}" text-anchor="middle" fill="var(--gold)">3년 뒤 ${Math.round(p.fut)}</text></g>
  </svg>`;
}

function renderCard(i) {
  const p = player(i);
  const t = tier(p.cur), tf = tier(p.fut);
  const g = Math.round(p.fut - p.cur);
  const gCls = g >= 1 ? "up" : g <= -1 ? "down" : "flat";
  const gTxt = g >= 1 ? "상승 예상" : g <= -1 ? "하락 예상" : "유지 예상";
  const pos = POS_KO[p.main] || p.main;
  const season = p.games == null ? "" : `지난 시즌 <strong>${p.games}경기</strong> (선발 ${p.starts}) · ${p.goals}골 ${p.assists}도움`;
  document.getElementById("card").innerHTML = `
    <h2>${esc(p.name)}</h2>
    <p class="meta"><strong>${p.age.toFixed(0)}세</strong> · ${esc(pos)}${p.detail && p.detail !== p.main ? " (" + esc(p.detail) + ")" : ""} · ${esc(p.club || "소속 정보 없음")}${p.league ? " · " + esc(p.league) : ""}${p.nation ? " · " + esc(p.nation) : ""}</p>
    <p class="meta">몸값 <strong>${money(p.value)}</strong>${season ? " · " + season : ""}</p>
    <div class="scores">
      <div class="score now">
        <span class="label">현재 기량</span>
        <span class="big"><span class="num">${Math.round(p.cur)}</span><span class="of">/ 100</span><span class="tier" style="color: var(${t[2]})">${t[1]}</span></span>
        <span class="sub">전체 선수 중 상위 ${topShare(p.curPct)}%</span>
      </div>
      <div class="score future">
        <span class="label">성장 가능성 · 3년 뒤 예상 기량</span>
        <span class="big"><span class="num">${Math.round(p.fut)}</span><span class="of">/ 100</span><span class="delta ${gCls}">${g > 0 ? "+" : ""}${g}</span><span class="tier" style="color: var(${tf[2]})">${gTxt}</span></span>
        <span class="sub">80% 범위 ${p.lo}~${p.hi} · 같은 나이 선수 중 상위 ${topShare(p.futPctAge)}%</span>
      </div>
    </div>
    <div class="scale">${scaleSvg(p)}</div>
    <div class="ref">
      <span>EA FC 25 (2025년 6월)<b>${p.eaPrev ?? "-"}</b></span>
      <span>EA 잠재력<b>${p.eaPot ?? "-"}</b></span>
      <span>EA FC 26 (2025년 9월)<b>${p.eaNow ?? "-"}</b></span>
    </div>`;
  current = i;
  document.querySelectorAll(".result").forEach(el => el.setAttribute("aria-current", String(+el.dataset.i === i)));
}

let current = null;
function renderResults(list, query) {
  const ol = document.getElementById("results");
  if (!query) { ol.innerHTML = `<li class="count">이름을 입력하면 여기 목록이 나옵니다. 지금은 예시 선수를 보여 드립니다.</li>`; return; }
  if (!list.length) { ol.innerHTML = `<li class="empty">"${esc(query)}"과 맞는 선수가 없습니다. 영문 이름의 일부만 넣어 보세요. (예: Son, Yamal)</li>`; return; }
  ol.innerHTML = `<li class="count">${list.length >= 50 ? "50명 이상" : list.length + "명"} · 현재 기량순</li>` + list.map(i => {
    const p = player(i);
    return `<li class="result" tabindex="0" data-i="${i}" aria-current="${i === current}">
      <span class="nm">${esc(p.name)}</span>
      <span class="mini"><span>${Math.round(p.cur)}</span><span class="arrow">→</span><span class="f">${Math.round(p.fut)}</span></span>
      <span class="sub">${p.age.toFixed(0)}세 · ${esc(p.club || "-")}</span>
    </li>`;
  }).join("");
}

function onSearch() {
  const q = document.getElementById("q").value;
  const list = search(q, 50);
  renderResults(list, q.trim());
  if (list.length) renderCard(list[0]);
}

// 유망주 순위
function renderRank() {
  const maxAge = +document.getElementById("f-age").value;
  const pos = document.getElementById("f-pos").value;
  const sort = document.getElementById("f-sort").value;
  const idx = [];
  for (let i = 0; i < N; i++) {
    if (D.age[i] / 10 > maxAge) continue;
    if (pos && D.mainList[D.main[i]] !== pos) continue;
    idx.push(i);
  }
  const key = sort === "cur" ? i => D.cur[i] : sort === "gain" ? i => D.fut[i] - D.cur[i] : i => D.fut[i];
  idx.sort((a, b) => key(b) - key(a) || D.fut[b] - D.fut[a]);
  document.getElementById("rank-body").innerHTML = idx.slice(0, 100).map((i, r) => {
    const p = player(i);
    return `<tr tabindex="0" data-i="${i}">
      <td class="rk">${r + 1}</td><td>${esc(p.name)}</td><td>${p.age.toFixed(1)}</td><td>${POS_KO[p.main] || ""}</td>
      <td>${esc(p.club || "-")}</td><td class="n c">${Math.round(p.cur)}</td><td class="n f">${Math.round(p.fut)}</td>
      <td>${p.lo}~${p.hi}</td><td>${money(p.value)}</td></tr>`;
  }).join("");
}

function showTab(name) {
  const isRank = name === "rank";
  document.getElementById("view-search").hidden = isRank;
  document.getElementById("view-rank").hidden = !isRank;
  document.getElementById("tab-search").setAttribute("aria-selected", String(!isRank));
  document.getElementById("tab-rank").setAttribute("aria-selected", String(isRank));
  if (isRank) renderRank();
}

function openPlayer(i) {
  document.getElementById("q").value = D.name[i];
  showTab("search");
  onSearch();
  renderCard(i);
}

// 화면 연결
document.getElementById("stamp").textContent = `2025년 7월 1일 기준 · 15~40세 선수 ${N.toLocaleString("ko-KR")}명 · 현재 기량과 3년 뒤 예상 기량 (1~100)`;
document.getElementById("examples").insertAdjacentHTML("beforeend",
  EXAMPLES.map(n => `<button class="chip" type="button" data-q="${esc(n)}">${esc(n)}</button>`).join(""));
document.getElementById("examples").addEventListener("click", e => {
  const b = e.target.closest(".chip"); if (!b) return;
  document.getElementById("q").value = b.dataset.q; onSearch();
});
document.getElementById("q").addEventListener("input", onSearch);
document.getElementById("results").addEventListener("click", e => { const li = e.target.closest(".result"); if (li) renderCard(+li.dataset.i); });
document.getElementById("results").addEventListener("keydown", e => { const li = e.target.closest(".result"); if (li && e.key === "Enter") renderCard(+li.dataset.i); });
document.getElementById("rank-body").addEventListener("click", e => { const tr = e.target.closest("tr"); if (tr) openPlayer(+tr.dataset.i); });
document.getElementById("rank-body").addEventListener("keydown", e => { const tr = e.target.closest("tr"); if (tr && e.key === "Enter") openPlayer(+tr.dataset.i); });
document.getElementById("tab-search").addEventListener("click", () => showTab("search"));
document.getElementById("tab-rank").addEventListener("click", () => showTab("rank"));
const ageSel = document.getElementById("f-age");
for (let a = 17; a <= 25; a++) ageSel.insertAdjacentHTML("beforeend", `<option value="${a}"${a === 21 ? " selected" : ""}>${a}세 이하</option>`);
["f-age", "f-pos", "f-sort"].forEach(id => document.getElementById(id).addEventListener("change", renderRank));

// 처음 화면: 예시 선수 한 명의 카드
renderResults([], "");
const first = search("Lamine Yamal", 1)[0];
renderCard(first ?? 0);
if (location.hash === "#rank") showTab("rank");
</script>
"""

LOCAL_HEAD = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>
"""


def main():
    data, n = build_data()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = PAGE.replace("__DATA__", payload)
    WEB.mkdir(exist_ok=True)
    (WEB / "chipscouting.html").write_text(page, encoding="utf-8")
    # 브라우저로 바로 여는 판: 아티팩트가 자동으로 붙여 주는 문서 머리 부분을 직접 붙인다
    head, body = page.split("<header", 1)
    (WEB / "index.html").write_text(LOCAL_HEAD + head + "</head>\n<body>\n<header" + body + "</body>\n</html>\n", encoding="utf-8")
    size = (WEB / "chipscouting.html").stat().st_size / 1e6
    print(f"선수 {n:,}명을 넣은 페이지를 만들었습니다 ({size:.1f}MB)")
    print(f"  {WEB / 'index.html'}  (브라우저로 열기)")
    print(f"  {WEB / 'chipscouting.html'}  (아티팩트용)")


if __name__ == "__main__":
    main()
