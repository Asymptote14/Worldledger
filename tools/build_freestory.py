"""从 rain5c.json 生成 freestory.txt 的时间线部分（storyclean 规则 + story4 修正）。

storyclean 规则（照搬 storyclean.py）：
- 去掉 time_passed
- npc_memory 复读（记住了：对 X 说 / 记住了：X 主动）→ 去掉
- npc_acted 台词与近期搭话重复、或动作退化 → 去掉
- 幸存 npc_memory 的「记住了：」→「听说：」
- 全文去重

story4 修正（经 story4.txt 逐行核对）：
- 去掉「中止了动作」「我搁下了这件事」记账行（story4: 0 处）
- world_event 的「A：A」自重复折叠为 A
- scene_entered 不去重（第二次进入同一场景 = 玩家归来）
- ⬤ 行 = 玩家原话（事件 cause），插在对应事件前
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldledger.store import game_time  # noqa: E402

SRC = "worldledger_save/rain5c.json"
OUT = "worldledger_save/freestory_timeline.txt"

PLAYER_LINES = {
    2: "你在等谁？",
    12: "这座城的人，为什么都不说实话？",
    21: "改法则：从今天起，这座城的人不再撒谎",
    54: "我不在的时候，发生了什么？",
    59: "你闻到的那股草药味，到底是什么？",
    80: "排水沟里的水，是不是被人动过手脚？",
    95: "我想离开这座城。",
    136: "这雨，是谁搅的？",
}


def quote_of(s: str) -> str:
    for m in re.findall(r"[「『]([^」』]{4,})[」』]", s):
        return m
    return ""


def is_duplicate_memory(summary: str, kind: str) -> bool:
    if kind != "npc_memory":
        return False
    if "记住了：" not in summary:
        return False
    tail = summary.split("记住了：", 1)[1].strip()
    return tail.startswith("对 ") or " 主动：" in tail


def is_duplicate_act(summary: str, kind: str, recent_quotes: list) -> bool:
    if kind != "npc_acted":
        return False
    q = quote_of(summary)
    if q:
        return q in recent_quotes
    tail = summary.split("主动：", 1)[-1].strip()
    return len(tail) < 10


d = json.load(open(SRC, encoding="utf-8"))
w = d["worlds"]["雨城"]
lines = []
seen = set()
recent_quotes: list[str] = []
dropped = {"dup": 0, "abort": 0, "搁下": 0, "time": 0, "memory": 0, "act": 0}

for e in w["events"]:
    turn, kind, summary = e["turn"], e["kind"], e["summary"]

    if kind == "time_passed":
        dropped["time"] += 1
        continue
    if "中止了动作" in summary:
        dropped["abort"] += 1
        continue
    if "我搁下了这件事" in summary:
        dropped["搁下"] += 1
        continue
    if kind == "world_event" and "：" in summary:
        a, b = summary.split("：", 1)
        if a == b:
            summary = a
    if is_duplicate_memory(summary, kind):
        dropped["memory"] += 1
        continue
    if is_duplicate_act(summary, kind, recent_quotes):
        dropped["act"] += 1
        continue
    if kind == "npc_interaction":
        q = quote_of(summary)
        if q:
            recent_quotes.append(q)
            recent_quotes = recent_quotes[-4:]
    if kind == "scene_entered":
        lines.append((turn, f"- **{game_time(turn)}** {summary}"))
        continue
    if summary in seen:
        dropped["dup"] += 1
        continue
    seen.add(summary)
    summary = summary.replace(" 记住了：", " 听说：")
    lines.append((turn, f"- **{game_time(turn)}** {summary}"))

star = [(t - 0.5, f"- **{game_time(t)}** ⬤ 你 > {text}")
        for t, text in PLAYER_LINES.items()]
merged = sorted(lines + star)
Path(OUT).write_text(
    "\n".join(t for _, t in merged) + "\n", encoding="utf-8")
print(f"事件 {len(w['events'])} → 时间线 {len(merged)} 行")
print("丢弃:", dropped)
