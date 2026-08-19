"""故事清洗：把存档日志整理成可分享的干净文本。

- 去掉「｜因：…」标签
- 去掉纯时间流逝行
- 去重：同一内容被 记忆/互动/动作 记了三遍的，只留一遍
- 重复的流言只留首次
用法：python -m tools.storyclean [存档路径]
      （默认 worldledger_save/troupe.json → worldledger_save/story_clean.md）
"""
from __future__ import annotations

import sys
from pathlib import Path

from worldledger.store import Universe, game_time

SKIP_KINDS = {"time_passed"}


def quote_of(s: str) -> str:
    import re
    for m in re.findall(r"[「『]([^」』]{4,})[」』]", s):
        return m
    return ""


def is_duplicate_memory(e) -> bool:
    """「X 记住了：X 主动…」= 动作的复读；「X 记住了：对 Y 说…」= 搭话的复读。"""
    if e.kind != "npc_memory":
        return False
    s = e.summary
    # 找到「记住了：」后面的部分
    if "记住了：" not in s:
        return False
    tail = s.split("记住了：", 1)[1].strip()
    return tail.startswith("对 ") or " 主动：" in tail


def is_duplicate_act(e, recent_quotes) -> bool:
    """动作台词与刚发生的搭话台词重复 → 只留搭话；退化动作（如 move）也去。"""
    if e.kind != "npc_acted":
        return False
    q = quote_of(e.summary)
    if q:
        return q in recent_quotes
    tail = e.summary.split("主动：", 1)[-1].strip()
    return len(tail) < 10


def preamble(w) -> list[str]:
    """背景与设定：让没看过原作的读者也能进门。"""
    desc = w.description
    lines = ["", "---", "",
             "## 背景：这个故事不是人写的", "",
             f"一句话描述丢进引擎：**{desc}**。引擎据此生成法则、"
             "场景与角色，随后导入原作班底。之后没有任何人写过一句台词"
             "——玩家以路人身份站在世界一角旁观。下面每一行都是从"
             "事件日志清洗出的原始记录（去掉重复与时间标签，文字未改动）。"]
    if "祈祷" in desc or "放晴" in desc:
        lines += ["", "## 给没看过《天气之子》的读者", "",
                  "原作：东京连日暴雨，离家少年帆高遇见能「祈祷放晴」的"
                  "少女阳菜——但每次放晴，她的身体都会透明一分，直至消失。"
                  "故事在「救她」与「让雨停」之间撕扯。",
                  "",
                  "这个世界里的「阳菜」名叫**祈**（有的局里叫雨宫汐）"
                  "——她是引擎自己生成的少女，不是原作角色；"
                  "帆高、凪、夏美、须贺是导入的原作班底，陪她走完了"
                  "属于这个世界自己的故事。"]
    elif "流星" in desc:
        lines += ["", "## 给没看过《你的名字》的读者", "",
                  "原作：乡下少女三叶与东京少年泷在梦中交换身体，"
                  "靠结绳与黄昏的逢魔时刻寻找彼此，最终要赶在"
                  "流星坠落前救下小镇。",
                  "",
                  "这个世界里的三叶由引擎生成；泷、早耶香、敕使是"
                  "导入的原作班底。"]
    lines += ["", "---", ""]
    return lines


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    src = sys.argv[1] if len(sys.argv) > 1 else "worldledger_save/troupe.json"
    out_path = (sys.argv[2] if len(sys.argv) > 2
                else "worldledger_save/story.txt")
    u = Universe.load(src)
    w = u.here
    out = Path(out_path)

    lines: list[str] = [f"# {w.name}",
                        f"> {w.description}",
                        f"> 氛围：{w.law_profile.atmosphere}",
                        "> 法则：" ]
    lines += [f"> {l.trigger} → {l.effect}"
              for l in w.law_profile.laws]
    lines += preamble(w)
    seen: set[str] = set()
    recent_quotes: list[str] = []
    dropped = 0
    for e in w.events:
        if e.kind in SKIP_KINDS:
            dropped += 1
            continue
        if is_duplicate_memory(e):
            dropped += 1
            continue
        if is_duplicate_act(e, recent_quotes):
            dropped += 1
            continue
        if e.kind == "npc_interaction":
            q = quote_of(e.summary)
            if q:
                recent_quotes.append(q)
                recent_quotes = recent_quotes[-4:]
        text = e.summary
        if text in seen:  # 流言/重复内容只留首次
            dropped += 1
            continue
        seen.add(text)
        # 幸存下来的「记住了」基本都是流言——读作「听说」
        text = text.replace(" 记住了：", " 听说：")
        lines.append(f"- **{game_time(e.turn)}** {text}")
    out.write_text("\n".join(lines), encoding="utf-8")
    kept = len(w.events) - dropped
    print(f"清洗完成：{len(w.events)} 条事件 → {kept} 行"
          f"（去掉 {dropped} 条重复/时间/标签）")
    print(f"输出：{out}")


if __name__ == "__main__":
    main()
