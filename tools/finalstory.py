"""finalstory：把一局存档筛成「玩家视角」的最终故事。

玩家视角 = 你眼前的世界：
- 你场景里发生的（看得见的）
- 你眼前人物的状态与开口（听得见的）
- 你不在场的事，一律不知道（账本有，但故事里没有）
加上世界背景 + 你该知道的人物背景，写进 finalstory 文件。
用法：python -m tools.finalstory [存档] [输出]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldledger.store import PHASE_NAMES, Universe, text_similarity  # noqa: E402

SRC = "worldledger_save/quiet_test.json"
OUT = "worldledger_save/finalstory_quiet.txt"
SEP = "\n" + "─" * 46 + "\n"


def params_of(e) -> dict:
    """事件参数：emit 包了 event_params 一层，直接 log 没有——两种形状都读。"""
    p = e.payload or {}
    return p.get("event_params", p) if isinstance(p, dict) else {}


def player_visible(w, e, loc: str) -> bool:
    """玩家的眼睛：只看自己「当时所在场景」发生的事 + 与自己有关的事。"""
    if e.kind in ("player_said", "player_acted", "action_refused",
                  "scene_entered"):
        return True
    # `dialogue` 同时承担 NPC 私人记忆的落库。只有带 reply 的事件才是
    # 玩家亲耳听到的回应；其他记录绝不能借玩家视角露出来。
    if e.kind == "dialogue":
        return bool(params_of(e).get("reply"))
    p = params_of(e)
    # 跨场景行动开始于出发地；location 是尚未兑现的目的地。玩家在
    # 目的地不能提前看见远处的人动身，抵达会另有 npc_moved 事件。
    if e.kind == "npc_acted" and p.get("origin"):
        return loc == p.get("origin")
    # 离场与抵达都发生在现场：玩家站在原处也会看见有人离开，不能只取
    # `to` 而吞掉 `from`。
    event_locs = {p.get("location"), p.get("from"), p.get("to"),
                  p.get("scene")}
    return loc in event_locs


def render_time(e) -> str:
    """事件时间：用事件自己的钟点戳（世界钟），不用回合反推——
    回合反推的日/相位与世界钟脱节，会写出「第 22 天」和相位错乱。"""
    day = int(e.day) + 1
    phase = PHASE_NAMES[int((e.day % 1.0) * 4) % 4]
    return f"第{day}天·{phase}"


def render(w, e) -> str:
    """把事件渲染成玩家视角的一句话。"""
    p = params_of(e)

    def body_name() -> str:
        body_id = p.get("body") or p.get("npc", "")
        body = w.npcs.get(body_id)
        name = body.name if body else body_id
        # 玩家只能看到身体的外在行为；行动者身份属于账本内部知识。
        if p.get("actor") and p.get("actor") != body_id:
            return f"{name}的身体"
        return name

    if e.kind == "player_said":
        return f"⬤ 你 > {p.get('content', e.summary)}"
    if e.kind == "player_acted":
        return f"⬤ 你做了：{p.get('action', '')}（被接受）"
    if e.kind == "action_refused":
        return f"⬤ 你做了：{p.get('action', '')}（被拒绝：{p.get('reason', '')}）"
    if e.kind == "scene_entered":
        return f"⬤ {e.summary}"
    if e.kind == "item_removed":
        # 自动清扫是引擎语言：玩家看到的是「东西不见了」
        return f"「{p.get('name', '某物')}」不见了。"
    if e.kind == "dialogue":
        npc = p.get("npc", "")
        reply = p.get("reply", "")
        name = (w.npcs.get(npc).name if npc in w.npcs else npc)
        if reply:
            return f"{name}：{reply}"
        return ""
    if e.kind == "npc_acted":
        return f"{body_name()} 主动：{p.get('action', '')}"
    if e.kind == "action_done":
        return (f"{body_name()} 完成了「{p.get('action', '')}」："
                f"{p.get('outcome', '')}")
    if e.kind == "action_aborted":
        return f"{body_name()} 中止了动作「{p.get('action', '')}」"
    s = e.summary
    if e.kind == "item_changed":
        # 玩家当时在场（可见才进来）：把账本状态转成目击
        s = "你看见" + s
    if e.kind == "npc_memory":
        for sep in (" 记住了：", " 经历："):
            if sep in s:
                head, content = s.split(sep, 1)
                name = head.strip()
                if content.startswith(name + "："):
                    content = content[len(name) + 1:]
                s = f"{name} 你看见：" + content
                break
    return s


def build(src: str, out: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    w = Universe.load(src).here
    start = w.scenes[w.player["location"]]
    out_lines: list[str] = []

    # 世界背景：你走进这座城时知道的
    out_lines.append("# 雨城 · 玩家视角最终故事" + SEP)
    out_lines.append(f"《{w.name}》· {w.law_profile.atmosphere}")
    out_lines.append("")
    out_lines.append("你走进这座城时，知道的只有这些：")
    background: list[str] = []
    for law in w.law_profile.laws:
        background.append(f"{law.trigger} → {law.effect}")
    for fact in w.facts:
        # 世界档案常把法则效果再表述一遍。玩家开场只需要得到一次信息；
        # 使用现有的文本近似，不把题材词汇写死到导出器里。
        if any(text_similarity(fact, known) >= 0.38 for known in background):
            continue
        background.append(fact)
    for line in background:
        out_lines.append(f"- {line}")
    out_lines.append("")
    out_lines.append("你认识的人（你在这里过日子，自然认得这些脸）：")
    for n in w.npcs.values():
        out_lines.append(f"- {n.name}：{n.persona}")
    out_lines.append(SEP)

    # 玩家视角时间线：只看自己场景里发生的
    out_lines.append(f"从【{start.name}】出发，雨城已过去约 {w.clock:.1f} 天。"
                     "你亲眼经历的，是这些：")
    out_lines.append("")
    seen = set()
    count = 0
    # 按时间重放玩家位置：初始位置 = 存档里的 start（若有），
    # 否则取第一条带 location 的事件；scene_entered 更新位置。
    loc = str(w.player.get("start", ""))
    if not loc:
        for first in w.events:
            l0 = params_of(first).get("location") or ""
            if l0:
                loc = l0
                break
    for e in w.events:
        if e.kind == "scene_entered":
            dest = params_of(e).get("scene", "")
            if dest:
                loc = dest
        if e.kind == "time_passed" or not player_visible(w, e, loc):
            continue
        s = render(w, e)
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out_lines.append(f"- **{render_time(e)}** {s}")
        count += 1
    out_lines.append("")
    out_lines.append("（远处仍在雾里。你不在场，所以无从得知。）")
    out_lines.append(SEP)
    out_lines.append(f"你亲历或目睹了 {count} 件事。")
    Path(out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"已写入 {out}（玩家可见 {count} 条）")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else SRC,
          sys.argv[2] if len(sys.argv) > 2 else OUT)
