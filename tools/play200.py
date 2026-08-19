"""200 回合参与式长局：我当玩家，适当参与（说话、做事、离开、回来）。

用法：python -m tools.play200 [轮数] [世界描述]
"""
from __future__ import annotations

import sys

from worldledger import evolution, interpreter
from worldledger.llm import LLMError, get_llm
from worldledger.store import Universe
from worldledger.worldgen import ensure_scene, generate_world

DESC = "一座永远在下雨的小城，人们撒谎时会漏出真心话"
SAVE = "worldledger_save/play200.json"
SEP = "\n" + "─" * 56 + "\n"


class RetryLLM:
    """真实 API 偶发坏 JSON：重试包装（工具侧，不动引擎）。"""

    def __init__(self, inner, attempts=3):
        self._inner = inner
        self.attempts = attempts

    @property
    def name(self):
        return self._inner.name

    def chat(self, system, user):
        last = None
        for _ in range(self.attempts):
            try:
                return self._inner.chat(system, user)
            except Exception as e:  # noqa: BLE001
                last = e
        raise last

    def chat_json(self, system, user, attempts=None):
        from worldledger.llm import extract_json
        import json as _json
        last = None
        for _ in range(self.attempts):
            try:
                return _json.loads(extract_json(self._inner.chat(system, user)))
            except Exception as e:  # noqa: BLE001
                last = e
        raise last if last else LLMError("chat_json 重试耗尽")


def say(llm, w, npc, text):
    r = interpreter.dialogue_turn(llm, w, npc, text)
    print(f"  你 > {text}")
    visible = r.reply or r.reaction or r.action
    print(f"  {npc.name}：{visible[:80]}")


def act(llm, w, npc, text):
    r = interpreter.player_action(llm, w, npc, text)
    print(f"  你 > {text}")
    print(f"  {npc.name}：{r.reply[:80]}")


def go(w, scene_id, note="你走过去"):
    previous = w.player.get("location", "")
    w.player["location"] = scene_id
    w.log("scene_entered",
          f"{note}：{w.scenes[scene_id].name}", "玩家移动",
          {"actor": "player", "from": previous, "to": scene_id,
           "scene": scene_id})
    print(f"  ⬤ {note}：{w.scenes[scene_id].name}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    llm = get_llm()
    llm = RetryLLM(llm)
    desc = sys.argv[2] if len(sys.argv) > 2 else DESC
    if desc.endswith(".txt"):  # 种子文件：直接读入作为世界描述
        from pathlib import Path
        desc = Path(desc).read_text(encoding="utf-8")
    w = generate_world(llm, "世界", desc)
    start = w.scenes[w.player["location"]]
    w.player["start"] = start.id  # 初始位置入档：finalstory 按时间重放用
    ensure_scene(llm, w, start.id)
    other_scene = next((s for s in w.scenes.values() if s.id != start.id), None)
    if other_scene:
        ensure_scene(llm, w, other_scene.id)
    npc = next((w.npcs[n] for n in start.npcs
                if n in w.npcs and not w.npcs[n].in_fog), None)
    if npc is None:  # 出生场景没人：拉一个活跃角色过来开场
        npc = next((n for n in w.npcs.values() if not n.in_fog),
                   next(iter(w.npcs.values())))
        evolution.move_npc(w, npc, start.id, cause="实验：开场在场")
    npc2 = next((w.npcs[n] for s in [other_scene] if s
                 for n in s.npcs if n in w.npcs), npc)
    print(SEP + f" 200 回合参与式长局：{w.now()}" + SEP)
    print(f"  你在【{start.name}】，在场：{npc.name}")

    rounds_total = int(sys.argv[1]) if len(sys.argv) > 1 else 34
    anchor = rounds_total / 34.0
    for r in range(rounds_total):  # 每轮 pass_time(6) ≈ 6 回合
        w.pass_time(6, cause="玩家等待")
        if r == int(2 * anchor):
            say(llm, w, npc, "你在等谁？")
        if r == int(6 * anchor):
            act(llm, w, npc, "把伞往他那边挪了挪，问他冷不冷")
        if r == int(12 * anchor) and other_scene:
            go(w, other_scene.id, "你离开广场，去了")
            for _ in range(2):  # 在新场景过一会儿
                w.pass_time(6)
                evolution.catch_up_scene(llm, w, other_scene.id)
        if r == int(15 * anchor):
            say(llm, w, npc2, "这城里，最近有什么新鲜事吗？")
        if r == int(22 * anchor):
            go(w, start.id, "你回到广场")
            evolution.catch_up_scene(llm, w, start.id)
        if r == int(25 * anchor):
            say(llm, w, npc, "我不在的这几天，发生了什么？")
        if r == int(31 * anchor):
            act(llm, w, npc, "离开前，把口袋里最后一枚硬币放在他的摊上")
        for s in evolution.world_pulse(llm, w):
            pass  # 只收账本

    u = Universe(worlds={w.name: w}, current=w.name)
    save_path = sys.argv[3] if len(sys.argv) > 3 else SAVE
    u.save(save_path)
    print(SEP + f"（已存档 {save_path}，{w.turn} 回合 / {w.clock:.1f} 天）" + SEP)


if __name__ == "__main__":
    main()
