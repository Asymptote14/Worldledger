"""真实长局试玩：我当玩家，¥10 预算内跑三局，验证机制与体验。

局 1：雨城（法则世界）——聊天、动作、离开两天、回来问发生了什么
局 2：潮汐镇（时刻锚）——跑到第七夜，看既定时刻到点必发、地点正确
局 3：普通小镇——验证平淡世界不奇幻乱入

每步打印世界钟（now），结尾打印成本账。
用法：python -m tools.playtest [world1|world2|world3]
"""
from __future__ import annotations

import json
import sys

from worldledger import evolution, interpreter
from worldledger.llm import get_llm
from worldledger.store import Universe, scene_changes
from worldledger.worldgen import ensure_scene, generate_world

SEP = "\n" + "─" * 56 + "\n"
SAVE_DIR = "worldledger_save"

RAIN = "一座永远在下雨的小城，人们撒谎时会漏出真心话"
COMET = ("两名居民每晚会在梦里交换一段短暂的行动权限。第七夜，"
         "潮汐发电塔进入维护窗口，海水会淹过旧堤，低洼仓库暂时停摆。"
         "其中一件关键物品必须在维护窗口前被转移；窗口结束后，"
         "城市恢复运转，但错过时机的行动不会凭空完成。")
PLAIN = "一座普通的小镇，日子平淡，人们照常上班买菜，天黑了就回家"


class CountingLLM:
    """委托真实 LLM，按 TASK 累计调用与 token 估算（deepseek-chat）。"""

    name = "counting"

    def __init__(self, inner):
        self.inner = inner
        self.stats: dict[str, dict] = {}

    @staticmethod
    def _task(system: str) -> str:
        for k in ("WORLDGEN", "DIALOGUE", "WORLDPULSE", "PLAYERACT",
                  "LAWCHANGE", "NPCGOAL", "ACTIONRESOLVE", "RUMOR",
                  "SCENEGEN", "INTERACT", "OPENER", "PATCHGEN"):
            if f"TASK:{k}" in system:
                return k
        return "OTHER"

    @staticmethod
    def _est(s: str) -> int:
        zh = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
        return zh + max(1, (len(s) - zh) // 4)

    def chat(self, system, user):
        out = self.inner.chat(system, user)
        st = self.stats.setdefault(self._task(system),
                                   {"calls": 0, "in": 0, "out": 0})
        st["calls"] += 1
        st["in"] += self._est(system) + self._est(user)
        st["out"] += self._est(out)
        return out

    def chat_json(self, system, user, attempts=2):
        from worldledger.llm import extract_json
        last = None
        for _ in range(attempts):
            text = self.chat(system, user)
            try:
                return json.loads(extract_json(text))
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"解析失败：{last}")

    def report(self) -> tuple[float, float]:
        """返回（估算成本元，总 token）。deepseek-chat：入 2 元/M，出 8 元/M。"""
        cost = 0.0
        tokens = 0
        for st in self.stats.values():
            tokens += st["in"] + st["out"]
            cost += st["in"] * 2.0 / 1e6 + st["out"] * 8.0 / 1e6
        return cost, tokens


def _utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def say(llm, w, npc, text):
    r = interpreter.dialogue_turn(llm, w, npc, text)
    print(f"  你 > {text}")
    print(f"  {npc.name}：{r.reply}")


def act(llm, w, npc, text):
    r = interpreter.player_action(llm, w, npc, text)
    print(f"  你 > {text}")
    print(f"  {npc.name}：{r.reply}")
    if r.law_triggers:
        print(f"  ⚡ 法则触发：{r.law_triggers}")


def save(w, name):
    u = Universe(worlds={w.name: w}, current=w.name)
    u.save(f"{SAVE_DIR}/{name}.json")
    print(f"  （已存档 {SAVE_DIR}/{name}.json）")


def world1(llm):
    print(SEP + " 局 1 · 雨城：玩家视角全流程" + SEP)
    w = generate_world(llm, "雨城", RAIN)
    print(f"✨ {w.name} 诞生：{w.law_profile.atmosphere}｜{w.now()}")
    start = w.scenes[w.player["location"]]
    ensure_scene(llm, w, start.id)
    print(f"你醒来在【{start.name}】")
    npc = next((w.npcs[n] for n in start.npcs if n in w.npcs),
               next(iter(w.npcs.values())))
    print(f"在场：{npc.name}｜{w.now()}")
    say(llm, w, npc, "你在等谁？")
    say(llm, w, npc, "这雨下多久了？")
    act(llm, w, npc, "把外套脱下来，披到他肩上")
    clock_before = w.clock
    print(f"\n  你离开了。{w.now()}")
    since = w.turn
    for _ in range(8):  # 离开期间世界照常呼吸
        w.pass_time(6)
        for s in evolution.world_pulse(llm, w):
            print(f"    （离开中·{w.now()}）{s[:70]}")
    gone_days = int(w.clock - clock_before + 0.5)
    print(f"  你回来了。{w.now()}（走了约 {gone_days} 天）")
    loc = w.player["location"]
    for s in evolution.catch_up_scene(llm, w, loc):
        print(f"  · {s}")
    say(llm, w, npc, "我不在的这两天，发生了什么？")
    print(f"\n  重返视图（账本导出）：")
    for c in scene_changes(w, loc, since, limit=6):
        print(f"    · [{c.get('event_id', '?')}] {c.get('fact', '')[:80]}"
              f"（因：{c.get('cause', '')}）")
    save(w, "playtest_rain")


def world2(llm):
    print(SEP + " 局 2 · 潮汐镇：既定时刻到点必发" + SEP)
    w = generate_world(llm, "潮汐镇", COMET)
    print(f"✨ {w.name} 诞生：{w.law_profile.atmosphere}｜{w.now()}")
    print(f"  生成提取的 moments：{w.moments}")
    start = w.scenes[w.player["location"]]
    ensure_scene(llm, w, start.id)
    for r in range(30):  # 7 天 ≈ 168 回合 ≈ 28 轮 × pass_time(6)
        w.pass_time(6)
        for s in evolution.world_pulse(llm, w):
            print(f"  [{w.now()}] {s[:90]}")
        if all(m.get("done") for m in w.moments) and r > 4:
            break
    fired = [e for e in w.events if e.kind == "world_event"
             and e.cause == "既定时刻"]
    print(f"\n  既定时刻事件（{len(fired)} 条）：")
    for e in fired:
        loc = (e.payload or {}).get("event_params", {}).get("location", "全局")
        print(f"    [{e.day:.2f}天] {e.summary}｜地点：{loc}")
    save(w, "playtest_comet")


def world3(llm):
    print(SEP + " 局 3 · 普通小镇：平淡不奇幻乱入" + SEP)
    w = generate_world(llm, "平淡镇", PLAIN)
    print(f"✨ {w.name} 诞生：{w.law_profile.atmosphere}｜{w.now()}")
    start = w.scenes[w.player["location"]]
    ensure_scene(llm, w, start.id)
    for r in range(12):
        w.pass_time(6)
        for s in evolution.world_pulse(llm, w):
            print(f"  [{w.now()}] {s[:90]}")
    kinds = {e.kind for e in w.events}
    print(f"\n  事件类型：{sorted(kinds)}")
    print(f"  场景：{[s.name for s in w.scenes.values()]}")
    save(w, "playtest_plain")


def main():
    _utf8()
    llm = CountingLLM(get_llm())
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("world1", "all"):
        world1(llm)
    if which in ("world2", "all"):
        world2(llm)
    if which in ("world3", "all"):
        world3(llm)
    cost, tokens = llm.report()
    print(SEP + f" 成本账：{llm.stats.get('', {})}" + SEP)
    for task, st in sorted(llm.stats.items(), key=lambda kv: -kv[1]["calls"]):
        print(f"  {task:>12}: {st['calls']:>3} 次｜"
              f"入 {st['in']:>7}｜出 {st['out']:>7} token")
    print(f"\n  合计 ≈ {cost:.2f} 元（预算 ¥10）｜{tokens} token")
    print(SEP)


if __name__ == "__main__":
    main()
