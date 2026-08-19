"""知识边界最小实验：两个 NPC + 一条只有 A 看得见的秘密。

Phase 1（现状测量）：用引擎现状分别裁决 A、B——
   B 的 recent_events 是全局最近 5 条，看他会不会「不该知道却知道」。
Phase 2（过滤演示）：把 B 的可见事件过滤到「他的场景 + 他参与的事件」，
   再裁决一次（干跑，不入账）——看知识边界是否成立。

零引擎改动：Phase 1 用 propose_proactive 原样；Phase 2 在脚本里
手工构造过滤后的载荷，用同一个 TASK:NPCGOAL 裁决器。
结论若是「Phase 1 泄漏、Phase 2 成立」→ 多 agent 的知识边界模式成立，
引擎将来只差「按知识过滤 recent_events」这一块。
用法：python -m tools.knowledge_test
"""
from __future__ import annotations

import json as _json
import sys

from worldledger import evolution
from worldledger.event import emit
from worldledger.evolution import _NPCGOAL_SYSTEM, move_npc
from worldledger.llm import get_llm
from worldledger.store import World, memory_effectiveness, resolve_target, \
    target_snapshot
from worldledger.worldgen import ensure_scene, generate_world

DESC = "一座永远在下雨的小城，人们撒谎时会漏出真心话"
SEP = "\n" + "─" * 56 + "\n"
SECRET = "今晚码头见，别告诉任何人"


def visible_events(world: World, npc_id: str, limit: int = 5) -> list[str]:
    """NPC 的知识边界：只看得见自己场景里发生的、或自己参与的事件。"""
    out = []
    for e in reversed(world.events):
        p = (e.payload or {}).get("event_params", {})
        loc = p.get("location") or p.get("to") or p.get("scene") or ""
        if loc == world.npcs[npc_id].state.location \
                or p.get("npc") == npc_id or p.get("target") == npc_id:
            out.append(e.summary)
        if len(out) >= limit:
            break
    return list(reversed(out))


def dry_run_goal(llm, world, npc):
    """Phase 2 干跑：用过滤后的可见事件构造载荷，同一裁决器，不入账。"""
    open_goals = [g for g in npc.goals if float(g.get("progress", 0)) < 1.0]
    payload = _json.dumps({
        "npc": {"id": npc.id, "name": npc.name, "goals": open_goals,
                "relationship": npc.relationship,
                "mood_value": npc.state.mood_value},
        "atmosphere": world.law_profile.atmosphere,
        "memories": [m.content for m in sorted(
            npc.memories, key=lambda m: memory_effectiveness(m, world.turn),
            reverse=True)[:8]],
        "player_traces": evolution._player_traces(world, npc),
        "recent_events": visible_events(world, npc.id),
        "laws": [f"{l.trigger} → {l.effect}"
                 for l in world.law_profile.laws],
        "facts": list(world.facts),
        "rejection_note": world.social.get(f"reject|{npc.id}", ""),
        "targets_now": [
            {"ref": ref, "display": tgt, "snapshot": snap}
            for ref in {r for g in open_goals
                        for t in g.get("targets", []) if isinstance(t, str)
                        for r in [resolve_target(world, t)] if r}
            for tgt in [""]
            for snap in [target_snapshot(world, ref)] if snap
        ],
        "scenes": {sid: s.name for sid, s in world.scenes.items()},
    }, ensure_ascii=False)
    return llm.chat_json(_NPCGOAL_SYSTEM, payload)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    llm = get_llm()
    w = generate_world(llm, "知识边界", DESC)
    start = w.scenes[w.player["location"]]
    ensure_scene(llm, w, start.id)
    others = [s for s in w.scenes.values() if s.id != start.id]
    for s in others[:1]:
        ensure_scene(llm, w, s.id)
    a = next(w.npcs[n] for n in start.npcs if n in w.npcs)
    b = next((w.npcs[n] for s in others[:1] for n in s.npcs if n in w.npcs),
             next(n for n in w.npcs.values() if n.id != a.id))
    if b.state.location == a.state.location:
        move_npc(w, b, others[0].id, cause="实验：分开两人")
    goal = {"id": "g-kb", "text": "留意最近城里发生的怪事，别错过什么",
            "progress": 0.0, "targets": [], "because": "实验注入"}
    for n in (a, b):
        n.goals.append(dict(goal))

    print(SEP + " 知识边界实验：两人 + 一个秘密" + SEP)
    print(f"  A = {a.name}（{w.scenes[a.state.location].name}）")
    print(f"  B = {b.name}（{w.scenes[b.state.location].name}）")
    # 秘密只在 A 的场景发生
    emit(w, "world_event",
         {"title": "窗台下的纸条",
          "detail": f"有人把一张纸条压在窗台下：『{SECRET}』",
          "location": a.state.location, "intensity": 0.4},
         cause="实验注入")
    print(f"  注入秘密（只在 {w.scenes[a.state.location].name}）：『{SECRET}』")

    print(SEP + " Phase 1 · 引擎现状（recent_events = 全局最近 5 条）" + SEP)
    before = len(w.events)
    sa = evolution.propose_proactive(llm, w, a)
    for s in sa:
        print(f"  A 提案：{s[:90]}")
    sb = evolution.propose_proactive(llm, w, b)
    for s in sb:
        print(f"  B 提案：{s[:90]}")
    b_acts = [e for e in w.events[before:]
              if e.kind == "npc_acted"
              and (e.payload or {}).get("event_params", {}).get("npc")
              == b.id]
    b_actions = [str((e.payload or {}).get("event_params", {}).get(
        "action", "")) for e in b_acts]
    leak = any(("码头" in t or "纸条" in t) for t in b_actions)
    print(f"\n  B 的行动里提到秘密？{'是（知识泄漏）' if leak else '否'}")

    print(SEP + " Phase 2 · 知识边界（可见事件 = 他的场景 + 他参与的）" + SEP)
    data = dry_run_goal(llm, w, b)
    evs = data.get("events", [])
    ups = data.get("goal_updates", {})
    print(f"  B 的可见事件：{visible_events(w, b.id)}")
    print(f"  B 提案：{evs}")
    print(f"  B 目标更新：{ups}")
    leak2 = any("码头" in _json.dumps(x, ensure_ascii=False)
                or "纸条" in _json.dumps(x, ensure_ascii=False)
                for x in (evs, ups))
    print(f"\n  B 的裁决里提到秘密？{'是' if leak2 else '否（边界成立）'}")

    print(SEP + " 结论" + SEP)
    if leak and not leak2:
        print("  Phase 1 泄漏、Phase 2 成立 → 多 agent 的知识边界模式可行；")
        print("  引擎将来只需把 recent_events 按「场景 + 参与」过滤给每个 NPC。")
    elif not leak and not leak2:
        print("  Phase 1 已由引擎结构过滤（_npc_visible）：A 看见并行动，")
        print("  B 根本拿不到秘密——知识边界从模型自律变成结构保证。")
    elif not leak:
        print("  现状就不泄漏——模型自己没敢用；过滤仍是应该补的纪律。")
    else:
        print("  过滤后仍泄漏——需要查载荷里还有哪个口子。")
    print(SEP)


if __name__ == "__main__":
    main()
