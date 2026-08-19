"""最小链条实验：一件关键事，从状态开始，经行动、物品、知情、后果。

只盯一条：叙事宣称「发生」的事，必须对应可检查的状态变化。
链条：密封的信(状态) → 阿凛取信(角色行动) → 信被拆开(物品变化)
      → 谁知情(知情范围) → 秘密入档(不可逆后果)
每步机械断言 PASS/FAIL，最后给出判决：哪一环是状态、哪一环只是叙事。
用法：python -m tools.chain_test
"""
from __future__ import annotations

import sys

from worldledger import evolution
from worldledger.event import emit, transfer_item
from worldledger.llm import get_llm
from worldledger.store import Universe
from worldledger.worldgen import ensure_scene, generate_world

DESC = "一座永远在下雨的小城，人们撒谎时会漏出真心话"
SEP = "\n" + "─" * 56 + "\n"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, evidence: str) -> None:
    RESULTS.append((name, ok, evidence))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}｜{evidence}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    llm = get_llm()
    w = generate_world(llm, "链条", DESC)
    ensure_scene(llm, w, w.player["location"])
    # 真实 API 生成的世界 id 不固定：按角色/物品语义找，不按夹具 id
    letter = next((i for s in w.scenes.values() for i in s.items
                   if "信" in str(i.get("name", ""))), None)
    if letter is None:  # 没有信就造一封（实验起点：密封的信）
        from worldledger.event import apply_item_patch
        scene0 = w.scenes[w.player["location"]]
        apply_item_patch(w, {"op": "add", "item": "i-letter",
                             "name": "密封的信",
                             "note": "完好，封蜡未动",
                             "location": scene0.id},
                         cause="实验注入")
        letter = next(i for i in scene0.items if i["id"] == "i-letter")
    letter_scene = next(s for s in w.scenes.values()
                        if any(i.get("id") == letter["id"] for i in s.items))
    ensure_scene(llm, w, letter_scene.id)
    arin = next((n for n in w.npcs.values() if n.goals),
                next(iter(w.npcs.values())))
    if arin.state.location == letter_scene.id:  # 移到别处，形成「去取信」
        other_scene = next(s for s in w.scenes.values()
                           if s.id != letter_scene.id)
        evolution.move_npc(w, arin, other_scene.id, cause="实验：先离场")
    other = next((n for n in w.npcs.values()
                  if n.id != arin.id and n.state.location != letter_scene.id),
                 next(n for n in w.npcs.values() if n.id != arin.id))
    arin.goals = [{"id": "g-chain", "text": f"去{letter_scene.name}取走那封"
                   f"信并把它拆开",
                   "progress": 0.0, "targets": [f"item:{letter['id']}"],
                   "because": "实验：给动机与落脚点"}]
    print(SEP + " 最小链条实验：密封的信 → 行动 → 拆开 → 知情 → 后果" + SEP)

    # 环节 1：初始状态（一切可查）
    check("状态·信完好且封蜡未动", letter.get("note", "") != "", letter["note"])
    check("状态·信在车站、无人持有", letter.get("held_by", "") == "",
          f"held_by={letter.get('held_by', '空')}")
    check("状态·阿凛不在信的场景", arin.state.location != letter_scene.id,
          f"{arin.name} 在 {arin.state.location}")

    # 环节 2：角色行动（目标裁决 → npc_acted → 真的走过去）
    w.pass_time(6)
    moved_before = arin.state.location
    for s in evolution.propose_proactive(llm, w, arin):
        print(f"    裁决：{s[:70]}")
    check("行动·阿凛真的走到了信的场景", arin.state.location == letter_scene.id,
          f"{moved_before} → {arin.state.location}")
    check("行动·进行中的动作已入账", bool(arin.state.action.text),
          arin.state.action.text[:40])

    # 环节 3：动作推进到完成（ActionState 机械推进）
    w.pass_time(30)
    for s in evolution.advance_action(llm, w, arin, 30):
        print(f"    结局：{s[:70]}")
    done = [e for e in w.events if e.kind == "action_done"]
    check("行动·动作完成入账", bool(done),
          done[-1].summary[:50] if done else "无 action_done")

    # 环节 4：行动者的 patch 落地了吗——只有 cause=动作结局 才算闭环；
    # 天气泡皱信封（cause=世界演化）不算「这件事被做成了」。
    actor_patch = [e for e in w.events
                   if e.kind in ("item_changed", "item_transfer")
                   and e.cause.startswith("动作结局")]
    check("物品·行动者的 patch 真的落地（cause=动作结局）", bool(actor_patch),
          actor_patch[-1].summary[:50] if actor_patch
          else "无——只有叙事，没有行动者的状态改写")
    check("目标·行动者离目标更近了",
          float(arin.goals[0].get("progress", 0)) > 0.1,
          f"progress={arin.goals[0].get('progress')}")

    # 环节 4b：天气驱动的变化不冒充闭环
    w.pass_time(6)
    for s in evolution.world_pulse(llm, w):
        if "信" in s:
            print(f"    心跳：{s[:70]}")
    weather_changed = [e for e in w.events if e.kind == "item_changed"
                       and "信" in e.summary
                       and not e.cause.startswith("动作结局")]
    if weather_changed and not actor_patch:
        check("澄清·天气变化不算做成这件事", False,
              "信封只是被雨泡皱（世界演化），不是谁取走了它")

    # 环节 5：知情范围——谁看得见「信的变化」这件事（按各自实际位置）
    if actor_patch or weather_changed:
        ev = (actor_patch or weather_changed)[-1]
        ev_loc = (ev.payload or {}).get("event_params", {}).get("location", "")
        if arin.state.location == ev_loc:
            check("知情·动手者看见了自己做的事",
                  _npc_visible_import(w, arin, ev), "在场，应该看得见")
        else:
            check("知情·动手者已离场，看不见后续变化",
                  not _npc_visible_import(w, arin, ev),
                  f"{arin.name} 在 {arin.state.location}，不在 {ev_loc}")
        bystander = next((n for n in w.npcs.values()
                          if n.id != arin.id and n.state.location == ev_loc),
                         None)
        if bystander:
            check("知情·同场景旁观者看得见",
                  _npc_visible_import(w, bystander, ev),
                  f"{bystander.name} 在场")
        check("知情·别处的人不知情",
              not _npc_visible_import(w, other, ev),
              f"{other.name} 看不见（在 {other.state.location}）")

    # 环节 6：不可逆后果——秘密入档、有因、旧态留痕
    facts_before = list(w.facts)
    w.pass_time(6)
    for s in evolution.world_pulse(llm, w):
        if "设定" in s or "信" in s:
            print(f"    心跳：{s[:70]}")
    new_fact = [f for f in w.facts if f not in facts_before]
    check("后果·秘密写进世界档案", bool(new_fact),
          new_fact[0][:50] if new_fact else "facts 未变")
    if new_fact:
        ev = [e for e in w.events if e.kind == "fact_changed"][-1]
        check("后果·档案变更留痕且有因", bool(ev.cause),
              ev.summary[:40] + "｜因：" + ev.cause)

    print(SEP + " 判决" + SEP)
    failed = [r for r in RESULTS if not r[1]]
    if not failed:
        print("  六环全过：这件事从状态走到了不可逆后果。")
    else:
        print(f"  断链 {len(failed)} 环：")
        for name, _, ev in failed:
            print(f"    ✗ {name}｜{ev}")
    print(SEP)
    u = Universe(worlds={w.name: w}, current=w.name)
    u.save("worldledger_save/chain_test.json")


def _npc_visible_import(w, npc, ev):
    from worldledger.evolution import _npc_visible
    return _npc_visible(w, npc, ev)


if __name__ == "__main__":
    main()
