"""世界展示：一句话 → 真模型跑一局有头有尾的体验。

输出是一段可分享的完整文本（世界的诞生、对话、天变、心跳、记忆）。
用法：python -m tools.showcase "世界描述"
"""
from __future__ import annotations

import sys

from worldledger import evolution, interpreter
from worldledger.llm import get_llm
from worldledger.worldgen import ensure_scene, generate_world

DESC = "永远在下雨的东京，少女只要祈祷就能让天空放晴，但每次放晴，她都在一点点消失"
SEP = "\n" + "─" * 56 + "\n"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    desc = sys.argv[1] if len(sys.argv) > 1 else DESC
    llm = get_llm()
    w = generate_world(llm, "天空世界", desc)
    print(SEP + f" 世界展示：{desc}" + SEP)
    print(f"✨ 世界「{w.name}」诞生")
    print(f"   氛围基调：{w.law_profile.atmosphere}")
    for law in w.law_profile.laws:
        print(f"   法则：{law.trigger} → {law.effect}")

    # 出生场景
    start = w.scenes[w.player["location"]]
    print(f"\n你醒来在【{start.name}】{start.description}")
    if start.items:
        print("物态：" + "、".join(i["name"] for i in start.items))

    # 去一个雾中的相邻场景，遇见第一个 NPC
    hero = None
    for e in start.exits:
        if e in w.scenes and not w.scenes[e].generated:
            w.player["location"] = e
            w.log("scene_entered", f"进入「{w.scenes[e].name}」", "去命令")
            for s in ensure_scene(llm, w, e):
                print(f"✨ {s}")
            scene = w.scenes[e]
            print(f"\n你走进【{scene.name}】{scene.description}")
            for nid in scene.npcs:
                if nid in w.npcs:
                    hero = w.npcs[nid]
                    break
            break
    if hero is None:
        for nid in start.npcs:
            if nid in w.npcs:
                hero = w.npcs[nid]
                break
    if hero is None:
        hero = next(iter(w.npcs.values()))
    # 主角是长出来的：按事件日志里的提及次数找故事的重心
    mentions = {nid: 0 for nid in w.npcs}
    for e in w.events:
        for nid, npc in w.npcs.items():
            if npc.name and npc.name in e.summary:
                mentions[nid] += 1
    # 故事重心 = 被世界事件提及最多的角色（没有则取目标最多者）
    focus = max(w.npcs.values(),
                key=lambda n: (mentions[n.id], len(n.goals)))
    hero = focus
    print(f"\n这个世界的故事，似乎一直围着【{hero.name}】转——"
          f"（{mentions[hero.id]} 次被世界提及）")
    print(f"人设：{hero.persona}")
    print(f"她的目标：{'、'.join(g['text'] for g in hero.goals)}")
    # 立绘提示词：文字世界到手后，脸只是一句提示词的事
    print(f"立绘提示词：二次元立绘，{hero.persona}；"
          f"世界观：{w.law_profile.atmosphere}，{w.description[:36]}；"
          f"新海诚风格，雨，半身像")

    # 三问
    questions = ["你叫什么名字？", "你能让雨停下来吗？",
                 "这么做，你要付出什么代价？"]
    for q in questions:
        r = interpreter.dialogue_turn(llm, w, hero, q)
        print(f"\n你 > {q}")
        print(f"{hero.name}：{r.reply}")

    # 天变
    print(SEP + "⚡ 天变：你对天空许了一个愿" + SEP)
    errors = interpreter.change_law(
        llm, w, "从此，这片天空的雨会记住每一个愿望")
    for e in errors:
        print(f"（驳回：{e}）")
    print(f"法则档案 v{w.law_profile.version}")
    for law in w.law_profile.laws:
        print(f"   法则：{law.trigger} → {law.effect}")

    # 世界心跳：你不在时
    print(SEP + "⏳ 时间流逝 30 回合：你没有说话，世界仍在发生" + SEP)
    w.pass_time(30)
    for s in evolution.world_pulse(llm, w):
        print(f"  · {s}")

    # 她的记忆与信念
    print(SEP + f"{hero.name} 的记忆（写入即记忆）" + SEP)
    for m in hero.memories[-8:]:
        print(f"  · [{m.turn}] {m.content}")
    if hero.beliefs:
        print(f"\n她的信念：{'、'.join(hero.beliefs)}")

    # 事件日志尾部
    print(SEP + "事件日志（尾部，可回放）" + SEP)
    for e in w.events[-10:]:
        print(f"  [{e.turn}·{e.kind}] {e.summary}｜因：{e.cause}")


if __name__ == "__main__":
    main()
