"""一键演示：跑完整个体验闭环（模板无关）。

用法：python -m tools.demo [世界描述]
LLM：优先读取 WORLDLEDGER_API_KEY 等环境变量；未配置时自动回落 Mock。
脚本不假设世界长什么样——场景、NPC、事件全部动态跟随。
"""
from __future__ import annotations

import sys

from worldledger import evolution, interpreter
from worldledger.llm import get_llm
from worldledger.store import Universe, game_time
from worldledger.worldgen import ensure_scene, generate_world

DEFAULT_DESC = "一座永远在下雨的小城，人们撒谎时会漏出真心话"

SEP = "\n" + "─" * 46 + "\n"


def _first_npc_in(world, scene_id):
    scene = world.scenes.get(scene_id)
    if scene:
        for nid in scene.npcs:
            if nid in world.npcs:
                return world.npcs[nid]
    return next(iter(world.npcs.values()), None)


def demo(desc: str = DEFAULT_DESC, llm=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    llm = llm or get_llm()
    u = Universe()
    print(SEP + " 世界账本 · 文本原型 v0 · 演示" + SEP)

    # 1. 创建世界
    world = generate_world(llm, "世界1", desc)
    u.worlds[world.name] = world
    u.current = world.name
    print(f"✨ 世界「{world.name}」诞生")
    print(f"   期望文本：{desc}")
    print(f"   氛围基调：{world.law_profile.atmosphere}")
    for law in world.law_profile.laws:
        print(f"   法则：{law.trigger} → {law.effect}")
    print()

    # 2. 探索：出生场景 → 雾中相邻场景（贴片式生成）
    start = world.scenes[world.player["location"]]
    print(f"你醒来在【{start.name}】{start.description}")
    if start.items:  # 物态：场景物品表（物品 = 场景一等状态）
        print("物态：" + "、".join(i["name"] for i in start.items))
    fog = [e for e in start.exits
           if e in world.scenes and not world.scenes[e].generated]
    hero = None
    if fog:
        print("可去：" + "、".join(
            f"{world.scenes[e].name}（雾中）" for e in fog))
        target = fog[0]
        world.player["location"] = target
        world.log("scene_entered",
                  f"进入「{world.scenes[target].name}」", "去命令")
        for s in ensure_scene(llm, world, target):
            print(f"✨ {s}")
        scene = world.scenes[target]
        print(f"你走进【{scene.name}】{scene.description}")
        hero = _first_npc_in(world, target)
        if hero:
            print(f"在场：{hero.name}")
            print(f"\n你 > 你在等谁？")
            r = interpreter.dialogue_turn(llm, world, hero, "你在等谁？")
            print(f"{hero.name}：{r.reply}")
            print(f"（关系 {r.relationship}）")
    else:
        hero = _first_npc_in(world, start.id)

    # 3. 天变
    print(SEP + "改法则：这座城的人从不撒谎" + SEP)
    errors = interpreter.change_law(llm, world, "这座城的人从不撒谎")
    if errors:
        print("天变失败：" + "；".join(errors))
    else:
        prof = world.law_profile
        print(f"⚡ 天变！法则档案 v{prof.version}")
        for law in prof.laws:
            print(f"   法则：{law.trigger} → {law.effect}")
    if hero:
        print(f"\n你 > 你究竟在等什么？")
        r = interpreter.dialogue_turn(llm, world, hero, "你究竟在等什么？")
        print(f"{hero.name}：{r.reply}")

    # 4. 建门
    print(SEP + "建门：一座永不落日的沙漠之城，火焰只会照亮不会灼伤" + SEP)
    new_world = generate_world(
        llm, "世界2", "一座永不落日的沙漠之城，火焰只会照亮不会灼伤")
    u.worlds[new_world.name] = new_world
    from worldledger.store import Door
    world.doors["d-1"] = Door(id="d-1", to_world=new_world.name,
                              to_scene=next(iter(new_world.scenes)),
                              note="通往新世界")
    new_world.doors["d-1"] = Door(id="d-1", to_world=world.name,
                                  to_scene=world.player.get("location", ""),
                                  note="返回旧世界")
    world.log("door_crossed", f"建立了通往「{new_world.name}」的门 d-1",
              "建门命令")
    print(f"🚪 门 d-1 ⇄ 「{new_world.name}」")
    print(f"   新世界氛围：{new_world.law_profile.atmosphere}")
    for law in new_world.law_profile.laws:
        print(f"   新法则：{law.trigger} → {law.effect}")

    # 5. 时间流逝：NPC 的生活在继续
    print(SEP + "时间流逝：12 回合后" + SEP)
    world.pass_time(12)
    loc = world.player.get("location", "")
    summaries = evolution.world_pulse(llm, world)
    if hero:
        where = (world.scenes[hero.state.location].name
                 if hero.state.location in world.scenes else "某处")
        print(f"你回来看看——{hero.name} 在「{where}」"
              f"（{hero.state.activity}·{hero.state.mood}）")
    print("\n（统一世界心跳）")
    for s in summaries:
        print(f"  · {s}")

    # 5.5 长时间缺席：世界事件 + 流言
    print(SEP + "又过了 25 回合：无人观察，世界仍在发生" + SEP)
    world.pass_time(25)
    print("你不在时（世界心跳）：")
    for s in evolution.world_pulse(llm, world):
        print(f"  · {s}")

    # 6. 角色的驱动力
    if hero:
        print(SEP + "角色的驱动力" + SEP)
        print(f"\n你 > 你在查什么？")
        r = interpreter.dialogue_turn(llm, world, hero, "你在查什么？")
        print(f"{hero.name}：{r.reply}")
        print("\n你静静等了一会儿……")
        for s in evolution.heartbeat(llm, world, hero):
            print(f"  · {s}")
        notes = [e for e in world.events if e.kind == "note_left"]
        if notes:
            print(f"纸条：{notes[-1].summary}")

    # 7. 跨门 + 返回（写入式记忆的证明）
    print(SEP + "过门 d-1 → 新世界" + SEP)
    u.current = new_world.name
    start2 = next(iter(new_world.scenes))
    for s in ensure_scene(llm, new_world, start2):
        print(f"✨ {s}")
    print(f"你抵达「{new_world.name}」："
          f"{new_world.scenes[start2].description}")
    print(SEP + "过门 d-1 → 回到旧世界" + SEP)
    u.current = world.name
    if hero:
        print(f"你回到「{world.name}」，{hero.name} 正在"
              f"「{world.scenes[hero.state.location].name}」"
              f"（{hero.state.activity}·{hero.state.mood}）。")
        print(f"\n{hero.name} 的记忆（写入即记忆）：")
        for m in hero.memories:
            print(f"  · [{m.turn}] {m.content}")

    # 8. 自由导入 NPC：角色卡 + 跨世界借人
    print(SEP + "自由导入 NPC：角色卡「信使」+ 跨世界借人" + SEP)
    from worldledger import cards
    errors = cards.import_npc(new_world, {
        "name": "信使", "persona": "披着油布雨衣的送信人，从不抬头，只递信。",
        "traits": {"寡言": True},
        "memories": ["曾在沙暴中弄丢过一封信。"], "relationship": 0},
        llm=llm)
    if errors:
        print("导入失败：" + "；".join(errors))
    else:
        messenger = next(n for n in new_world.npcs.values()
                         if n.name == "信使")
        print(f"「信使」出现在「{new_world.name}」")
        print("\n你 > 你有我的信吗？")
        r = interpreter.dialogue_turn(llm, new_world, messenger,
                                      "你有我的信吗？")
        print(f"信使：{r.reply}")
    if hero:
        errors = cards.import_from_world(new_world, world, hero.name, llm=llm)
        if not errors:
            clone = next(n for n in new_world.npcs.values()
                         if n.name == hero.name)
            print(f"\n「{hero.name}」跨世界迁入「{new_world.name}」，"
                  f"人设与 {len(clone.memories)} 条记忆完整保留。")
            r = interpreter.dialogue_turn(llm, new_world, clone,
                                          "你怎么会在这里？")
            print(f"{hero.name}：{r.reply}")

    # 9. 回放事件日志
    print(SEP + "事件日志（append-only，每条携带原因引用与游戏内时间戳）" + SEP)
    for e in world.events:
        print(f"[{e.turn}·{game_time(e.turn)}] {e.kind}｜{e.summary}｜"
              f"因：{e.cause}")
    print()


if __name__ == "__main__":
    import sys
    desc = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DESC
    demo(desc)
