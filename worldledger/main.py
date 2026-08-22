"""世界账本 · 类 galgame 文本原型 REPL 入口。

用法：python -m worldledger.main
"""
from __future__ import annotations

import os
import sys

from . import cards, evolution, interpreter, worldgen
from .event import event_identity_note
from .llm import get_llm
from .store import (DEFAULT_SAVE_PATH, Door, NPC, Universe, World,
                    active_items, game_time, mood_now, scene_changes,
                    weather_now)

WELCOME = """\
══════════════════════════════════════════
  世界账本 · 文本原型 v0
  一句话起世界 · 探索 · 天变 · 门
  输入「帮助」查看命令
══════════════════════════════════════════
"""


def _state_runtime_enabled() -> bool:
    """Read the optional generic validation switch for the CLI heartbeat."""
    value = os.environ.get("WORLDLEDGER_VALIDATE_WITH_STATE_RUNTIME", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}

HELP = """\
创建世界 <描述>   一句话生成世界（场景 + NPC + 法则）
看看              观察当前场景
去 <场景名>       移动到相邻场景
聊 <NPC名>        与 NPC 对话（输入 1/2/3 选行动，或直接输入对话，输入 离开 退出）
动 <NPC名> <动作> 对 NPC 做动作（握手/拥抱/亲吻/推开…，接受与否由 NPC 按人设与关系裁决）
创建NPC <名字>：<人设>  手动创建 NPC（带性格/记忆可选）
导入NPC <路径或世界:NPC名>  导入角色卡 / 跨世界借人
导出NPC <名字>    导出角色卡（worldledger_cards/）
设记忆 <NPC名> <内容>     给 NPC 写入一条记忆
改法则 <文本>     天变：修改世界法则，即时重解释
建门 <新世界描述> 生成新世界并建立互连的门
过门 <门名>       穿过门进入另一个世界
世界列表          查看所有世界
关系              查看当前世界 NPC 与你的关系
回放              打印事件日志
存档 / 读档       保存与加载（用户数据目录，可用 WORLDLEDGER_SAVE_PATH 覆盖）
帮助 / 退出
"""


def _find_scene_by_name(world: World, name: str):
    for scene in world.scenes.values():
        if scene.name == name:
            return scene
    return None


def _find_npc_by_name(world: World, name: str):
    for npc in world.npcs.values():
        if npc.name == name:
            return npc
    return None


def _print_scene_changes(world: World, scene_id: str) -> None:
    """重返场景：从账本机械导出你不在时这里的变化（零 LLM）。

    阅读进度是玩家状态（player["seen"]），不是世界状态——
    不进 social，不进模型载荷。
    """
    seen = world.player.setdefault("seen", {})
    seen_indices = world.player.setdefault("seen_event_indices", {})
    since = int(seen.get(scene_id, 0))
    if scene_id in seen_indices:
        after_index = int(seen_indices[scene_id])
    else:
        # Old saves only have a turn cursor. Convert it to the last event
        # known at that turn; new reads then use the lossless index cursor.
        after_index = -1
        for index, event in enumerate(world.events):
            if event.turn <= since:
                after_index = index
    changes = scene_changes(world, scene_id, since, limit=4,
                            after_index=after_index, include_index=True)
    if changes:
        print("\n你不在的时候，这里变了：")
        for c in changes:
            print(f"  · {c['fact']}（#{c['event_id']}｜因：{c['cause']}）")
    if changes:
        seen_indices[scene_id] = changes[-1]["_event_index"]
        seen[scene_id] = world.events[changes[-1]["_event_index"]].turn
    else:
        seen_indices[scene_id] = len(world.events) - 1
        seen[scene_id] = world.turn


def _look(world: World) -> str:
    loc = world.player.get("location", "")
    scene = world.scenes.get(loc)
    if scene is None:
        return "（你不在任何场景中）"
    lines = [f"【{scene.name}】{scene.description}",
             f"天气：{weather_now(world)}｜人心：{mood_now(world)}"]
    here_npcs = []
    for nid in scene.npcs:
        npc = world.npcs.get(nid)
        if npc is not None:
            here_npcs.append(f"{npc.name}（{npc.state.activity}·"
                             f"{npc.state.mood}）")
    lines.append("在场：" + ("、".join(here_npcs) if here_npcs else "无人"))
    # 物态：场景物品表（活跃窗口）——物品是场景的一等状态
    if scene.items:
        names = [i["name"] for i in active_items(scene)]
        if len(scene.items) > len(names):
            names.append("以及一些杂物")
        lines.append("物态：" + "、".join(names))
    if world.player.get("items"):
        lines.append("携带：" + "、".join(
            i["name"] for i in world.player.get("items", [])))
    for nid in scene.npcs:
        npc = world.npcs.get(nid)
        if npc is not None and not npc.state.can_act:
            continue
        if npc is not None and npc.state.pending_opener:
            lines.append(f"「{npc.name}」叫住了你："
                         f"『{npc.state.pending_opener}』（聊 {npc.name} 回应）")
        if npc is not None and npc.state.action.text:
            pct = int(npc.state.action.progress * 100)
            lines.append(f"{npc.name} 正在做的事：{npc.state.action.text}"
                         f"（{pct}%，进行中）")
    notes = [e for e in world.events if e.kind == "note_left"
             and e.payload.get("event_params", {}).get("location") == loc]
    for note in notes[-2:]:
        lines.append(f"纸条：{note.summary}")
    if scene.exits:
        parts = []
        for e in scene.exits:
            if e not in world.scenes:
                continue
            nxt = world.scenes[e]
            parts.append(f"{nxt.name}（雾中）" if not nxt.generated else nxt.name)
        lines.append("可去：" + "、".join(parts))
    doors = [d for d in world.doors.values() if d.to_world]
    if doors:
        lines.append("门：" + "、".join(f"{d.id}→{d.to_world}" for d in doors))
    return "\n".join(lines)


def dialogue_loop(llm, world: World, npc: NPC):
    print(f"\n—— 你走向了 {npc.name}（关系 {npc.relationship}）——")
    print(f"「{npc.persona}」")
    while True:
        raw = input("你 > ").strip()
        if not raw:
            continue
        if raw in ("离开", "再见", "退出"):
            break
        result = interpreter.dialogue_turn(llm, world, npc, raw)
        if result.reaction:
            print(f"\n{npc.name}：{result.reaction}")
        if result.reply:
            print(f"\n{npc.name}：{result.reply}")
        if result.action:
            print(f"\n{npc.name} 开始：{result.action}")
        if result.choices:
            print("选择：" + "  ".join(
                f"{i + 1}.{c}" for i, c in enumerate(result.choices)))
        print(f"（关系 {result.relationship}）")
        for w in result.grounding_warnings:
            print(f"⚠ 一致性提示：{w}")


def create_world(llm, universe: Universe, description: str) -> World:
    name = f"世界{len(universe.worlds) + 1}"
    world = worldgen.generate_world(llm, name, description)
    universe.worlds[name] = world
    universe.current = name
    print(f"\n✨ 世界「{name}」诞生")
    print(f"氛围基调：{world.law_profile.atmosphere}")
    print(f"期望文本：{description}")
    if world.law_profile.laws:
        print("法则：")
        for law in world.law_profile.laws:
            print(f"  · {law.trigger} → {law.effect}")
    else:
        print("法则：（无特殊法则，平凡世界）")
    print(_look(world))
    return world


def build_door(llm, universe: Universe, description: str):
    here = universe.here
    new_world = create_world(llm, universe, description)
    door_id = f"d-{len(here.doors) + 1}"
    back_id = f"d-{len(new_world.doors) + 1}"
    here.doors[door_id] = Door(
        id=door_id, to_world=new_world.name,
        to_scene=next(iter(new_world.scenes)), note="通往新世界")
    new_world.doors[back_id] = Door(
        id=back_id, to_world=here.name,
        to_scene=here.player.get("location", ""), note="返回旧世界")
    here.log("door_crossed", f"建立了通往「{new_world.name}」的门 {door_id}",
             "建门命令", {"door": door_id})
    print(f"\n🚪 门已建立：{door_id} ⇄ {new_world.name}")


def cross_door(llm, universe: Universe, door_id: str):
    here = universe.here
    if not evolution.player_is_actionable(here):
        print(f"你当前{evolution.player_condition(here)}，无法穿过门。")
        return
    door = here.doors.get(door_id)
    if door is None:
        print(f"没有叫「{door_id}」的门")
        return
    target = universe.worlds.get(door.to_world)
    if target is None:
        print("门另一侧的世界不存在")
        return
    here.log("door_crossed", f"穿过门 {door_id} 去往「{target.name}」",
             "过门命令", {"door": door_id})
    universe.current = target.name
    target.player["location"] = door.to_scene or next(iter(target.scenes))
    for summary in worldgen.ensure_scene(llm, target,
                                         target.player["location"]):
        print(f"✨ {summary}")
    print(f"\n🚪 你穿过门，抵达「{target.name}」")
    print(_look(target))


def repl(llm, universe: Universe):
    print(WELCOME)
    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        try:
            if cmd in ("帮助", "help"):
                print(HELP)
            elif cmd == "创建世界":
                if not arg:
                    print("用法：创建世界 <描述>")
                    continue
                create_world(llm, universe, arg)
            elif cmd == "看看":
                here = universe.here
                sid = here.player.get("location", "")
                evolution.catch_up_scene(llm, here, sid)
                print(_look(here))
                _print_scene_changes(here, sid)
            elif cmd == "去":
                here = universe.here
                if not evolution.player_is_actionable(here):
                    print(f"你当前{evolution.player_condition(here)}，无法移动。")
                    continue
                scene = _find_scene_by_name(here, arg)
                loc = here.player.get("location", "")
                cur = here.scenes.get(loc)
                if scene is None:
                    print("没有这个场景（试试「看看」里的可去列表）")
                elif cur is not None and scene.id not in cur.exits:
                    print("那里不在此处可去（先到相邻场景）")
                else:
                    errors = evolution.move_player(here, scene.id, cause="去命令")
                    if errors and here.player.get("location", "") != scene.id:
                        print(errors[0])
                        continue
                    # 贴片波：雾中场景首次抵达时生成
                    for summary in worldgen.ensure_scene(llm, here, scene.id):
                        print(f"✨ {summary}")
                    evolution.catch_up_scene(llm, here, scene.id)
                    print(_look(here))
                    _print_scene_changes(here, scene.id)
            elif cmd == "聊":
                npc = _find_npc_by_name(universe.here, arg)
                if npc is None:
                    print("没有这个 NPC（试试「看看」里的在场列表）")
                else:
                    # 雾中人：先兑现其驻地贴片，再对话
                    home = universe.here.scenes.get(npc.state.location)
                    if home is not None and not home.generated:
                        for summary in worldgen.ensure_scene(
                                llm, universe.here, home.id):
                            print(f"✨ {summary}")
                    dialogue_loop(llm, universe.here, npc)
            elif cmd == "动":
                # 玩家动作：动 <NPC名> <动作描述>（握手/拥抱/亲吻/推开…）
                if " " not in arg:
                    print("用法：动 <NPC名> <动作描述>")
                    continue
                npc_name, action = arg.split(" ", 1)
                npc = _find_npc_by_name(universe.here, npc_name)
                if npc is None:
                    print("没有这个 NPC（试试「看看」里的在场列表）")
                else:
                    result = interpreter.player_action(
                        llm, universe.here, npc, action)
                    print(f"\n你 对 {npc.name}：{action}")
                    print(f"{npc.name}：{result.reply}")
                    verdict = "接受" if result.accepted else "拒绝"
                    print(f"（动作被{verdict}｜关系 {result.relationship}）")
            elif cmd == "创建NPC":
                if "：" not in arg:
                    print("用法：创建NPC <名字>：<人设>")
                    continue
                name, persona = arg.split("：", 1)
                errors = cards.create_npc(universe.here, name, persona,
                                          llm=llm)
                if errors:
                    print("创建失败：" + "；".join(errors))
                else:
                    print(f"✨ NPC「{name.strip()}」登场（当前场景）")
            elif cmd == "导入NPC":
                if not arg:
                    print("用法：导入NPC <路径> 或 <世界名>:<NPC名>")
                    continue
                if ":" in arg:
                    wname, npc_name = arg.split(":", 1)
                    src = universe.worlds.get(wname)
                    if src is None:
                        print(f"没有世界「{wname}」")
                        continue
                    errors = cards.import_from_world(universe.here, src,
                                                     npc_name, llm=llm)
                else:
                    try:
                        import json as _json
                        card = _json.loads(
                            open(arg, encoding="utf-8").read())
                    except (OSError, ValueError) as e:
                        print(f"读卡失败：{e}")
                        continue
                    errors = cards.import_npc(universe.here, card, llm=llm)
                if errors:
                    print("导入失败：" + "；".join(errors))
                else:
                    print("✨ 导入成功")
            elif cmd == "导出NPC":
                npc = _find_npc_by_name(universe.here, arg)
                if npc is None:
                    print("没有这个 NPC")
                    continue
                path = cards.export_npc(npc)
                print(f"已导出角色卡 → {path}")
            elif cmd == "设记忆":
                if " " not in arg:
                    print("用法：设记忆 <NPC名> <内容>")
                    continue
                npc_name, content = arg.split(" ", 1)
                npc = _find_npc_by_name(universe.here, npc_name)
                if npc is None:
                    print("没有这个 NPC")
                    continue
                errors = cards.inject_memory(universe.here, npc, content,
                                             llm=llm)
                if errors:
                    print("写入失败：" + "；".join(errors))
                else:
                    print(f"已写入 {npc.name} 的记忆")
            elif cmd == "改法则":
                if not arg:
                    print("用法：改法则 <新法则文本>")
                    continue
                errors = interpreter.change_law(llm, universe.here, arg)
                if errors:
                    print("天变失败：" + "；".join(errors))
                else:
                    prof = universe.here.law_profile
                    print(f"\n⚡ 天变！法则档案 v{prof.version}")
                    for law in prof.laws:
                        print(f"  · {law.trigger} → {law.effect}")
            elif cmd == "建门":
                if not arg:
                    print("用法：建门 <新世界描述>")
                    continue
                build_door(llm, universe, arg)
            elif cmd == "过门":
                if not arg:
                    print("用法：过门 <门名>")
                    continue
                cross_door(llm, universe, arg)
            elif cmd == "世界列表":
                for name, w in universe.worlds.items():
                    mark = " ← 当前" if name == universe.current else ""
                    print(f"· {name}{mark}｜{w.law_profile.atmosphere}｜"
                          f"场景 {len(w.scenes)} 个｜NPC {len(w.npcs)} 名｜"
                          f"turn {w.turn}")
            elif cmd == "关系":
                for npc in universe.here.npcs.values():
                    print(f"· {npc.name}：关系 {npc.relationship}｜"
                          f"记忆 {len(npc.memories)} 条｜"
                          f"{npc.state.activity}·{npc.state.mood} "
                          f"@ {npc.state.last_time or '尚未活动'}")
            elif cmd == "回放":
                for e in universe.here.events:
                    identity = event_identity_note(
                        universe.here, e.kind, e.payload)
                    print(f"[{e.turn}·{game_time(e.turn)}] {e.kind}｜"
                          f"{e.summary}{identity}｜因：{e.cause}")
            elif cmd == "存档":
                universe.save()
                print(f"已存档 → {DEFAULT_SAVE_PATH}")
            elif cmd == "读档":
                new_u = Universe.load()
                universe.worlds = new_u.worlds
                universe.current = new_u.current
                print(f"已读档，当前世界「{universe.current}」")
            elif cmd in ("退出", "exit", "quit"):
                break
            else:
                print("未知命令，输入「帮助」查看")
        except Exception as e:  # REPL 永不崩：回显错误继续
            print(f"出错：{e}")
        # 世界心跳：你不在的场景也在过日子（事件日志继续生长）
        try:
            if universe.current and universe.worlds:
                evolution.world_pulse(
                    llm, universe.here,
                    use_state_runtime=_state_runtime_enabled())
        except Exception as e:
            print(f"心跳出错：{e}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    llm = get_llm()
    universe = Universe()
    try:
        universe = Universe.load()
        print(f"检测到存档，已读入（当前世界「{universe.current}」）")
    except (OSError, KeyError):
        pass
    except ValueError as e:
        print(f"存档未读入：{e}")
    repl(llm, universe)


if __name__ == "__main__":
    sys.exit(main())
