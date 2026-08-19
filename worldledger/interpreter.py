"""对话与行动裁决：检索 → 推理 → 写回。

统一循环的游玩侧实例：检索世界片段 → LLM 推理后果 →
机械裁决法则触发 → 写回世界库（记忆、关系、事件日志）。
"""
from __future__ import annotations

import json as _json
from dataclasses import dataclass, field

from . import physics
from .event import apply_item_patch, emit
from .evolution import (catch_up, cooldown_ready, grounding_warnings,
                         mood_label, on_law_change, player_condition,
                         player_is_actionable,
                         propose_proactive, push_mood, _begin_action,
                         _normalize_action_destination, _npc_visible)
from .llm import BaseLLM
from .store import (NPC, World, active_items, canonical_targets,
                    experience_payload, experience_window,
                    memory_gaps_payload, mood_now, resolve_target,
                    scene_associations, target_snapshot, touch_items,
                    weather_now)


@dataclass
class DialogueResult:
    reply: str
    reaction: str = ""
    action: str = ""
    choices: list[str] = field(default_factory=list)
    law_triggers: list[str] = field(default_factory=list)  # 触发的法则 id
    relationship: int = 0
    grounding_warnings: list[str] = field(default_factory=list)  # 一致性软校验


# —— 玩家动作层：自由裁决 ——
# 引擎不定义动作类型清单、不定义关系阈值表——什么动作、什么分寸，
# 由 AI 按角色人设、关系、情绪自己判断。引擎只守三件事：
# 幅度有界（钳制）、引用真实、全程留痕。
_PLAYERACT_SYSTEM = """TASK:PLAYERACT
你是角色的动作裁决器。玩家对角色做了一个动作。由你裁决角色是否接受，
并给出回应。输出严格 JSON，不要输出 JSON 以外的任何内容。

JSON 契约：
{
  "accepted": true或false,
  "reply": "角色的回应（一两句话，按人设演）",
  "relationship_delta": 动作后关系的变化（-20 到 20 的整数）,
  "mood_delta": 动作对角色情绪的影响（-0.8 到 0.8）,
  "memory_importance": 0.0 到 1.0（被做了动作，角色多半会记住）,
  "memory": "角色对这次动作记住的内容（第一人称「我」）",
  "targets": ["本次动作实际作用到的实体引用：npc:/item:/scene:/player；
    省略时默认是当前角色，最多 3 个"],
  "law_ids": ["触发的法则 id（没有则空数组）"],
  "days": 这次动作的时间成本（天）：一句问候是一瞬（0.001 左右），
         一起喝杯茶是半刻，结伴赶路是半天，睡一觉是一天
}

裁决要求：
- persona_origin / trait_origins 只是这个人的起点，不是永久行为指令。
  接受与否必须综合 lived_experiences、beliefs、与玩家的关系和此刻状态；
  后来的真实经历可以改变早期性格。同一个动作，由这个人实际活过的历史
  决定，不由动作类型或固定性格标签决定。
- lived_experiences 只含当前可访问的经历。memory_gaps 只说明某段时间为空白，
  不得据此编造或泄露那段时间由别人做过什么。
- 拒绝也要留下痕迹：角色会记住「他对我做了什么」。
- targets 必须只引用当前场景中真实可见的角色、物品、场景或 player；
  不要把动作文本里的自然语言名称当成已绑定目标，必须返回稳定引用。
"""


@dataclass
class ActionResult:
    accepted: bool
    reply: str
    relationship: int = 0
    law_triggers: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)


def _player_action_target_refs(world: World, npc: NPC) -> list[str]:
    """当前玩家动作可绑定的实体集合。

    玩家动作仍以一个 NPC 作为裁决者，但动作本身可以同时作用于同场景
    的物品、地点、其他角色或玩家。远处实体不进入候选集，避免模型把
    自由文本里的名字误当成已经接触到的对象。
    """
    scene_id = str(world.player.get("location", ""))
    scene = world.scenes.get(scene_id)
    refs = {f"npc:{npc.id}", f"scene:{scene_id}", "player"}
    if scene is None:
        return sorted(refs)
    refs.update(f"npc:{nid}" for nid in scene.npcs if nid in world.npcs)
    refs.update(f"item:{item.get('id')}" for item in active_items(scene)
                if item.get("id"))
    return sorted(refs)


_DIALOGUE_SYSTEM = """TASK:DIALOGUE
你是世界里的角色扮演裁决器。根据世界状态与玩家输入，生成严格 JSON，
不要输出 JSON 以外的任何内容。

JSON 契约：
{
  "reply": "NPC 说出的话；可为空字符串",
  "reaction": "当场可见的非语言反应；可为空字符串",
  "action": null 或 {"text": "NPC 开始做什么", "location": "场景id（可选）",
    "days": 持续天数, "targets": ["相关实体引用"],
    "requires": ["完成时必须仍同场的实体引用"]},
  "choices": ["玩家接下来可选的 2-3 个动作"],
  "law_ids": ["本次对话触发到的法则 id（没触发就是空数组）"],
  "relationship_delta": -2 到 2 之间的整数,
  "mood_delta": -0.2 到 0.2 之间的数（这次对话对 NPC 情绪的影响）,
  "memory_importance": 0.0 到 1.0 之间的数（这条记忆的重要度）,
  "memory": "NPC 对这次对话记住的内容（一句话，第一人称「我」；
    不要出现「玩家」「NPC」等系统术语）。
    如果玩家在这轮对话里给出了关键事实（名字、秘密、承诺、物件、委托），
    memory 必须用原词保留该事实本身——不许只写「他说了一个名字」，
    要写「他说他叫黑猫」",
  "memory_keywords": ["0-4 个检索关键词，将来有人问起时靠它们找到这条记忆；
    只写名词与专名（人名、地名、物件、身份），不要写句子"],
  "memory_refs": ["本轮回答实际依赖的 lived_experiences.id；没有则空数组"],
  "item_patches": [{"op": "add|remove|change", "item": "物品id",
    "location": "场景id", "name": "名称", "note": "备注"}]
}

裁决要求：
- 角色不必回答。reply / reaction / action 三者至少给一个，
  也可组合。沉默、移开视线、装作没听见、反问、转身离开都是
  合法反应，由人设、关系、心情和现场决定，不要为了有问必答强行说话。
- reaction 只写当场表现，不得声称已移动到别处、已改变物品或
  已与第三人互动。需要持续、移动或改变世界的反应写进 action。
- persona_origin / trait_origins 只是人物出生或导入时的起点。当前说话与
  行动必须从 lived_experiences、beliefs、关系和此刻状态共同长出来；
  后来的经历可以改变、细化甚至推翻初始人设。不得只复读 persona。
- lived_experiences 只含人物现在能想起的经历；memory_gaps 没有事件内容。
  不得让角色说出被封锁记忆，也不得从断档反推出借身者具体做过什么。
- 世界法则（laws）是物理级约束：
  法则触发条件满足时，可见反应必须体现法则后果，law_ids 必须列出。
- 回复里不要出现世界法则本身的名字，要自然地演出来。
- relationship_delta 要反映这次对话的关系变化：坦白、共情、被信任
  给正数；敷衍、冒犯给负数；无关痛痒的寒暄给 0。
- item_patches 是玩家在本轮对话里对物品的动作（拿起信、放下杯子）：
  remove 的 item id 必须是 scene_items 里真实存在的；add 给新 id 和名称；
  玩家没碰物品就是空数组。
"""


def _mark_scene_seen(world: World) -> None:
    """玩家在场景里做过事 = 这个场景是「你在过的地方」。

    重返场景视图只报告「你不在时」的变化——你刚行动过的地方，
    不该把你自己做过的事当新闻报给你。通用规则，与互换无关。
    """
    seen = world.player.setdefault("seen", {})
    seen[world.player.get("location", "")] = world.turn


def dialogue_turn(llm: BaseLLM, world: World, npc: NPC,
                  player_input: str) -> DialogueResult:
    """一轮对话：检索世界片段 → LLM 裁决 → 法则机械生效 → 写回。"""
    actor = world.actor_for_body(npc)
    if not player_is_actionable(world):
        return DialogueResult(reply=f"你当前{player_condition(world)}，无法开口。",
                              relationship=actor.relationship)
    if not npc.state.can_act:
        return DialogueResult(reply=f"{npc.name}此刻无法回应。",
                              relationship=actor.relationship)
    # 读取补算：先把 NPC 的状态推进到当前回合，再开始对话
    catch_up(llm, world, npc)
    # 补算可能让 NPC 离开。此时玩家的话没有被她听见，不能继续走
    # 对话裁决或把远处的回复写进当前场景。
    if npc.state.location != world.player.get("location", ""):
        npc.state.pending_opener = ""
        return DialogueResult(
            reply=f"{npc.name}已经不在这里了。",
            relationship=actor.relationship,
        )
    npc.state.pending_opener = ""  # 回应了她的主动开口
    pscene = world.scenes.get(world.player.get("location", ""))
    if pscene is not None:
        touch_items(pscene, world.turn)  # 读取即保鲜：物品进入切片 = 被提及
    lived = experience_window(
        actor, world.turn, query=player_input,
        focus_ids=actor.state.memory_focus, limit=12)
    lived_payload = experience_payload(lived)
    state = {
        "player": world.player.get("profile", {}),
        "world": {
            "atmosphere": weather_now(world),
            "world_mood": mood_now(world),
            "laws": [{"id": l.id, "trigger": l.trigger, "effect": l.effect,
                      "intensity": l.intensity}
                     for l in world.law_profile.laws],
            "facts": list(world.facts),  # 世界档案：稳定归属/规则
            "turn": world.turn,
            "recent_events": [
                {"kind": e.kind, "summary": e.summary}
                for e in world.events[-8:]
                if _npc_visible(world, npc, e)
            ][-5:],
            # 场景级记忆：当前场景的变化 + 与之关联的场景
            "scene_recent": [r["summary"] for r in (pscene.recent[-5:]
                                                    if pscene else [])],
            "associations": [
                {"scene": s.name, "strength": st}
                for s, st in scene_associations(
                    world, world.player.get("location", ""))
            ],
            # 物品表（读取有界）：玩家所在场景的活跃物品
            "scene_items": [
                {"id": i["id"], "name": i["name"], "note": i.get("note", "")}
                for i in (active_items(pscene) if pscene else [])
            ],
        },
        "npc": {
            "id": npc.id, "name": npc.name,
            "actor": {"id": actor.id, "name": actor.name},
            "body": {"id": npc.id, "name": npc.name},
            "persona_origin": actor.persona,
            "trait_origins": actor.traits,
            "persona": actor.persona, "traits": actor.traits,
            "relationship": actor.relationship,
            "experience_count": len(actor.memories),
            "lived_experiences": lived_payload,
            # 旧模型兼容：同一有界切片的纯文本投影，不再塞入全部经历。
            "memories": [m.content for m in lived],
            "beliefs": actor.beliefs[-5:],
            "memory_gaps": memory_gaps_payload(actor),
            "goals": actor.goals,
            "state": {"activity": npc.state.activity,
                      "mood": actor.state.mood,
                      "mood_value": actor.state.mood_value,
                      "facts": [dict(f) for f in actor.state.facts[-12:]]},
            "action": {"text": npc.state.action.text,
                       "location": npc.state.action.location,
                       "progress": npc.state.action.progress}
            if npc.state.action.text else None,
        },
        "player_input": player_input,
    }
    import json as _json
    data = llm.chat_json(_DIALOGUE_SYSTEM, _json.dumps(state,
                                                       ensure_ascii=False))

    supplied_ids = {row["id"] for row in lived_payload}
    cited = [str(ref) for ref in data.get("memory_refs", [])
             if str(ref) in supplied_ids]
    if cited:
        actor.state.memory_focus = cited[-4:]

    reply = str(data.get("reply") or "").strip()
    reaction = str(data.get("reaction") or "").strip()
    raw_action = data.get("action")
    action_text = ""
    law_ids = [str(x) for x in data.get("law_ids", [])]

    # 机械裁决：只有档案里真实存在的法则才能触发（包络线之外的触发无效）
    valid_laws = {law.id: law for law in world.law_profile.laws}
    triggered = []
    for law_id in law_ids:
        law = valid_laws.get(law_id)
        if law is not None:
            triggered.append(law_id)
            marker = f"[法则触发] {law.effect}"
            if reply:
                reply += "\n" + marker
            else:
                reaction = (reaction + "\n" + marker).strip()

    # 一致性软校验：回复里提及的世界实体，若不在「检索切片」里，浮出提示。
    # 必须在写回之前算——写回后自己的记忆就会污染切片。
    slice_text = " ".join(
        [npc.name, actor.name, actor.persona, world.law_profile.atmosphere]
        + [m.content for m in lived]
        + [e.summary for e in world.events[-8:]
           if _npc_visible(world, npc, e)]
        + [world.scenes[s].name for s in
           [npc.state.location, world.player.get("location", "")]]
        + [world.npcs[n].name for n in
           world.scenes.get(npc.state.location, world.scenes.get(
               next(iter(world.scenes), ""))).npcs]
    )
    proposed_action_text = (str(raw_action.get("text", ""))
                            if isinstance(raw_action, dict) else "")
    warnings = grounding_warnings(
        world, slice_text, " ".join((reply, reaction, proposed_action_text)))

    # 因先于果：NPC 的私人记忆、物态变化和回应，都只能发生在玩家
    # 已经开口之后。账本回放必须保住这条顺序。
    emit(world, "player_said",
         {"npc": npc.id, "content": str(player_input)[:120],
          "location": world.player.get("location", "")},
         cause="玩家对话")

    # 反应与说话正交：沉默、目光和手势是玩家当场看见的事，
    # 不伪装成台词。持续行动则复用已有的动作承诺与世界钟。
    if reaction:
        errors = emit(world, "npc_reacted", {
            "npc": npc.id, "reaction": reaction[:160],
            "location": npc.state.location,
        }, cause="玩家对话")
        if errors:
            reaction = ""

    action_applied = False
    if isinstance(raw_action, dict) and raw_action.get("text"):
        params = {
            "npc": npc.id,
            "action": str(raw_action.get("text", "")).strip(),
            "location": str(raw_action.get("location", "")).strip(),
            "days": raw_action.get("days", 0.0),
            "targets": raw_action.get("targets", []),
            "requires": raw_action.get("requires", []),
        }
        _normalize_action_destination(world, params)
        if not params["location"]:
            params.pop("location")
        try:
            days = float(params.get("days", 0.0) or 0.0)
        except (TypeError, ValueError):
            days = -1.0
        destination = str(params.get("location") or npc.state.location)
        if npc.state.action.text:
            action_error = "角色已有进行中的动作"
        elif destination != npc.state.location and days <= 0.0:
            action_error = "跨场景行动必须给出正时长"
        else:
            errors = emit(world, "npc_acted", params,
                          cause="对玩家话语的反应")
            action_error = errors[0] if errors else ""
        if not action_error:
            action_text = params["action"]
            _begin_action(world, npc, params, source_turn=world.events[-1].turn)
            world.remember_as(npc, world.events[-1].summary,
                           cause="对玩家话语的反应",
                           kind="npc_memory", importance=0.6)
            action_applied = True

    # 真正的“不回应”也是一个可观察结果，不再兜底成说出“……”。
    if not reply and not reaction and not action_applied:
        reaction = "没有作出可见的回应。"
        emit(world, "npc_reacted", {
            "npc": npc.id, "reaction": reaction,
            "location": npc.state.location,
        }, cause="玩家对话")

    # 写回：情绪（模型已结合经历裁决）、关系积分（意图加权 + 情绪加成）、
    # 记忆（带重要度，写入式）
    mood_delta = float(data.get("mood_delta", 0.0))
    push_mood(actor, mood_delta, f"与玩家的对话：{player_input}")
    actor.state.mood = mood_label(actor, actor.state.mood)
    delta = int(data.get("relationship_delta", 0))
    if actor.state.mood_value > 0.2:  # 情绪加成：她心情好，关系涨得更多
        delta += 1
    physics.adjust_relationship(actor, delta)
    memory = str(data.get("memory", "")).strip()
    if memory:
        world.remember_as(npc, memory, cause=player_input,
                       importance=float(data.get("memory_importance", 0.5)),
                       keywords=[str(k) for k in
                                 data.get("memory_keywords", [])][:4])

    # 场景物态补丁：玩家对物品的动作（remove = 拿走，进携带物）
    for patch in data.get("item_patches", []):
        if not isinstance(patch, dict):
            continue
        taken = None
        if patch.get("op") == "remove" and pscene is not None:
            iid = str(patch.get("item", ""))
            taken = next((i for i in pscene.items if i["id"] == iid), None)
        errors = apply_item_patch(world, patch, cause=player_input)
        if taken is not None and not errors:
            world.player.setdefault("items", []).append(dict(taken))

    # 对话本身不驱动别人的心跳；角色生活由统一世界脉冲处理。
    # 这里仅保留对话者的主动目标裁决，且冷却按真实世界时间计算。
    key = f"{npc.id}->proactive"
    if not action_applied and cooldown_ready(world, key, 6):
        world.social[key] = world.turn
        world.social_clock[key] = world.clock
        propose_proactive(llm, world, npc)

    # 你听到的回应也进账本（玩家视角的「我听到」是一等事实）
    if reply:
        world.log("dialogue", f"{npc.name}：{reply}", "玩家对话",
                  {"event_params": {
                      "npc": npc.id, "reply": str(reply)[:120],
                      "location": world.player.get("location", "")}})
    _mark_scene_seen(world)  # 你在这里说话 = 你在这里（放在写回后）

    return DialogueResult(reply=reply, reaction=reaction, action=action_text,
                          choices=[str(c) for c in data.get("choices", [])],
                          law_triggers=triggered,
                          relationship=actor.relationship,
                          grounding_warnings=warnings)


def player_action(llm: BaseLLM, world: World, npc: NPC,
                  action_text: str) -> ActionResult:
    """玩家动作：自由裁决 + 后果写回。

    接受与否由 AI 按角色人设、与玩家的关系、此刻情绪自由判断——
    引擎不设动作类型清单与关系阈值表（纪律留给账本，分寸留给角色）。
    引擎只守三件事：幅度有界（全局钳制）、引用真实、全程留痕。
    """
    actor = world.actor_for_body(npc)
    if not player_is_actionable(world):
        return ActionResult(accepted=False,
                            reply=f"你当前{player_condition(world)}，无法行动。",
                            relationship=actor.relationship)
    if not npc.state.can_act:
        text = str(action_text).strip()
        emit(world, "action_refused",
             {"npc": npc.id, "action": text[:120],
              "reason": f"{npc.name}此刻无法回应"}, cause="玩家动作")
        return ActionResult(accepted=False, reply=f"{npc.name}此刻无法回应。",
                            relationship=actor.relationship)
    catch_up(llm, world, npc)
    # 与对话相同：补算后若 NPC 已经离开，动作没有作用对象。
    # 只记录一次被拒绝的尝试，不伪造触碰、关系变化或 NPC 记忆。
    if npc.state.location != world.player.get("location", ""):
        text = str(action_text).strip()
        emit(world, "action_refused",
             {"npc": npc.id, "action": text[:120],
              "reason": f"{npc.name}已经不在这里了"}, cause="玩家动作")
        return ActionResult(accepted=False,
            reply=f"{npc.name}已经不在这里了。",
            relationship=actor.relationship)
    text = str(action_text).strip()
    lived = experience_window(
        actor, world.turn, query=text,
        focus_ids=actor.state.memory_focus, limit=12)
    data = llm.chat_json(_PLAYERACT_SYSTEM, _json.dumps({
        "npc": {
            "id": npc.id, "name": npc.name,
            "actor": {"id": actor.id, "name": actor.name},
            "body": {"id": npc.id, "name": npc.name},
            "persona_origin": actor.persona,
            "trait_origins": actor.traits,
            "persona": actor.persona, "traits": actor.traits,
            "relationship": actor.relationship,
            "experience_count": len(actor.memories),
            "lived_experiences": experience_payload(lived),
            "beliefs": actor.beliefs[-5:],
            "memory_gaps": memory_gaps_payload(actor),
            "state": {"activity": npc.state.activity,
                      "mood": actor.state.mood,
                      "mood_value": actor.state.mood_value,
                      "facts": [dict(f) for f in actor.state.facts[-12:]]},
        },
        "action": text,
        "target_options": [
            {"ref": ref, "snapshot": target_snapshot(world, ref)}
            for ref in _player_action_target_refs(world, npc)
        ],
        "world": {
            "atmosphere": weather_now(world),
            "laws": [{"id": l.id, "trigger": l.trigger,
                      "effect": l.effect} for l in world.law_profile.laws],
        },
    }, ensure_ascii=False))

    accepted = bool(data.get("accepted", False))
    reply = str(data.get("reply", "……"))

    raw_targets = data.get("targets")
    if raw_targets is None:
        target_refs = [f"npc:{npc.id}"]
        target_error = ""
    elif isinstance(raw_targets, list):
        supplied = [str(ref).strip() for ref in raw_targets
                    if isinstance(ref, str) and str(ref).strip()]
        target_refs = canonical_targets(world, supplied)
        allowed = set(_player_action_target_refs(world, npc))
        target_error = ""
        if len(target_refs) != len(supplied):
            target_error = "动作包含不存在的目标引用"
        elif any(ref not in allowed for ref in target_refs):
            target_error = "动作目标不在玩家当前可见范围"
        elif not target_refs:
            target_error = "动作必须至少绑定一个目标"
    else:
        target_refs = [f"npc:{npc.id}"]
        target_error = "动作 targets 必须是引用列表"
    if target_error:
        accepted = False
        reply = target_error
        data["law_ids"] = []

    # 引擎只守三件事之一：幅度有界（全局理智钳制，不规定「什么动作多少值」）
    rel_delta = max(-20, min(20, int(data.get("relationship_delta", 0))))
    mood_delta = max(-0.8, min(0.8,
                               float(data.get("mood_delta", 0.0))))
    if target_error:
        rel_delta = 0
        mood_delta = 0.0

    # 法则触发（机械校验：只有档案里真实存在的法则能触发）
    valid_laws = {l.id: l for l in world.law_profile.laws}
    triggered = []
    for law_id in data.get("law_ids", []):
        if str(law_id) in valid_laws:
            triggered.append(str(law_id))
            reply += f"\n[法则触发] {valid_laws[str(law_id)].effect}"

    # 写回：事件 + 关系 + 情绪 + 记忆（动作有它的时间成本）
    days = max(0.0, min(1.0, float(data.get("days", 0.0) or 0.0)))
    emit(world, "player_acted",
         {"npc": npc.id, "action": text[:120],
          "accepted": accepted,
          "location": world.player.get("location", ""),
          "targets": target_refs},
         cause="玩家动作", duration=days)
    if not accepted:
        emit(world, "action_refused",
             {"npc": npc.id, "action": text[:120],
              "reason": reply[:80], "targets": target_refs},
             cause="玩家动作")
    physics.adjust_relationship(actor, rel_delta)
    push_mood(actor, mood_delta, f"玩家动作：{text[:40]}")
    actor.state.mood = mood_label(actor, actor.state.mood)
    memory = str(data.get("memory", "")).strip()
    if memory and not target_error:
        world.remember_as(npc, memory, cause=f"动作：{text[:40]}",
                       kind="npc_memory",
                       importance=float(data.get("memory_importance", 0.7)))
    # 玩家动作的回应也是玩家亲历的事实；否则玩家视角只会看到
    # 「被接受」，而账本丢掉了 NPC 当时真正说了什么。
    world.log("dialogue", f"{npc.name}：{reply}", "玩家动作",
              {"event_params": {
                  "npc": npc.id, "reply": str(reply)[:120],
                  "location": world.player.get("location", ""),
                  "action": text[:120]}})
    _mark_scene_seen(world)  # 你在这里做事 = 你在这里
    return ActionResult(accepted=accepted, reply=reply,
                        relationship=actor.relationship,
                        law_triggers=triggered, targets=target_refs)

def change_law(llm: BaseLLM, world: World, request: str) -> list[str]:
    """天变：转译新法则文本 → 包络线校验 → 即时重解释。返回错误列表。"""
    import json as _json
    system = (
        "TASK:LAWCHANGE\n你是世界法则的转译器。用户的请求是对世界法则的"
        "修改（天变）。新法则集必须体现请求的语义变化；与请求矛盾的旧法则"
        "必须移除，无关的旧法则可保留。例如请求「人从不撒谎」时，必须写出"
        "一条与诚实相关的新法则（如「人不会撒谎 → 每句话都是完整的真相」），"
        "而不是保留旧的撒谎法则。"
        "输出严格 JSON：{\"atmosphere\": \"氛围（保持不变也可改写）\", "
        "\"laws\": [{\"id\": \"law-N\", \"trigger\": \"触发条件\", "
        "\"effect\": \"触发后果\", \"intensity\": 0.0-1.0}]}。"
        "不要输出 JSON 以外的任何内容。"
    )
    current = {
        "atmosphere": world.law_profile.atmosphere,
        "laws": [{"id": l.id, "trigger": l.trigger, "effect": l.effect,
                  "intensity": l.intensity}
                 for l in world.law_profile.laws],
    }
    data = llm.chat_json(system, _json.dumps(
        {"current": current, "request": request}, ensure_ascii=False))

    from .store import Law, LawProfile
    new_profile = LawProfile(
        expectation=world.law_profile.expectation,
        atmosphere=str(data.get("atmosphere",
                                world.law_profile.atmosphere)),
        laws=[Law.from_dict(x) for x in data.get("laws", [])],
    )
    errors = physics.apply_law_change(world, new_profile)
    if not errors:
        on_law_change(world)  # 天变波及所有 NPC
    return errors
