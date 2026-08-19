"""事件固化：事件也是带类型的参数记录。

事件类型有 schema（参数、边界、引用完整性）；AI 只能提出「事件调用」，
包络线机械校验通过后才写入事件日志。事件与物品、法则一样：
可复现、可追溯、跨世界兼容（门的两端说同一种事件语言）。
"""
from __future__ import annotations

from .store import World, resolve_target

# 事件类型注册表：type → (描述, 参数 schema)
# 参数 spec：kind: text|float|int|id；required；min/max（数值与文本长度）
EVENT_TYPES: dict[str, tuple[str, dict]] = {
    "npc_moved": ("NPC 移动", {
        "npc": {"kind": "id", "required": True},
        "from": {"kind": "id", "required": True},
        "to": {"kind": "id", "required": True},
    }),
    "npc_interaction": ("行动者间搭话", {
        "npc": {"kind": "id", "required": True},
        "target": {"kind": "id", "required": True},
        "line": {"kind": "text", "required": True, "max": 120},
        "location": {"kind": "id", "required": False},
    }),
    "npc_reacted": ("NPC 当场反应", {
        "npc": {"kind": "id", "required": True},
        "reaction": {"kind": "text", "required": True, "max": 160},
        "location": {"kind": "id", "required": True},
    }),
    "world_event": ("世界事件（通用）", {
        "title": {"kind": "text", "required": True, "max": 60},
        "detail": {"kind": "text", "required": True, "max": 200},
        "location": {"kind": "id", "required": False},
        "intensity": {"kind": "float", "required": True, "min": 0.0,
                      "max": 1.0},
        "refs": {"kind": "refs", "required": False, "max_items": 8},
    }),
    "item_arrive": ("物品抵达", {
        "item": {"kind": "text", "required": True, "max": 60},
        "location": {"kind": "id", "required": True},
        "note": {"kind": "text", "required": False, "max": 120},
    }),
    "item_added": ("物品出现", {
        "item": {"kind": "id", "required": True},
        "name": {"kind": "text", "required": True, "max": 20},
        "location": {"kind": "id", "required": True},
        "note": {"kind": "text", "required": False, "max": 80},
    }),
    "item_removed": ("物品消失", {
        "item": {"kind": "id", "required": True},
        "name": {"kind": "text", "required": False, "max": 20},
        "location": {"kind": "id", "required": True},
        "note": {"kind": "text", "required": False, "max": 80},
    }),
    "item_transfer": ("物品转手", {
        "item": {"kind": "text", "required": True, "max": 40},
        "holder": {"kind": "text", "required": True, "max": 40},
        "location": {"kind": "id", "required": True},
    }),
    "fact_changed": ("世界设定变更", {
        "old": {"kind": "text", "required": False, "max": 80},
        "new": {"kind": "text", "required": True, "max": 80},
    }),
    "item_changed": ("物品跃变", {
        "item": {"kind": "id", "required": True},
        "name": {"kind": "text", "required": False, "max": 20},
        "location": {"kind": "id", "required": True},
        "note": {"kind": "text", "required": False, "max": 80},
    }),
    "weather_shift": ("天气变化", {
        "to": {"kind": "text", "required": True, "max": 30},
        "intensity": {"kind": "float", "required": False, "min": 0.0,
                      "max": 1.0},
    }),
    "scene_state_changed": ("局部场景状态变化", {
        "scene": {"kind": "id", "required": True},
        "fact": {"kind": "text", "required": True, "max": 80},
        "op": {"kind": "text", "required": True, "max": 10},
        "text": {"kind": "text", "required": False, "max": 160},
        "expires_clock": {"kind": "float", "required": False,
                           "min": 0.0, "max": 100000.0},
    }),
    "daily_life": ("日常小事", {
        "detail": {"kind": "text", "required": True, "max": 120},
        "location": {"kind": "id", "required": True},
        "intensity": {"kind": "float", "required": False, "min": 0.0,
                      "max": 0.4},
        "item": {"kind": "text", "required": False, "max": 40},
    }),
    "scene_generated": ("贴片生成", {
        "scene": {"kind": "id", "required": True},
    }),
    "npc_acted": ("NPC 主动行动", {
        "npc": {"kind": "id", "required": True},
        # 由引擎盖章，记录「谁的身体」与「谁在行动」；模型不能伪造。
        "body": {"kind": "id", "required": False},
        "actor": {"kind": "id", "required": False},
        "action": {"kind": "text", "required": True, "max": 120},
        "location": {"kind": "id", "required": False},
        # 由 emit 依据行动者当前状态盖章。location 是目的地，origin 才是
        # 行动开始时真正发生的现场；模型不能自行决定这个字段。
        "origin": {"kind": "id", "required": False},
        "travel": {"kind": "bool", "required": False},
        "place": {"kind": "text", "required": False, "max": 20},
        "days": {"kind": "float", "required": False, "min": 0.0,
                 "max": 30.0},
        "earliest_clock": {"kind": "float", "required": False, "min": 0.0,
                            "max": 100000.0},
        "latest_clock": {"kind": "float", "required": False, "min": 0.0,
                          "max": 100000.0},
        "targets": {"kind": "refs", "required": False, "max_items": 3},
        "requires": {"kind": "refs", "required": False, "max_items": 3},
    }),
    "npc_intent": ("NPC 短期打算变更", {
        "npc": {"kind": "id", "required": True},
        "intent": {"kind": "text", "required": False, "max": 120},
        "previous": {"kind": "text", "required": False, "max": 120},
        "earliest_clock": {"kind": "float", "required": False, "min": 0.0,
                            "max": 100000.0},
        "latest_clock": {"kind": "float", "required": False, "min": 0.0,
                          "max": 100000.0},
        "targets": {"kind": "refs", "required": False, "max_items": 3},
    }),
    "npc_state_changed": ("NPC 行动状态变更", {
        "npc": {"kind": "id", "required": True},
        "can_act": {"kind": "bool", "required": True},
        "condition": {"kind": "text", "required": True, "max": 20},
        "cause_event": {"kind": "text", "required": True, "max": 200},
    }),
    "note_left": ("NPC 留纸条", {
        "npc": {"kind": "id", "required": True},
        "location": {"kind": "id", "required": True},
        "content": {"kind": "text", "required": True, "max": 120},
    }),
    "goal_completed": ("NPC 达成目标", {
        "npc": {"kind": "id", "required": True},
        "goal": {"kind": "text", "required": True, "max": 200},
    }),
    "goal_emerged": ("NPC 长出目标", {
        "npc": {"kind": "id", "required": True},
        "goal": {"kind": "text", "required": True, "max": 200},
    }),
    "npc_initiated": ("NPC 主动开口", {
        "npc": {"kind": "id", "required": True},
        "line": {"kind": "text", "required": True, "max": 120},
    }),
    "action_done": ("NPC 完成主动动作", {
        "npc": {"kind": "id", "required": True},
        "body": {"kind": "id", "required": False},
        "actor": {"kind": "id", "required": False},
        "action": {"kind": "text", "required": True, "max": 200},
        "outcome": {"kind": "text", "required": True, "max": 200},
        "location": {"kind": "id", "required": False},
    }),
    "action_aborted": ("NPC 中止主动动作", {
        "npc": {"kind": "id", "required": True},
        "body": {"kind": "id", "required": False},
        "actor": {"kind": "id", "required": False},
        "action": {"kind": "text", "required": True, "max": 200},
    }),
    "collision": ("目标碰撞", {
        "a": {"kind": "id", "required": True},
        "b": {"kind": "id", "required": True},
        "thing": {"kind": "text", "required": True, "max": 60},
    }),
    "scene_extended": ("世界生长", {
        "scene": {"kind": "id", "required": True},
        "from": {"kind": "id", "required": True},
    }),
    "player_acted": ("玩家动作", {
        "npc": {"kind": "id", "required": True},
        "action": {"kind": "text", "required": True, "max": 120},
        "type": {"kind": "text", "required": False, "max": 20},
        "accepted": {"kind": "bool", "required": True},
        "location": {"kind": "id", "required": False},
        "targets": {"kind": "refs", "required": False, "max_items": 3},
    }),
    "player_said": ("玩家对话", {
        "npc": {"kind": "id", "required": True},
        "content": {"kind": "text", "required": True, "max": 120},
        "location": {"kind": "id", "required": False},
    }),
    "action_refused": ("玩家动作被拒绝", {
        "npc": {"kind": "id", "required": True},
        "action": {"kind": "text", "required": True, "max": 120},
        "type": {"kind": "text", "required": False, "max": 20},
        "reason": {"kind": "text", "required": False, "max": 80},
        "targets": {"kind": "refs", "required": False, "max_items": 3},
    }),
}


def validate_event(event_type: str, params: dict) -> list[str]:
    """包络线校验：类型已注册、必填齐全、数值与文本在界内。"""
    entry = EVENT_TYPES.get(event_type)
    if entry is None:
        return [f"未注册的事件类型：{event_type}"]
    _, schema = entry
    errors: list[str] = []
    for name, spec in schema.items():
        if spec.get("required") and name not in params:
            errors.append(f"事件 {event_type} 缺少参数 {name}")
    for name, value in params.items():
        spec = schema.get(name)
        if spec is None:
            errors.append(f"事件 {event_type} 携带未知参数 {name}")
            continue
        kind = spec["kind"]
        if kind in ("float", "int"):
            try:
                num = float(value)
            except (TypeError, ValueError):
                errors.append(f"事件 {event_type} 参数 {name} 不是数值")
                continue
            if "min" in spec and num < spec["min"]:
                errors.append(f"事件 {event_type} 参数 {name} 低于下界")
            if "max" in spec and num > spec["max"]:
                errors.append(f"事件 {event_type} 参数 {name} 超出上界")
        elif kind == "text":
            text = str(value)
            if not text.strip():
                if spec.get("required"):
                    errors.append(f"事件 {event_type} 参数 {name} 为空文本")
                continue  # 可选文本允许为空（如 note）
            if len(text) > spec.get("max", 200):
                errors.append(f"事件 {event_type} 参数 {name} 文本超长")
        elif kind == "bool":
            if not isinstance(value, bool):
                errors.append(f"事件 {event_type} 参数 {name} 不是布尔值")
        elif kind == "refs":
            if not isinstance(value, list):
                errors.append(f"事件 {event_type} 参数 {name} 不是引用列表")
                continue
            if len(value) > spec.get("max_items", 3):
                errors.append(f"事件 {event_type} 参数 {name} 引用过多")
            if any(not isinstance(ref, str) or not ref.strip() for ref in value):
                errors.append(f"事件 {event_type} 参数 {name} 含有无效引用")
    return errors


def named_active_npcs_in_text(world: World, text: str) -> list[str]:
    """返回世界层自由文本中提到的活跃角色名。

    这不是语义解析器：它只守一个账本边界。世界层文本不能替已存在的
    角色声明行为；需要提到角色时，调用方必须改走有身份的事件通道。
    """
    matches: list[str] = []
    for npc in world.npcs.values():
        name = npc.name
        if not name:
            continue
        # 中文叙述常把「神秘修钟人」简称成「修钟人」。后缀不是新身份，
        # 只是同一稳定名字的自然省略；长度至少 2，避开单字误伤。
        forms = {name}
        forms.update(name[i:] for i in range(len(name) - 1)
                     if len(name[i:]) >= 2)
        if any(form in text for form in forms):
            matches.append(name)
    return matches


def validate_refs(world: World, event_type: str, params: dict,
                  cause: str = "") -> list[str]:
    """引用完整性：事件引用的 NPC / 场景必须是世界内真实存在的。"""
    errors: list[str] = []
    for key in ("npc", "target", "actor", "body"):
        if key in params and params[key] not in world.npcs \
                and not (key == "target" and params[key] == "player"):
            errors.append(f"事件引用不存在的 NPC：{params[key]}")
    # from/to 只在「移动/生长」类事件里是场景 id；weather_shift 的 to 是天气名
    if event_type in ("npc_moved", "scene_extended"):
        for key in ("from", "to"):
            if key in params and params[key] \
                    and params[key] not in world.scenes:
                errors.append(f"事件引用不存在的场景：{params[key]}")
    for key in ("location", "origin"):
        if key in params and params[key] \
                and params[key] not in world.scenes:
            errors.append(f"事件引用不存在的场景：{params[key]}")
    if event_type == "scene_state_changed":
        scene_id = str(params.get("scene", ""))
        if scene_id not in world.scenes:
            errors.append(f"事件引用不存在的场景：{scene_id}")
    refs = params.get("refs", [])
    if isinstance(refs, list):
        for ref in refs:
            if resolve_target(world, str(ref)) is None:
                errors.append(f"事件引用不存在的实体：{ref}")
    targets = params.get("targets", [])
    requires = params.get("requires", [])
    if event_type == "npc_acted" and isinstance(targets, list):
        for target in targets + (requires if isinstance(requires, list) else []):
            if resolve_target(world, str(target)) is None:
                errors.append(f"行动引用不存在的实体：{target}")
    # 世界层自由文本不能绕过 NPC 的行动、移动、对话账本。角色一旦被提到，
    # 必须走现有的 npc_* 事件路径。
    if event_type in ("world_event", "daily_life"):
        text = (f"{params.get('title', '')}{params.get('detail', '')}"
                f"{cause}")
        named = named_active_npcs_in_text(world, text)
        if event_type == "world_event" and isinstance(refs, list):
            referenced = {
                world.npcs[resolved[4:]].name
                for ref in refs
                if (resolved := resolve_target(world, str(ref)))
                and resolved.startswith("npc:")
                and resolved[4:] in world.npcs
            }
            named = [name for name in named if name not in referenced]
        if named:
            errors.append(f"{EVENT_TYPES[event_type][0]}不得直接叙述已有角色：" +
                          "、".join(named))
    # `npc_acted` 只记录一个角色自己的行动。已有角色之间的直接开口必须
    # 走 npc_interaction，才能让双方记忆、关系和现场可见性一起入账。
    # 这不是剧情分类，只是阻止自由文本绕过既有互动事件。
    if event_type == "npc_acted":
        actor = world.npcs.get(str(params.get("npc", "")))
        action = str(params.get("action", ""))
        named = [name for name in named_active_npcs_in_text(world, action)
                 if actor is None or name != actor.name]
        speech_markers = ("说", "问", "答", "告诉", "回应", "交谈", "聊天")
        if named and any(marker in action for marker in speech_markers):
            errors.append("NPC 主动行动不得替代与已有角色的对话；请走 npc_interaction")
        # 目的地不是从自由叙事猜出来的。动作提到另一处已知场景且包含移动
        # 语义时，必须显式给 location；否则「走到那里」会只发生在文字里。
        other_scenes = [scene for scene in world.scenes.values()
                        if (actor is None or scene.id != actor.state.location)
                        and scene.name and scene.name in action]
        # 只拦已经在移动/已经抵达的表述；「以后去」「想去」「准备去」仍是
        # 原地发生的念头或准备，不应被引擎替角色强行兑现。
        motion_markers = ("走向", "走到", "来到", "前往", "赶往", "走去")
        if other_scenes and any(marker in action for marker in motion_markers):
            destination = str(params.get("location", ""))
            if not destination:
                errors.append("跨场景行动必须给 location 目的地，不能只写在 action 文本里")
            elif destination not in {scene.id for scene in other_scenes}:
                errors.append("跨场景行动的 location 必须与 action 中的目的地一致")
        destination = str(params.get("location", ""))
        try:
            duration = float(params.get("days", 0.0) or 0.0)
        except (TypeError, ValueError):
            duration = None
        if (actor is not None and destination
                and destination != actor.state.location
                and "days" in params and duration is not None
                and duration <= 0.0):
            errors.append("跨场景行动的 days 必须大于 0")
    if event_type == "npc_intent":
        intent = str(params.get("intent", "")).strip()
        previous = str(params.get("previous", "")).strip()
        if not intent and not previous:
            errors.append("短期打算变更必须说明形成的新打算或放下的旧打算")
        targets = params.get("targets", [])
        if not intent and targets:
            errors.append("放下短期打算时不能保留目标引用")
        if isinstance(targets, list):
            for target in targets:
                if resolve_target(world, str(target)) is None:
                    errors.append(f"短期打算引用不存在的实体：{target}")
        try:
            earliest = float(params.get("earliest_clock", 0.0) or 0.0)
            latest = float(params.get("latest_clock", 0.0) or 0.0)
            if latest and earliest and latest < earliest:
                errors.append("短期打算的 latest_clock 早于 earliest_clock")
        except (TypeError, ValueError):
            errors.append("短期打算的时间窗口不是数值")
    if event_type == "npc_acted":
        try:
            earliest = float(params.get("earliest_clock", 0.0) or 0.0)
            latest = float(params.get("latest_clock", 0.0) or 0.0)
            if latest and earliest and latest < earliest:
                errors.append("行动的 latest_clock 早于 earliest_clock")
        except (TypeError, ValueError):
            errors.append("行动的时间窗口不是数值")
    if event_type == "action_done":
        actor = world.npcs.get(str(params.get("npc", "")))
        outcome = str(params.get("outcome", ""))
        named = [name for name in named_active_npcs_in_text(world, outcome)
                 if actor is None or name != actor.name]
        contact_markers = ("说", "问", "答", "告诉", "回应", "交谈", "聊天",
                           "相见", "遇见", "见到", "看见", "对视")
        if named and any(marker in outcome for marker in contact_markers):
            errors.append("动作结局不得替代与已有角色的相遇或对话；请走 npc_interaction")
    return errors


def build_summary(world: World, event_type: str, params: dict) -> str:
    """机械生成事件摘要（确定性：同类型同参数 → 同摘要）。"""
    p = params
    if event_type == "npc_moved":
        npc = world.npcs.get(p["npc"])
        frm = world.scenes.get(p["from"])
        to = world.scenes.get(p["to"])
        return (f"{npc.name if npc else p['npc']} 从"
                f"「{frm.name if frm else p['from']}」来到"
                f"「{to.name if to else p['to']}」")
    if event_type == "npc_interaction":
        a = world.npcs.get(p["npc"])
        b = world.npcs.get(p["target"])
        target = "你" if p["target"] == "player" else \
            (b.name if b else p["target"])
        return f"{a.name if a else p['npc']} 对 {target}：{p['line']}"
    if event_type == "npc_reacted":
        npc = world.npcs.get(p["npc"])
        return f"{npc.name if npc else p['npc']}：{p['reaction']}"
    if event_type == "world_event":
        if p["title"] == p["detail"]:
            return p["title"]
        return f"{p['title']}：{p['detail']}"
    if event_type == "item_arrive":
        scene = world.scenes.get(p["location"])
        note = f"（{p['note']}）" if p.get("note") else ""
        return (f"「{p['item']}」抵达"
                f"「{scene.name if scene else p['location']}」{note}")
    if event_type == "item_added":
        scene = world.scenes.get(p["location"])
        note = f"（{p['note']}）" if p.get("note") else ""
        return (f"「{p['name']}」出现在"
                f"「{scene.name if scene else p['location']}」{note}")
    if event_type == "item_removed":
        scene = world.scenes.get(p["location"])
        note = f"（{p['note']}）" if p.get("note") else ""
        return (f"「{p.get('name') or p['item']}」从"
                f"「{scene.name if scene else p['location']}」消失{note}")
    if event_type == "item_transfer":
        scene = world.scenes.get(p["location"])
        where = f"「{scene.name if scene else p['location']}」"
        holder = p.get("holder", "")
        if not holder or holder == "无":
            return f"「{p['item']}」被放回{where}"
        return f"「{p['item']}」被 {holder} 拿走了（{where}）"
    if event_type == "fact_changed":
        if p.get("old"):
            return f"世界设定变了：「{p['old']}」→「{p['new']}」"
        return f"世界设定新增：「{p['new']}」"
    if event_type == "item_changed":
        scene = world.scenes.get(p["location"])
        note = f"（{p['note']}）" if p.get("note") else ""
        return (f"「{p.get('name') or p['item']}」变了："
                f"「{scene.name if scene else p['location']}」{note}")
    if event_type == "weather_shift":
        return f"天气变为「{p['to']}」"
    if event_type == "scene_state_changed":
        scene = world.scenes.get(p["scene"])
        subject = scene.name if scene else p["scene"]
        text = str(p.get("text", "")).strip()
        if p.get("op") == "remove":
            return f"「{subject}」的局部状态结束：{p['fact']}"
        return f"「{subject}」的局部状态：{text or p['fact']}"
    if event_type == "daily_life":
        scene = world.scenes.get(p["location"])
        return (f"{p['detail']}（{scene.name if scene else p['location']}）")
    if event_type == "scene_generated":
        scene = world.scenes.get(p["scene"])
        return f"贴片「{scene.name if scene else p['scene']}」首次生成"
    if event_type == "npc_acted":
        npc = world.npcs.get(p["npc"])
        return f"{npc.name if npc else p['npc']} 主动：{p['action']}"
    if event_type == "npc_intent":
        npc = world.npcs.get(p["npc"])
        name = npc.name if npc else p["npc"]
        intent = str(p.get("intent", "")).strip()
        previous = str(p.get("previous", "")).strip()
        if intent and previous:
            return f"{name} 改了打算：「{previous}」→「{intent}」"
        if intent:
            return f"{name} 有了打算：「{intent}」"
        return f"{name} 放下了打算：「{previous}」"
    if event_type == "npc_state_changed":
        npc = world.npcs.get(p["npc"])
        name = npc.name if npc else p["npc"]
        if p["can_act"]:
            return f"{name} 恢复行动：{p['condition']}"
        return f"{name} 不再能行动：{p['condition']}"
    if event_type == "note_left":
        npc = world.npcs.get(p["npc"])
        return f"{npc.name if npc else p['npc']} 留下一张纸条：「{p['content']}」"
    if event_type == "goal_completed":
        npc = world.npcs.get(p["npc"])
        return f"{npc.name if npc else p['npc']} 达成目标：「{p['goal']}」"
    if event_type == "goal_emerged":
        npc = world.npcs.get(p["npc"])
        return f"{npc.name if npc else p['npc']} 长出了新目标：「{p['goal']}」"
    if event_type == "npc_initiated":
        npc = world.npcs.get(p["npc"])
        return f"{npc.name if npc else p['npc']} 主动开口：「{p['line']}」"
    if event_type == "action_done":
        npc = world.npcs.get(p["npc"])
        return (f"{npc.name if npc else p['npc']} 完成了「{p['action']}」："
                f"{p['outcome']}")
    if event_type == "action_aborted":
        npc = world.npcs.get(p["npc"])
        return (f"{npc.name if npc else p['npc']} 中止了动作"
                f"「{p['action']}」")
    if event_type == "collision":
        a = world.npcs.get(p["a"])
        b = world.npcs.get(p["b"])
        return (f"{(a.name if a else p['a'])} 与 "
                f"{(b.name if b else p['b'])} 的目标撞在「{p['thing']}」上")
    if event_type == "scene_extended":
        scene = world.scenes.get(p["scene"])
        frm = world.scenes.get(p["from"])
        return (f"世界生长：新贴片「{scene.name if scene else p['scene']}」"
                f"从「{frm.name if frm else p['from']}」长出来")
    if event_type == "player_acted":
        npc = world.npcs.get(p["npc"])
        veredict = "接受" if p.get("accepted") else "拒绝"
        return (f"你 对 {npc.name if npc else p['npc']}：{p['action']}"
                f"（被{veredict}）")
    if event_type == "player_said":
        npc = world.npcs.get(p["npc"])
        return (f"你 对 {npc.name if npc else p['npc']} 说："
                f"「{p['content']}」")
    if event_type == "action_refused":
        npc = world.npcs.get(p["npc"])
        reason = f"：{p['reason']}" if p.get("reason") else ""
        return (f"{npc.name if npc else p['npc']} 拒绝了你的动作"
                f"「{p['action']}」{reason}")
    return f"{event_type}"


def event_identity_note(world: World, event_type: str, payload: dict) -> str:
    """为引擎回放补充借身身份；普通事件不增加显示噪音。"""
    if event_type not in ("npc_acted", "action_done", "action_aborted"):
        return ""
    params = payload.get("event_params", payload) if isinstance(payload, dict) \
        else {}
    body_id = str(params.get("body") or params.get("npc") or "")
    actor_id = str(params.get("actor") or "")
    if not body_id or not actor_id or actor_id == body_id:
        return ""
    body = world.npcs.get(body_id)
    actor = world.npcs.get(actor_id)
    if body is None or actor is None:
        return ""
    return f"（行动者：{actor.name}；身体：{body.name}）"


def emit(world: World, event_type: str, params: dict,
         cause: str, duration: float = 0.0) -> list[str]:
    """事件调用：包络线 → 引用完整性 → 写入事件日志。返回错误列表。

    duration 是这件事的时间成本（天）——时间由运动决定：
    说话一瞬、赶路半天、睡一夜一天，世界钟随事件走。
    """
    if not cause or not str(cause).strip():
        raise ValueError(f"事件 {event_type} 必须有原因（有因才生）")
    # 跨场景行动的 location 是承诺的目的地，不是当前现场。出发地只能
    # 从账本状态机械取得，不能让模型声称自己已经在目的地。
    if event_type == "npc_acted":
        params = dict(params)
        body = world.npcs.get(str(params.get("npc", "")))
        if body is not None:
            # 身份由当前账本盖章，不能接受模型自己声称的 actor/body。
            params["body"] = body.id
            params["actor"] = world.actor_for_body(body).id
            params["origin"] = body.state.location
    elif event_type in ("action_done", "action_aborted"):
        params = dict(params)
        body = world.npcs.get(str(params.get("body") or params.get("npc", "")))
        if body is not None:
            params["body"] = body.id
            # 完成/中止动作可携带动作开始时固化的 actor；旧调用缺失时才回退当前映射。
            params.setdefault("actor", world.actor_for_body(body).id)
    errors = (validate_event(event_type, params)
              + validate_refs(world, event_type, params, cause))
    if errors:
        return errors
    world.log(event_type, build_summary(world, event_type, params), cause,
              {"event_params": params}, duration=duration)
    return []


def _close_item_fold(world: World, item: dict) -> list[str]:
    """折叠收尾：连续渐变的中间微变只覆写快照，收尾时一条总结入账。

    保留因果根（首条事件）与末态（note）；中间无后果的微变折叠——
    世界不是监控日志：状态永远最新，账本只有跃变与收尾总结。
    """
    fold = item.get("fold")
    if not isinstance(fold, dict) or int(fold.get("count", 1)) <= 1:
        return []
    n = int(fold.get("count", 0))
    last = str(fold.get("last", ""))[:60]
    item.pop("fold", None)
    item.pop("fold_last_turn", None)
    # 正常世界里物品 id 全局唯一；对象身份优先仍能让旧存档中的重复 id
    # 找到自己的真实场景，不会把别处同 id 物品的渐变串过来。
    scene = next((s for s in world.scenes.values()
                  if any(i is item for i in s.items)), None)
    if scene is None:
        matches = [s for s in world.scenes.values()
                   if any(i.get("id") == item.get("id") for i in s.items)]
        scene = matches[0] if len(matches) == 1 else None
    if scene is None:
        return []
    return emit(world, "daily_life",
                {"detail": f"「{item.get('name', '某物')}」又经历了 {n - 1} 次渐变，"
                           f"最后停在：{last}",
                 "location": scene.id, "item": str(item.get("id", ""))},
                cause="世界演化")


def _canonical_holder(world: World, holder: str) -> tuple[str, str]:
    """把持有者归一化；返回（规范引用，错误文本）。"""
    holder = str(holder or "").strip()
    if not holder or holder in ("无", "-"):
        return "", ""
    if holder == "player":
        return "player", ""
    resolved = resolve_target(world, holder)
    if resolved is None or not resolved.startswith("npc:"):
        return "", f"持有者无效：{holder}（只能是玩家或 NPC）"
    return resolved, ""


def transfer_item(world: World, item: dict, holder: str,
                  cause: str) -> list[str]:
    """物品转手：持有关系的一等变更（有因才转，引用验真）。

    holder：'player' / npc:引用 / NPC 裸名（解析到 NPC）/ 空或「无」= 放下。
    被持有的物品从场景物品表隐去（跟着持有者走）。
    """
    if not isinstance(item, dict) or not item.get("id"):
        return ["转手对象不是物品"]
    canonical, error = _canonical_holder(world, holder)
    if error:
        return [error]
    item["held_by"] = canonical
    item["last_turn"] = world.turn
    item["cause"] = cause
    summaries = _close_item_fold(world, item)  # 转手是跃变：先收尾渐变折叠
    scene = next((s for s in world.scenes.values()
                  if any(i is item for i in s.items)), None)
    if scene is None:
        matches = [s for s in world.scenes.values()
                   if any(i.get("id") == item.get("id") for i in s.items)]
        scene = matches[0] if len(matches) == 1 else None
    errors = emit(world, "item_transfer",
                {"item": str(item.get("name", item.get("id", ""))),
                 "holder": canonical or "无",
                 "location": scene.id if scene else
                 world.player.get("location", "")}, cause)
    if not errors:
        item["cause_turn"] = world.events[-1].turn  # 指向变更事件本身
    return summaries + errors

def apply_item_patch(world: World, patch: dict, cause: str) -> list[str]:
    """物品补丁：三律机械校验 → 覆写场景物品表 → 跃变写入事件日志。

    补丁：{"op": "add|remove|change", "item": id, "location": 场景id,
          "name": 名称(可选), "note": 备注(可选),
          "held_by": "player|npc:id|空（可选）"}
    三律：
    - 有因才存在：cause 必填；add 名称必填、id 不重复。
    - 有因才消失：remove/change 必须引用表中真实存在的物品。
    - 账本不矛盾：change 必须有内容（不许空覆写制造摇摆）。
    覆写总是发生（表永远最新）；日志只记跃变。
    """
    op = str(patch.get("op", ""))
    if op not in ("add", "remove", "change"):
        return [f"未知物品操作：{op}"]
    if not str(cause).strip():
        return ["物品补丁缺少原因（有因才存在/消失）"]
    sid = str(patch.get("location", ""))
    scene = world.scenes.get(sid)
    if scene is None:
        return [f"物品补丁引用不存在的场景：{sid}"]
    iid = str(patch.get("item", "")).strip()
    if not iid:
        return ["物品补丁缺少 item id"]

    if op == "add":
        name = str(patch.get("name", "")).strip()
        if not name:
            return ["物品出现缺少名称"]
        owner = next((s for s in world.scenes.values()
                      if any(i.get("id") == iid for i in s.items)), None)
        if owner is not None:
            return [f"物品 {iid} 已存在于「{owner.name}」（物品 id 必须全局唯一）"]
        held = patch.get("held_by") if "held_by" in patch else None
        canonical, error = _canonical_holder(world, held) if held is not None \
            else ("", "")
        if error:
            return [error]
        item = {"id": iid, "name": name[:20],
                "note": str(patch.get("note", ""))[:80], "cause": cause,
                "last_turn": world.turn}
        scene.items.append(item)
        errors = emit(world, "item_added",
                    {"item": iid, "name": item["name"], "location": sid,
                     "note": item["note"]}, cause)
        if not errors:
            item["cause_turn"] = world.events[-1].turn  # 指向诞生事件本身
        # 新物品在场景里诞生时，不是一次「被放回」的转手。
        # 只有它出生时就由某人持有，才额外记一条持有关系变更。
        if errors or held is None or not canonical:
            return errors
        return transfer_item(world, item, canonical, cause)

    if not any(i["id"] == iid for i in scene.items):
        return [f"物品 {iid} 不在「{scene.name}」的物品表里（不凭空删改）"]
    item = next(i for i in scene.items if i["id"] == iid)

    # no-op 守卫：新内容与现状完全一致 → 没有状态变化就没有事件，
    # 也不刷新活跃时间（账本基本纪律，不是反重复标签）
    if op == "change":
        new_name_0 = str(patch.get("name", "")).strip()
        new_note_0 = str(patch.get("note", "")).strip()
        spawns_0 = [s for s in patch.get("spawn", []) if isinstance(s, dict)]
        held_0 = patch.get("held_by") if "held_by" in patch else None
        if ((new_name_0 or new_note_0 or spawns_0 or held_0 is not None)
                and (not new_name_0 or new_name_0 == item.get("name", ""))
                and (not new_note_0 or new_note_0 == item.get("note", ""))
                and not spawns_0 and held_0 is None):
            return []

    if op == "remove":
        summaries = _close_item_fold(world, item)  # 消逝是跃变：先收尾渐变折叠
        scene.items = [i for i in scene.items if i["id"] != iid]
        return summaries + emit(world, "item_removed",
                                {"item": iid, "name": item["name"],
                                 "location": sid,
                                 "note": str(patch.get("note", ""))[:80]},
                                cause)

    # change：跃变覆写 + 防摇摆 + 转化 spawn（烟 → 烟蒂 + 烟灰）
    new_name = str(patch.get("name", "")).strip()
    new_note = str(patch.get("note", "")).strip()
    held = patch.get("held_by") if "held_by" in patch else None
    spawns = [s for s in patch.get("spawn", []) if isinstance(s, dict)]
    if not new_name and not new_note and not spawns and held is None:
        return [f"物品 {iid} 的跃变没有内容（不许空覆写制造摇摆）"]
    if new_name:
        item["name"] = new_name[:20]
    if new_note:
        item["note"] = new_note[:80]
    item["cause"] = cause
    item["last_turn"] = world.turn
    summaries = _close_item_fold(world, item)  # 跃变先收尾渐变折叠
    scene.items.remove(item)
    scene.items.append(item)  # 活跃：移到表尾
    # 持有关系变更：转手是一等事件（有因才转、引用验真）
    if held is not None:
        errors = transfer_item(world, item, held, cause)
        return summaries + errors
    errors = emit(world, "item_changed",
                  {"item": iid, "name": item["name"], "location": sid,
                   "note": item["note"]}, cause)
    if not errors:
        item["cause_turn"] = world.events[-1].turn  # 指向变更事件本身
    # 转化产物：同一跃变的副产品入账（三律照旧：有因、id 不重复）
    for sp in spawns:
        s_iid = str(sp.get("id", "")).strip()
        s_name = str(sp.get("name", "")).strip()
        if not s_iid or not s_name:
            errors.append("转化产物缺少 id 或名称")
            continue
        owner = next((s for s in world.scenes.values()
                      if any(i.get("id") == s_iid for i in s.items)), None)
        if owner is not None:
            errors.append(f"产物 {s_iid} 已存在于「{owner.name}」")
            continue
        scene.items.append({"id": s_iid, "name": s_name[:20],
                            "note": str(sp.get("note", ""))[:80],
                            "cause": cause,
                            "last_turn": world.turn})
        errors.extend(emit(world, "item_added",
                           {"item": s_iid, "name": s_name[:20],
                            "location": sid,
                            "note": str(sp.get("note", ""))[:80]}, cause))
    return summaries + errors
