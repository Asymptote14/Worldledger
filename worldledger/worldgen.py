"""世界生成：一句话 → 法则档案 + 场景贴片 + 批量 NPC。

期望驱动的入口：创作者的描述是期望源，转译为可执行的世界数据。
"""
from __future__ import annotations

import json as _json
import math
import re

from . import physics
from .event import emit, named_active_npcs_in_text
from .history import materialize_stored_memories
from .llm import BaseLLM
from .store import Law, LawProfile, NPC, Scene, World


class WorldGenError(Exception):
    pass


# 这是输入契约的语义校验，不是世界题材枚举。创作者一旦写下明确的
# 周期承诺，世界 DNA 就必须留下一个可由世界钟执行的时刻锚；否则它
# 只能停留在 persona/law/fact 的散文里，运行时无从知道何时兑现。
_PERIODIC_COMMITMENT_PATTERNS = (
    "每天", "每日", "每夜", "每晚", "每隔", "每逢", "每到",
    "隔天", "隔日", "周期性", "周期", "反复", "循环", "轮流",
)
_AGENCY_COMMITMENT_PATTERNS = ("交换身体", "身体互换", "互换身体",
                               "附身", "借用身体", "操控身体", "占据身体")

_AGENCY_NAME_PATTERNS = (
    re.compile(r"(?P<actor>[^，。；、,\s]+)通过(?P<body>[^，。；、,\s]+)的身体"),
    re.compile(r"(?P<actor>[^，。；、,\s]+)借用(?P<body>[^，。；、,\s]+)的身体"),
    re.compile(r"(?P<actor>[^，。；、,\s]+)的灵魂进入(?P<body>[^，。；、,\s]+)的身体"),
)


def _has_periodic_commitment(text: str) -> bool:
    return any(marker in text for marker in _PERIODIC_COMMITMENT_PATTERNS)


def _moment_repeat_days(moment: dict) -> float:
    try:
        value = float(moment.get("repeat_days", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return value if value > 0 else 0.0


def _periodic_commitment_error(description: str, moments: list) -> str:
    """返回周期承诺缺口；普通世界和一次性时刻不受影响。"""
    if not _has_periodic_commitment(description):
        return ""
    if any(isinstance(moment, dict) and _moment_repeat_days(moment) > 0
           for moment in moments):
        return ""
    return ("描述包含明确的周期性时间承诺，但 moments 没有 repeat_days；"
            "请把该规律落成至少一个可执行时刻（due_day + repeat_days），"
            "不要只写进 laws/facts。")


def _agency_commitment_error(description: str, moments: list) -> str:
    """明确的行动归属变化必须有通用投影，避免只剩设定文字。"""
    if not any(marker in description for marker in _AGENCY_COMMITMENT_PATTERNS):
        return ""
    patches = [patch for moment in moments
               if isinstance(moment, dict)
               for patch in (moment.get("agency_patches") or [])
               if isinstance(patch, dict)]
    if not patches:
        return ("描述明确包含行动归属变化，但 moments 没有 agency_patches；"
                "请为相关时刻声明 body、actor、duration_days 和 why。")
    if "交换" in description or "互换" in description:
        bodies = {str(patch.get("body", "")).removeprefix("npc:")
                  for patch in patches if patch.get("body")}
        actors = {str(patch.get("actor", "")).removeprefix("npc:")
                  for patch in patches if patch.get("actor")}
        if len(bodies) < 2 or len(actors) < 2:
            return ("描述的是双方交换/互换，但 agency_patches 没有覆盖双方；"
                    "请在同一时刻原子声明两具身体与两位行动者。")
    restore_markers = ("归位", "恢复行动主体", "恢复意识", "回到各自身体",
                       "互换结束", "交换结束")
    restore_moments = [
        moment for moment in moments
        if isinstance(moment, dict)
        and any(marker in str(moment.get("what", ""))
                for marker in restore_markers)
    ]
    if restore_moments:
        borrowed_bodies = {
            str(patch.get("body", "")).removeprefix("npc:")
            for moment in moments
            for patch in (moment.get("agency_patches") or [])
            if isinstance(patch, dict)
            and str(patch.get("body", "")).removeprefix("npc:")
            != str(patch.get("actor", "")).removeprefix("npc:")
        }
        for moment in restore_moments:
            patches = [patch for patch in (moment.get("agency_patches") or [])
                       if isinstance(patch, dict)]
            if not patches:
                return ("归位/恢复 moment 必须声明 agency_patches；"
                        "不能只靠到期清理隐式恢复行动主体。")
            restored_bodies = {
                str(patch.get("body", "")).removeprefix("npc:")
                for patch in patches
                if str(patch.get("body", "")).removeprefix("npc:")
                == str(patch.get("actor", "")).removeprefix("npc:")
            }
            missing = borrowed_bodies - restored_bodies
            if missing:
                return ("归位/恢复 moment 没有覆盖所有被借用的身体："
                        + "、".join(sorted(missing)))
    return ""


_WORLDGEN_SYSTEM = """TASK:WORLDGEN
你是世界引擎的转译器。把用户对世界的文字描述转译为严格 JSON，
不要输出 JSON 以外的任何内容。

JSON 契约：
{
  "atmosphere": "氛围基调，2-10 字，如「雨·永续」",
  "laws": [ {"id": "law-1", "trigger": "触发条件", "effect": "触发后果",
             "intensity": 0.0 到 1.0 之间的数} ],
  "scenes": [ {"id": "s-1", "name": "场景名", "description": "场景贴片描述",
               "npcs": ["n-1"], "exits": ["s-2"],
               "items": [{"id": "i-1", "name": "物品名", "note": "备注"}]} ],
   "npcs": [ {"id": "n-1", "name": "角色名", "persona": "人设",
              "traits": {"性格词": true},
              "goals": [{"id": "g-1", "text": "目标", "progress": 0.0}],
              "memories": [{"content": "可选初始记忆（第一人称）",
                "projection": {"age_days": 数字或 null,
                  "duration_days": 数字或 null,
                  "embodied_as": "self 或已有 NPC id",
                  "accessible": true或false, "access_cause": "受限原因或空",
                  "scene": {"ref": "已有场景id或空", "name": "地点名",
                            "then": "当时状态"} 或 null,
                  "items": [{"ref": "已有物品id或空", "name": "物品名",
                    "then": "当时状态", "exists_now": true|false|null,
                    "current_location": "场景id或空", "held_by": "self/player/npc id/空",
                    "current_note": "当前状态或空"}],
                  "current_states": [{"text": "仍持续到现在的人物状态",
                                      "review_days": 数字或 null}]}}],
              "links": {"n-2": 0.2} } ],
  "facts": ["这个世界稳定的归属/规则条目，0-5 条，自由文本"],
  "moments": [{"due_day": 数字（第几夜/第几天）, "what": "到那时必然发生的事",
    "location": "发生地的场景 id（可选；不写 = 全局事件）",
    "refs": ["npc:id / item:id / scene:id"],
    "repeat_days": 数字（可选；每隔多少天重复一次）, 
    "agency_patches": [{"body": "npc:id", "actor": "npc:id",
      "duration_days": 数字, "why": "为何在这个时刻成立"}]}],
  "heartbeat": 0.0417
}

要求：
- laws 不超过 3 条；场景 2-4 个并互相连通；NPC 2-5 名。
- 每个场景的 npcs 列出在场的 NPC id。
- due_day 从 1 开始；第 1 天对应 due_day=1，不要输出 0 或负数。
- moments 的 agency_patches 必须复制 npcs 数组中真实角色的 id。若 why 明确写出
  「某人通过/借用某人的身体」，两个名字必须分别对应 actor/body，不能用同场景的
  其他 NPC 代替，也不能只凭数组顺序猜 id。
- 铁律：每个 NPC id 只能出现在一个场景的 npcs 数组里——
  角色一次只能在一个地方。只列「此刻就在这个场景里的人」，
  不要把认识的人都列上；别的场景列别人。
- 场景的 items 是该场景里本来就有的东西（每个场景 0-3 件，
  出生场景可以多一点）：桌子、招牌、晾着的衣服——它们是场景的
  一等状态，会被持续覆写更新，不是一次性的道具。
- 每个物品 id 在整个世界中必须唯一，不能在不同场景重复使用同一个 id。
- memories 是可选的角色出生记忆（0-2 条，第一人称）：只物化描述或
  persona 里明确写出的个人过往。没有明确过往就留空数组，不要编造履历。
- 每条 memory 必须同时给 projection。记忆里的地点、具体物品和仍持续到
  现在的人物后果都要分别写入；不知道物品今天是否存在就写 null，过去的
  地点状态只写 then，不得冒充今天。没有这些内容就用空值/空数组。
- 默认 embodied_as=self、accessible=true。只有描述明确存在附身、互换、
  操控、断片或失忆时才改变；使用别人身体的经历写给行动者，身体主人由
  引擎得到时间断档，不得复制行动内容。
- links 是可选的初始 NPC 关系表（NPC id -> -1.0 到 1.0）：只在描述
  明确说两人相识、亲近或敌对时写。仅仅同处一个场景不等于认识，省略即无关系。
- 法则要能在对话中触发：trigger 是「有人做什么」，effect 是「发生什么」。
- facts 是这个世界「稳定不变的事实」（0-5 条）：从描述里归纳出的
  归属与规则——哪些东西跟着谁走（衣服跟着身体）、哪些东西是唯一证据、
  哪些规律在所有裁决中成立。不要编，只归纳描述里隐含的。
  没有明显的归属规则就留空数组。
- moments 是「既定的时刻」（0-2 个）：从描述里提取带时刻的必然事件
  （第七夜彗星落下、第三天的黄昏门会开）。到点引擎会强制执行——
  时刻是世界的承诺，不是氛围。持续性规律可给 repeat_days，避免把
  「每隔几天发生」退化成只在背景里提一句。若规律会改变实体之间的
  行动归属，可在同一时刻声明 agency_patches；它是通用状态投影，不是
  某个题材的专用字段。location 写清发生地（可选）：
  彗星落在「陨石坑」、钟在「钟楼」响起；世界级事件不写 location。
  what 中明确提到的人物、物品或地点必须同时写入 refs；refs 是事实引用，
  不是额外叙事。没有明确时刻就留空数组。
- 描述中出现「每天/隔天/每夜/周期性/再次发生」等持续性时间承诺时，
  不得只把它写进 laws 或 facts；必须在 moments 中给出 repeat_days，
  并把该承诺造成的实体状态变化放进对应的投影字段。
- heartbeat 是世界的粒度：一个心跳有多少天。快节奏世界给大数
  （如 0.5 = 半天），慢节奏世界给小数（如 0.01）。缺省 0.0417（约一小时）。
- 贴片式生成：只完整描述第一个场景（出生场景）；其余场景只给
  name + hint（一句话线索）+ exits + npcs，description 留空字符串。
"""


def validate_world(world: World) -> list[str]:
    """世界结构校验：场景/NPC 数量、引用关系、连通、物品结构。

    硬校验覆盖：不只法则——世界的骨架也必须完整。
    """
    errors: list[str] = []
    if not world.scenes:
        errors.append("世界没有场景")
    if not world.npcs:
        errors.append("世界没有 NPC")
    if len(world.scenes) > 12:
        errors.append(f"场景过多（{len(world.scenes)} > 12）")
    if len(world.npcs) > 12:
        errors.append(f"NPC 过多（{len(world.npcs)} > 12）")
    placed: set[str] = set()
    seen: dict[str, str] = {}
    seen_items: dict[str, str] = {}
    for sid, s in world.scenes.items():
        for nid in s.npcs:
            if nid not in world.npcs:
                errors.append(f"场景「{s.name}」引用不存在的 NPC：{nid}")
                continue
            if nid in seen:
                errors.append(
                    f"NPC「{world.npcs[nid].name}」同时属于"
                    f"「{world.scenes[seen[nid]].name}」和「{s.name}」")
            else:
                seen[nid] = sid
            placed.add(nid)
        for e in s.exits:
            if e not in world.scenes:
                errors.append(f"场景「{s.name}」的出口不存在：{e}")
        for item in s.items:
            if not isinstance(item, dict) or \
                    not str(item.get("id", "")).strip() or \
                    not str(item.get("name", "")).strip():
                errors.append(f"场景「{s.name}」存在无名物品")
                continue
            iid = str(item["id"])
            if iid in seen_items:
                errors.append(
                    f"物品 ID「{iid}」同时属于「{world.scenes[seen_items[iid]].name}」"
                    f"和「{s.name}」")
            else:
                seen_items[iid] = sid
    for nid, npc in world.npcs.items():
        if nid not in placed and npc.state.can_act and not npc.in_fog:
            errors.append(f"NPC「{npc.name}」不属于任何场景")
        for other_id, strength in npc.links.items():
            if other_id not in world.npcs:
                errors.append(f"NPC「{npc.name}」的关系对象不存在：{other_id}")
            elif other_id == nid:
                errors.append(f"NPC「{npc.name}」不能与自己建立关系")
            elif not -1.0 <= strength <= 1.0:
                errors.append(f"NPC「{npc.name}」的关系值超界：{other_id}")
    if world.player.get("location") not in world.scenes:
        errors.append("玩家出生场景不存在")
    for moment in world.moments:
        try:
            due_day = float(moment.get("due_day", 1))
        except (TypeError, ValueError):
            due_day = float("nan")
        if not math.isfinite(due_day) or due_day < 1:
            errors.append("既定时刻的 due_day 必须是从 1 开始的有限数字")
        location = str(moment.get("location", "")).strip()
        if location and location not in world.scenes:
            errors.append(f"既定时刻的发生地不存在：{location}")
        refs = moment.get("refs", [])
        if refs is None:
            refs = []
        if not isinstance(refs, list):
            errors.append("既定时刻的 refs 必须是数组")
            refs = []
        for ref in refs:
            value = str(ref).strip()
            kind, _, ident = value.partition(":")
            valid = ((kind == "npc" and ident in world.npcs)
                     or (kind == "scene" and ident in world.scenes)
                     or (kind == "item" and any(
                         item.get("id") == ident
                         for scene in world.scenes.values()
                         for item in scene.items)))
            if not valid:
                errors.append(f"既定时刻引用了不存在的实体：{value}")
        # 既定时刻不是自由叙事出口。`what` 点名已有角色时，必须用 refs
        # 把它们纳入同一个事件事实；否则运行时会被引用完整性拒绝，承诺
        # 也就永远无法兑现。物品和场景名可能是普通词，仍由生成提示约束；
        # 角色名在世界内稳定，适合做这条机械校验。
        named = named_active_npcs_in_text(world, str(moment.get("what", "")))
        npc_refs = {
            str(ref).strip().removeprefix("npc:")
            for ref in refs if str(ref).strip().startswith("npc:")
        }
        missing_named = [name for name in named
                         if not any(npc.id in npc_refs and npc.name == name
                                    for npc in world.npcs.values())]
        if missing_named:
            errors.append("既定时刻提到的角色未写入 refs：" +
                          "、".join(missing_named))
        patches = moment.get("agency_patches", [])
        if patches is None:
            patches = []
        if not isinstance(patches, list):
            errors.append("既定时刻的 agency_patches 必须是数组")
            continue
        seen_bodies: set[str] = set()
        for patch in patches:
            if not isinstance(patch, dict):
                errors.append("既定时刻的行动主体映射必须是对象")
                continue
            body = str(patch.get("body", "")).removeprefix("npc:")
            actor = str(patch.get("actor", "")).removeprefix("npc:")
            if body not in world.npcs or actor not in world.npcs:
                errors.append("既定时刻的行动主体映射引用了不存在的角色")
            if body in seen_bodies:
                errors.append(f"既定时刻重复映射身体：{body}")
            seen_bodies.add(body)
            if body not in world.npcs or actor not in world.npcs:
                continue
            why = str(patch.get("why", ""))
            for match_pattern in _AGENCY_NAME_PATTERNS:
                for match in match_pattern.finditer(why):
                    actor_label = match.group("actor").strip()
                    body_label = match.group("body").strip()
                    actor_ids = [nid for nid, npc in world.npcs.items()
                                 if npc.name == actor_label
                                 or npc.name.endswith(actor_label)]
                    body_ids = [nid for nid, npc in world.npcs.items()
                                if npc.name == body_label
                                or npc.name.endswith(body_label)]
                    if len(actor_ids) != 1 or actor_ids[0] != actor:
                        errors.append(
                            f"既定时刻行动者文字与引用不一致：{actor_label} -> {actor}")
                    if len(body_ids) != 1 or body_ids[0] != body:
                        errors.append(
                            f"既定时刻身体文字与引用不一致：{body_label} -> {body}")
    # 连通性：从出生场景出发应能走到所有场景（孤岛不可达 = 结构残缺）
    start = world.player.get("location", "")
    if start in world.scenes:
        reached: set[str] = {start}
        frontier = [start]
        while frontier:
            cur = frontier.pop()
            for e in world.scenes[cur].exits:
                if e in world.scenes and e not in reached:
                    reached.add(e)
                    frontier.append(e)
        for sid in set(world.scenes) - reached:
            if world.scenes[sid].memory_only:
                continue
            errors.append(f"场景「{world.scenes[sid].name}」是孤岛"
                          f"（从出生场景走不到）")
    return errors


def _build_world(name: str, description: str, data: dict) -> World:
    profile = LawProfile(
        expectation=description,
        atmosphere=str(data.get("atmosphere", "")),
        laws=[Law.from_dict(x) for x in data.get("laws", [])],
        version=0,
    )
    errors = physics.validate_profile(profile)
    if errors:
        raise WorldGenError("包络线拒绝：" + "；".join(errors))

    scenes = {s["id"]: Scene.from_dict(s) for s in data.get("scenes", [])}
    npcs = {n["id"]: NPC.from_dict(n) for n in data.get("npcs", [])}
    # ID 唯一性（字典会静默去重，必须在构造前查）
    for label, raw in (("场景", data.get("scenes", [])),
                       ("NPC", data.get("npcs", []))):
        ids = [str(x.get("id", "")) for x in raw]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise WorldGenError(f"{label} ID 重复：{dupes}")
    world = World(name=name, description=description, law_profile=profile,
                  scenes=scenes, npcs=npcs)
    world.player = {"location": next(iter(scenes)) if scenes else ""}

    world.log("world_created",
              f"世界「{name}」诞生：{profile.atmosphere}",
              "创建世界命令",
              {"laws": [law.id for law in profile.laws]})
    # NPC 初始状态快照：位置 = 生成时所在场景，演化进度对齐当前回合
    for npc in world.npcs.values():
        for scene in world.scenes.values():
            if npc.id in scene.npcs:
                npc.state.location = scene.id
                break
        npc.state.mark(world.turn, world.clock)
    # 出生对齐只保留生成描述明确声明的 memories / links；
    # 同场景不自动推断相识，陌生人也可以同处一室。
    # 初始记忆盖时间戳：生成时它们已是角色的过去。
    for npc in world.npcs.values():
        for m in npc.memories:
            if m.turn <= 0:
                m.turn = world.turn
        projection_errors = materialize_stored_memories(world, npc,
                                                        require_all=True)
        if projection_errors:
            raise WorldGenError("出生记忆投影失败：" + "；".join(projection_errors))
    # 初始物品补活跃时间戳（世界生成时它们就被「看见」了）
    for scene in world.scenes.values():
        for item in scene.items:
            item.setdefault("last_turn", world.turn)
    # 初始天气 = 氛围基调（强度平稳），情绪从平静开始
    world.weather = profile.atmosphere
    world.weather_intensity = 0.2
    world.weather_reason = "世界生成"
    # 雾中标记：description 为空的场景 = 未生成的贴片
    for scene in world.scenes.values():
        scene.generated = bool(scene.description.strip())
    # 世界档案：AI 归纳的稳定归属/规则条目（给裁决的锚，可为空）
    world.facts = [str(f).strip() for f in data.get("facts", [])
                   if str(f).strip()][:5]
    def _moment_repeat(raw: dict) -> float:
        try:
            value = float(raw.get("repeat_days", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
        return value if value > 0 else 0.0

    # 既定时刻：到点必发的世界承诺（时刻锚；location 可选 = 全局事件）
    world.moments = [
        {"due_day": int(m.get("due_day", 1)),
         "what": str(m.get("what", "")).strip()[:80],
         "done": False,
         **({"location": str(m.get("location", "")).strip()}
            if str(m.get("location", "")).strip() else {}),
         **({"repeat_days": _moment_repeat(m)}
            if _moment_repeat(m) > 0 else {}),
         **({"refs": [str(ref).strip() for ref in m.get("refs", [])[:8]
                       if str(ref).strip()]}
            if isinstance(m.get("refs"), list) else {}),
         **({"agency_patches": [dict(p) for p in
             m.get("agency_patches", [])[:8]
             if isinstance(p, dict)]}
            if isinstance(m.get("agency_patches"), list) else {})}
        for m in data.get("moments", [])
        if isinstance(m, dict) and str(m.get("what", "")).strip()
    ][:4]
    periodic_error = _periodic_commitment_error(description, world.moments)
    if periodic_error:
        raise WorldGenError(periodic_error)
    agency_error = _agency_commitment_error(description, world.moments)
    if agency_error:
        raise WorldGenError(agency_error)
    # 世界粒度：一个心跳多少天（快世界大、慢世界小）
    hb = float(data.get("heartbeat", 1.0 / 24.0))
    world.heartbeat = min(1.0, max(0.001, hb))
    # 世界骨架硬校验：引用/连通/结构不完整就拒绝生成
    problems = validate_world(world)
    if problems:
        raise WorldGenError("世界结构校验失败：" + "；".join(problems))
    return world


def generate_world(llm: BaseLLM, name: str, description: str) -> World:
    """DNA 波：一句话 → 法则 + 氛围 + 场景骨架 + 批量 NPC + 出生场景。

    结构硬校验失败 = 这一波生成有结构矛盾（重复归属等）——
    重试而不是替世界改错；重试仍失败才把错误交给调用者。
    重试时把校验错误带回给模型：它看得见自己错在哪，下次才改得对。
    """
    last: Exception | None = None
    prompt = description
    for attempt in range(5):
        try:
            data = llm.chat_json(_WORLDGEN_SYSTEM, prompt)
            world = _build_world(name, description, data)
        except WorldGenError as e:
            last = e
            prompt = (f"{description}\n（上次生成被拒绝：{e}。"
                      f"请避免这些问题，重新生成。）")
            continue
        # 出生场景若未被 DNA 波描述（真模型可能漏），立即补一波
        start = world.player.get("location", "")
        if start and not world.scenes[start].generated:
            ensure_scene(llm, world, start)
        return world
    raise last or WorldGenError("世界生成失败")


_SCENEGEN_SYSTEM = """TASK:SCENEGEN
你是世界引擎的贴片生成器。这个场景首次被玩家抵达，按世界 DNA 生成
它的完整描述。输入里有 npcs_here（当前在场 NPC 的名字与人设）——
描述要与在场人物协调，并参考 historical_anchors 与当前时间相隔多久；
过去的状态可以延续、改变或消失，但不要直接复制成今天。不要把具体名字
当作「此刻必然在场」写死，
人物状态由引擎实时维护。
输出严格 JSON：
{"description": "场景贴片描述（2-3 句）", "atmosphere": "场景氛围（可为空字符串）"}
不要输出 JSON 以外的任何内容。"""


def ensure_scene(llm: BaseLLM, world: World, scene_id: str) -> list[str]:
    """贴片波：场景首次抵达时生成完整描述。返回摘要，空列表 = 无需生成。"""
    scene = world.scenes.get(scene_id)
    if scene is None:
        return [f"场景不存在：{scene_id}"]
    if scene.generated:
        return []
    data = llm.chat_json(_SCENEGEN_SYSTEM, _json.dumps({
        "scene": {"id": scene.id, "name": scene.name, "hint": scene.hint},
        "world": {
            "expectation": world.description,
            "atmosphere": world.law_profile.atmosphere,
            "laws": [f"{l.trigger} → {l.effect}"
                     for l in world.law_profile.laws],
        },
        "neighbors": [world.scenes[e].name
                      for e in scene.exits if e in world.scenes],
        "world_clock": {"days": world.clock, "now": world.now()},
        "historical_anchors": [
            {**anchor,
             "age_days": (world.clock - float(anchor["occurred_clock"])
                          if anchor.get("occurred_clock") is not None else None)}
            for anchor in scene.history
        ],
        # 贴片波读库，不猜：当前在场 NPC（放置与移动都是已写回的固化事件）
        "npcs_here": [
            {"name": world.npcs[nid].name, "persona": world.npcs[nid].persona}
            for nid in scene.npcs if nid in world.npcs
        ],
    }, ensure_ascii=False))
    desc = str(data.get("description", "")).strip()
    if len(desc) < physics.MIN_TEXT:
        return ["贴片生成失败：描述为空"]
    scene.description = desc
    scene.atmosphere = str(data.get("atmosphere", ""))
    scene.generated = True
    emit(world, "scene_generated", {"scene": scene.id}, cause="贴片生成")
    return [f"贴片生成：「{scene.name}」"]


def extend_scene(world: World, from_id: str, name: str,
                 hint: str) -> list[str]:
    """世界生长：从已有场景长出一块新的雾中贴片（双向链接）。"""
    frm = world.scenes.get(from_id)
    if frm is None:
        return [f"场景不存在：{from_id}"]
    i = len(world.scenes) + 1
    while f"s-{i}" in world.scenes:
        i += 1
    sid = f"s-{i}"
    scene = Scene(id=sid, name=name.strip() or f"未知之地{i}",
                  description="", atmosphere="",
                  exits=[from_id], hint=hint, generated=False)
    world.scenes[sid] = scene
    frm.exits.append(sid)
    emit(world, "scene_extended", {"scene": sid, "from": from_id},
         cause="世界心跳")
    return [f"世界生长：「{scene.name}」（雾中）"]


def emerge_place(world: World, sid: str, name: str,
                 from_id: str) -> list[str]:
    """新地名涌现：NPC 要去的地方不存在 → 涌现成地名级场景（雾中）。

    不细化（没有描述、没有 NPC 名单），只有名字与来路链接——
    舞台细不细化不影响角色去那里做事；玩家将来去了才贴片生成。
    """
    frm = world.scenes.get(from_id)
    if frm is None:
        return [f"来路场景不存在：{from_id}"]
    if sid in world.scenes:
        return []  # 已存在：不是新地名
    scene = Scene(id=sid, name=name.strip() or f"未知之地{sid}",
                  description="", atmosphere="",
                  exits=[from_id], hint="传说中，未证实", generated=False)
    world.scenes[sid] = scene
    frm.exits.append(sid)
    emit(world, "scene_extended", {"scene": sid, "from": from_id},
         cause="NPC 去往新地方")
    return [f"新地名涌现：「{scene.name}」（雾中）"]
