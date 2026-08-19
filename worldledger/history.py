"""个人往事与世界账本的对齐。

记忆是角色对过去的叙述；场景、物品和仍持续的后果是世界事实。
模型只负责从叙述中提出投影，引擎负责保留不确定性、验证引用并落库。
"""
from __future__ import annotations

import json
from typing import Any

from .llm import BaseLLM
from .store import Memory, NPC, Scene, World, ensure_memory_ids


_ALIGN_SYSTEM = """TASK:MEMORYALIGN
你是世界账本的往事对齐器。输入给出一个人物的自由文本记忆和当前世界。
把记忆中明确提到的地点、物品，以及过去动作仍持续到现在的后果投影成
严格 JSON。不要补写记忆没有依据的履历，不要把过去状态冒充今天的状态。

输出：
{"memories": [{
  "memory_id": "输入 id",
  "age_days": 数字或 null,
  "duration_days": 数字或 null,
  "embodied_as": "self 或已有 NPC id",
  "accessible": true或false,
  "access_cause": "不可访问的原因；accessible=true 时可空",
  "scene": {"ref": "已有场景 id 或空", "name": "地点名",
            "then": "这段记忆发生时地点是什么样"} 或 null,
  "items": [{"ref": "已有物品 id 或空", "name": "物品名",
              "then": "当时状态", "exists_now": true|false|null,
              "current_location": "已有场景 id/本条 scene/空",
              "held_by": "self|player|npc id|空",
              "current_note": "明确可推出的当前状态或空"}],
  "current_states": [{"text": "仍对人物现在成立的事实",
                      "review_days": 数字或 null}]
}]}

规则：
- age_days 只有能从“三年前/昨天”等表述合理换算时才填，否则 null。
- embodied_as 是这段经历发生时行动者使用的身体；没有身体分离就写 self。
  accessible 表示人物现在能否直接想起。即使为 false，仍要完整提取世界投影。
- scene 记录的是当时。即使地点可能还在，也不要猜今天是否原样。
  - 每件被明确提到的具体物品都输出；丢失写 false，明确保留到现在写 true，
  无法判断写 null。不要把“雨、伤心、战争”当物品。
  - 身体、身体部位、透明度、伤口和伤疤是人物当前状态，不是可持有物品；
    只写进 current_states，不要放进 items。
- current_states 只写现在仍成立的后果。昨天被刀划伤通常仍有伤口；十年前
  受伤只有明确或高度必然留下的疤痕才写。已结束的动作不写当前状态。
- review_days 表示多久后应重新判断该状态；永久或无需复查写 null。
- 输出不得增加记忆里没有依据的人、地点、物品、伤势或关系。
不要输出 JSON 以外内容。"""


def memory_from_value(value: Any, turn: int, clock: float) -> Memory:
    """兼容字符串与结构化记忆；结构化版本可随卡携带已提议的投影。"""
    if isinstance(value, dict):
        raw = dict(value)
        raw.setdefault("turn", turn)
        memory = Memory.from_dict(raw)
        age = raw.get("age_days")
        if memory.occurred_clock is None and age is not None:
            try:
                memory.occurred_clock = clock - max(0.0, float(age))
            except (TypeError, ValueError):
                pass
        return memory
    return Memory(turn=turn, content=str(value))


def _scene_ref(world: World, raw: str, name: str) -> Scene | None:
    ref = str(raw or "").removeprefix("scene:").strip()
    if ref in world.scenes:
        return world.scenes[ref]
    wanted = str(name or "").strip()
    return next((scene for scene in world.scenes.values()
                 if wanted and scene.name == wanted), None)


def _new_scene_id(world: World) -> str:
    index = 1
    while f"s-memory-{index}" in world.scenes:
        index += 1
    return f"s-memory-{index}"


def _find_active_item(world: World, raw: str, name: str = ""):
    ref = str(raw or "").removeprefix("item:").strip()
    for scene in world.scenes.values():
        for item in scene.items:
            if (ref and item.get("id") == ref) or (
                    not ref and name and item.get("name") == name):
                return scene, item
    return None, None


def _looks_like_body_state(spec: dict) -> bool:
    """识别无明确物品引用的身体状态，避免把人物状态物化成物品。"""
    name = str(spec.get("name", "")).strip()
    context = " ".join(str(spec.get(key, ""))
                        for key in ("name", "then", "current_note"))
    if name in {"身体", "我的身体", "自己的身体", "身体状态"}:
        return True
    body_parts = ("指尖", "手臂", "胳膊", "腿", "眼睛", "耳朵",
                  "头发", "皮肤", "脸", "伤口", "伤疤")
    if name in body_parts:
        return True
    return ("身体" in name and
            any(word in context for word in
                ("透明", "受伤", "伤口", "伤疤", "疼", "消耗")))


def _new_item_id(world: World) -> str:
    used = set(world.past_items)
    used.update(str(item.get("id", "")) for scene in world.scenes.values()
                for item in scene.items)
    index = 1
    while f"i-memory-{index}" in used:
        index += 1
    return f"i-memory-{index}"


def _holder(world: World, npc: NPC, raw: str) -> str:
    value = str(raw or "").strip()
    if value == "self":
        return f"npc:{npc.id}"
    if value == "player":
        return "player"
    value = value.removeprefix("npc:")
    return f"npc:{value}" if value in world.npcs else ""


def _projection_scene(world: World, npc: NPC, memory: Memory,
                      spec: dict) -> Scene | None:
    name = str(spec.get("name", "")).strip()[:40]
    then = str(spec.get("then", "")).strip()[:240]
    if not name and not str(spec.get("ref", "")).strip():
        return None
    scene = _scene_ref(world, spec.get("ref", ""), name)
    if scene is None:
        scene = Scene(id=_new_scene_id(world), name=name or "记忆中的地点",
                      description="", atmosphere="", generated=False,
                      memory_only=True,
                      hint="由人物往事留下的地点；今天的样子尚未被观察")
        world.scenes[scene.id] = scene
    anchor = {"memory_id": memory.id, "npc": npc.id, "then": then,
              "occurred_clock": memory.occurred_clock}
    if not any(h.get("memory_id") == memory.id and h.get("then") == then
               for h in scene.history):
        scene.history.append(anchor)
    return scene


def _projection_item(world: World, npc: NPC, memory: Memory, spec: dict,
                     memory_scene: Scene | None) -> None:
    name = str(spec.get("name", "")).strip()[:40]
    ref = str(spec.get("ref", "")).removeprefix("item:").strip()
    active_scene, active = _find_active_item(world, ref, name)
    # 空 ref 的身体状态不能凭名称变成可持有物；有明确 ref 的世界物品
    # 仍允许使用任意名称，只要它确实已经存在于账本中。
    if active is None and not ref and _looks_like_body_state(spec):
        return
    remembered = next((item for item in world.past_items.values()
                       if name and item.get("name") == name), None)
    item_id = str(active.get("id")) if active is not None else (
        ref if ref and (ref in world.past_items) else
        str(remembered.get("id")) if remembered is not None else
        _new_item_id(world))
    if not name and active is None:
        return
    record = world.past_items.setdefault(item_id, {
        "id": item_id, "name": name or str(active.get("name", "")),
        "history": [], "current_assertions": [], "current_assertion": None,
    })
    record.setdefault("current_assertions", [])
    entry = {"memory_id": memory.id, "npc": npc.id,
             "occurred_clock": memory.occurred_clock,
             "then": str(spec.get("then", "")).strip()[:240]}
    if not any(h.get("memory_id") == memory.id for h in record["history"]):
        record["history"].append(entry)
    exists = spec.get("exists_now")
    if exists not in (True, False, None):
        exists = None
    assertion = {
        "exists": exists, "source_memory": memory.id,
        "note": str(spec.get("current_note", "")).strip()[:160],
    }
    if not any(a.get("source_memory") == memory.id
               for a in record["current_assertions"]):
        record["current_assertions"].append(assertion)
    known = {a.get("exists") for a in record["current_assertions"]
             if a.get("exists") in (True, False)}
    if len(known) > 1:
        record["current_assertion"] = {
            "exists": None, "conflict": True,
            "note": "不同记忆对这件物品今天是否仍存在说法冲突",
        }
        if active is not None and str(active.get("cause", "")).startswith(
                "往事投影"):
            from .event import apply_item_patch
            apply_item_patch(world, {"op": "remove", "item": item_id,
                                     "location": active_scene.id,
                                     "note": "往事对当前存在状态互相冲突"},
                             cause=f"往事投影冲突：{memory.id}")
        return
    record["current_assertion"] = assertion
    if exists is not True or active is not None:
        return
    location_raw = str(spec.get("current_location", "")).strip()
    location = (_scene_ref(world, location_raw, "") or
                (memory_scene if location_raw in ("scene", "本条 scene") else None))
    held_by = _holder(world, npc, spec.get("held_by", ""))
    if location is None and held_by == f"npc:{npc.id}":
        location = world.scenes.get(npc.state.location)
    if location is None:
        return  # 明确仍存在，但位置未知：只进档案，不伪造所在场景。
    from .event import apply_item_patch
    patch = {"op": "add", "item": item_id, "name": record["name"],
             "location": location.id,
             "note": record["current_assertion"]["note"] or entry["then"]}
    if held_by:
        patch["held_by"] = held_by
    apply_item_patch(world, patch, cause=f"往事投影：{memory.id}")


def _projection_states(world: World, npc: NPC, memory: Memory,
                       specs: list[dict]) -> None:
    for raw in specs:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()[:160]
        if not text or any(f.get("source_memory") == memory.id and
                           f.get("text") == text for f in npc.state.facts):
            continue
        index = len(npc.state.facts) + 1
        fact_id = f"sf-{npc.id}-{index}"
        used = {str(f.get("id", "")) for f in npc.state.facts}
        while fact_id in used:
            index += 1
            fact_id = f"sf-{npc.id}-{index}"
        review = raw.get("review_days")
        try:
            review_clock = (world.clock + max(0.0, float(review))
                            if review is not None else None)
        except (TypeError, ValueError):
            review_clock = None
        npc.state.facts.append({
            "id": fact_id, "text": text, "source_memory": memory.id,
            "since_clock": memory.occurred_clock,
            "review_clock": review_clock,
        })


def materialize_memory(world: World, npc: NPC, memory: Memory,
                       projection: dict) -> list[str]:
    """把一条已提议投影落进既有实体账本；未知值保持未知。"""
    if not isinstance(projection, dict):
        return ["记忆投影必须是对象"]
    age = projection.get("age_days")
    if memory.occurred_clock is None and age is not None:
        try:
            memory.occurred_clock = world.clock - max(0.0, float(age))
        except (TypeError, ValueError):
            return ["记忆 age_days 不是数字"]
    body_id = str(projection.get("embodied_as", memory.embodied_as or npc.id))
    if body_id == "self":
        body_id = npc.id
    if body_id not in world.npcs:
        return [f"记忆 {memory.id} 引用了不存在的身体实体：{body_id}"]
    memory.embodied_as = body_id
    if "accessible" in projection:
        memory.accessible = bool(projection.get("accessible"))
        memory.access_cause = str(projection.get("access_cause", ""))[:160]
    if body_id != npc.id:
        start = (memory.occurred_clock if memory.occurred_clock is not None
                 else world.clock)
        duration = projection.get("duration_days")
        try:
            end = start + max(0.0, float(duration or 0.0))
        except (TypeError, ValueError):
            return [f"记忆 {memory.id} 的 duration_days 不是数字"]
        world.record_memory_gap(
            world.npcs[body_id], actor=npc.id, body=body_id,
            started_clock=start, ended_clock=end,
            cause=memory.access_cause or f"行动者与身体分离：{memory.id}",
            source_memory=memory.id)
    scene_spec = projection.get("scene")
    memory_scene = (_projection_scene(world, npc, memory, scene_spec)
                    if isinstance(scene_spec, dict) else None)
    items = projection.get("items", [])
    if not isinstance(items, list):
        return ["记忆物品投影必须是列表"]
    materialized_items: list[dict] = []
    for item in items[:12]:
        if isinstance(item, dict):
            _projection_item(world, npc, memory, item, memory_scene)
            if not (not str(item.get("ref", "")).strip()
                    and _looks_like_body_state(item)):
                materialized_items.append(dict(item))
    states = projection.get("current_states", [])
    if not isinstance(states, list):
        return ["记忆当前状态投影必须是列表"]
    _projection_states(world, npc, memory, states[:12])
    canonical_projection = dict(projection)
    canonical_projection["items"] = materialized_items
    memory.projections = [canonical_projection]
    world.log("memory_projected", f"{npc.name} 的一段往事已对齐世界账本",
              f"记忆：{memory.id}", {"npc": npc.id, "memory": memory.id,
                                      "scene": memory_scene.id
                                      if memory_scene else ""})
    return []


def _projection_errors(memory: Memory, projection: dict) -> list[str]:
    if not isinstance(projection, dict):
        return [f"记忆 {memory.id} 的 projection 必须是对象"]
    if not isinstance(projection.get("items", []), list):
        return [f"记忆 {memory.id} 的 items 必须是列表"]
    if not isinstance(projection.get("current_states", []), list):
        return [f"记忆 {memory.id} 的 current_states 必须是列表"]
    scene = projection.get("scene")
    if scene is not None and not isinstance(scene, dict):
        return [f"记忆 {memory.id} 的 scene 必须是对象或 null"]
    age = projection.get("age_days")
    if age is not None:
        try:
            float(age)
        except (TypeError, ValueError):
            return [f"记忆 {memory.id} 的 age_days 不是数字"]
    duration = projection.get("duration_days")
    if duration is not None:
        try:
            if float(duration) < 0:
                return [f"记忆 {memory.id} 的 duration_days 不能为负数"]
        except (TypeError, ValueError):
            return [f"记忆 {memory.id} 的 duration_days 不是数字"]
    body_id = str(projection.get("embodied_as",
                                 memory.embodied_as or "self"))
    # world 引用在 materialize_memory 中校验；这里仅拦明显坏结构。
    if body_id not in ("", "self") and len(body_id) > 80:
        return [f"记忆 {memory.id} 的 embodied_as 过长"]
    if "accessible" in projection and not isinstance(
            projection.get("accessible"), bool):
        return [f"记忆 {memory.id} 的 accessible 必须是布尔值"]
    return []


def materialize_stored_memories(world: World, npc: NPC,
                                require_all: bool = False) -> list[str]:
    """物化角色卡/世界生成结果中已经随记忆携带的结构化投影。"""
    ensure_memory_ids(npc)
    errors: list[str] = []
    for memory in npc.memories:
        if not memory.projections:
            if require_all:
                errors.append(f"记忆 {memory.id} 缺少结构化 projection")
            continue
        errors.extend(_projection_errors(memory, memory.projections[0]))
    if errors:
        return errors
    for memory in npc.memories:
        if memory.projections:
            errors.extend(materialize_memory(world, npc, memory,
                                             memory.projections[0]))
    return errors


def align_memories(llm: BaseLLM, world: World, npc: NPC,
                   memories: list[Memory]) -> list[str]:
    """分批对齐自由文本记忆；全部批次有效后才开始写实体账本。"""
    if not memories:
        return []
    ensure_memory_ids(npc)
    pending = [memory for memory in memories if not memory.projections]
    if not pending:
        return materialize_stored_memories(world, npc)
    base_payload = {
        "world": {"now_clock": world.clock, "now": world.now(),
                  "description": world.description},
        "npc": {"id": npc.id, "name": npc.name,
                "current_location": npc.state.location},
        "known_scenes": [{"id": s.id, "name": s.name}
                         for s in world.scenes.values()],
        "known_items": ([{"id": i.get("id"), "name": i.get("name"),
                          "location": s.id, "active_now": True}
                         for s in world.scenes.values() for i in s.items]
                        + [{"id": item.get("id"), "name": item.get("name"),
                            "active_now": False}
                           for item in world.past_items.values()]),
    }
    by_id = {memory.id: memory for memory in pending}
    proposed: dict[str, dict] = {}
    # 导入可以有上千条过去；一次塞满会重新制造上下文上限。每批只读 24 条，
    # 但在所有批次结构检查通过前不落库，避免半个角色被导入。
    for start in range(0, len(pending), 24):
        batch = pending[start:start + 24]
        payload = dict(base_payload)
        payload["memories"] = [{"id": m.id, "content": m.content}
                               for m in batch]
        data = llm.chat_json(_ALIGN_SYSTEM,
                             json.dumps(payload, ensure_ascii=False))
        rows = data.get("memories", [])
        if not isinstance(rows, list):
            return ["记忆对齐输出缺少 memories 列表"]
        batch_ids = {memory.id for memory in batch}
        for row in rows:
            if not isinstance(row, dict):
                continue
            memory_id = str(row.get("memory_id", ""))
            if memory_id not in batch_ids or memory_id in proposed:
                continue
            structure_errors = _projection_errors(by_id[memory_id], row)
            if structure_errors:
                return structure_errors
            proposed[memory_id] = row
    missing = set(by_id) - set(proposed)
    if missing:
        return [f"记忆 {memory_id} 没有得到事实投影"
                for memory_id in sorted(missing)]
    errors: list[str] = []
    for memory_id, row in proposed.items():
        errors.extend(materialize_memory(world, npc, by_id[memory_id], row))
    return errors


def apply_state_fact_patch(world: World, patch: dict,
                           cause: str = "状态复查") -> list[str]:
    """通用当前状态覆写；只提供增删改语法，不解释事实属于哪种题材。"""
    npc_id = str(patch.get("npc", patch.get("target", ""))).removeprefix("npc:")
    npc = world.npcs.get(npc_id)
    if npc is None:
        return ["状态事实引用了不存在的 NPC"]
    op = str(patch.get("op", "")).strip()
    if op not in ("add", "change", "remove"):
        return ["状态事实操作必须是 add/change/remove"]
    fact_id = str(patch.get("fact", patch.get("id", ""))).strip()
    existing = next((fact for fact in npc.state.facts
                     if str(fact.get("id", "")) == fact_id), None)
    if op in ("change", "remove") and existing is None:
        return [f"状态事实不存在：{fact_id}"]
    text = str(patch.get("text", "")).strip()[:160]
    if op in ("add", "change") and not text:
        return ["新增或改写状态事实必须给出 text"]
    if op == "add":
        index = len(npc.state.facts) + 1
        used = {str(f.get("id", "")) for f in npc.state.facts}
        fact_id = fact_id or f"sf-{npc.id}-{index}"
        while fact_id in used:
            index += 1
            fact_id = f"sf-{npc.id}-{index}"
        existing = {"id": fact_id, "since_clock": world.clock,
                    "source_event": world.turn}
        npc.state.facts.append(existing)
    if op == "remove":
        npc.state.facts.remove(existing)
    else:
        review = patch.get("review_days")
        try:
            review_clock = (world.clock + max(0.0, float(review))
                            if review is not None else None)
        except (TypeError, ValueError):
            return ["状态事实 review_days 不是数字"]
        existing["text"] = text
        existing["review_clock"] = review_clock
        existing["last_cause"] = str(patch.get("why", cause))[:160]
    world.log("npc_state_fact_changed",
              f"{npc.name} 的当前状态：{text if op != 'remove' else '不再有 ' + fact_id}",
              str(patch.get("why", cause))[:160] or cause,
              {"npc": npc.id, "fact": fact_id, "op": op,
               "location": npc.state.location})
    return []


def apply_scene_state_patch(world: World, patch: dict,
                            cause: str = "局部状态变化") -> list[str]:
    """写入场景的局部状态；duration_days 到期后由心跳自动清除。"""
    from .event import emit

    scene_id = str(patch.get("scene", patch.get("location", ""))).strip()
    # 模型载荷允许用统一实体引用（scene:id）；历史层内部统一使用
    # 场景字典的裸 id，避免 entity_events 与普通心跳走出两套语义。
    if scene_id.startswith("scene:"):
        scene_id = scene_id[6:]
    scene = world.scenes.get(scene_id)
    if scene is None:
        return ["局部场景状态引用了不存在的场景"]
    op = str(patch.get("op", "")).strip()
    if op not in ("add", "change", "remove"):
        return ["局部场景状态操作必须是 add/change/remove"]
    fact_id = str(patch.get("fact", patch.get("id", ""))).strip()
    existing = next((fact for fact in scene.state_facts
                     if str(fact.get("id", "")) == fact_id), None)
    if op in ("change", "remove") and existing is None:
        return [f"局部场景状态不存在：{fact_id}"]
    text = str(patch.get("text", "")).strip()[:160]
    if op in ("add", "change") and not text:
        return ["新增或改写局部场景状态必须给出 text"]
    try:
        duration = (None if patch.get("duration_days") is None
                    else float(patch.get("duration_days")))
    except (TypeError, ValueError):
        return ["局部场景状态 duration_days 不是数字"]
    if duration is not None and duration <= 0:
        return ["局部场景状态 duration_days 必须大于 0"]
    if op == "add":
        index = len(scene.state_facts) + 1
        used = {str(f.get("id", "")) for f in scene.state_facts}
        fact_id = fact_id or f"ssf-{scene.id}-{index}"
        while fact_id in used:
            index += 1
            fact_id = f"ssf-{scene.id}-{index}"
        existing = {"id": fact_id, "since_clock": world.clock,
                    "source_event": world.turn}
        scene.state_facts.append(existing)
    expires = existing.get("expires_clock") if existing else None
    if duration is not None:
        expires = world.clock + duration
    if op == "remove":
        text = str(existing.get("text", ""))
        expires = existing.get("expires_clock")
        scene.state_facts.remove(existing)
    else:
        existing["text"] = text
        existing["expires_clock"] = expires
        existing["last_cause"] = str(patch.get("why", cause))[:160]
    params = {"scene": scene.id, "fact": fact_id, "op": op,
              "text": text}
    if expires is not None:
        params["expires_clock"] = float(expires)
    errors = emit(world, "scene_state_changed", params,
                  cause=str(patch.get("why", cause))[:160] or cause)
    if errors:
        # emit 只会因结构/引用失败；回滚这次内存修改，避免半条状态。
        if op == "add" and existing in scene.state_facts:
            scene.state_facts.remove(existing)
        elif op == "remove":
            scene.state_facts.append(existing)
        return errors
    return []


def expire_scene_state_facts(world: World) -> list[str]:
    """清除到期的局部状态，并把结束本身写入账本。"""
    summaries: list[str] = []
    for scene in world.scenes.values():
        due = [dict(fact) for fact in scene.state_facts
               if fact.get("expires_clock") is not None and
               float(fact.get("expires_clock", 0.0)) <= world.clock + 1e-9]
        for fact in due:
            errors = apply_scene_state_patch(
                world, {"scene": scene.id, "op": "remove",
                        "fact": fact.get("id", ""),
                        "why": "局部场景状态到期"},
                cause="局部场景状态到期")
            if errors:
                summaries.extend(errors)
            else:
                summaries.append(world.events[-1].summary)
    return summaries
