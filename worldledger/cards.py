"""NPC 卡：按需生成之外，玩家可以自由创建、导入、导出 NPC。

角色卡 = 名字 + 人设 + 性格 + 记忆 + 关系值（+ 可选初始位置）。
导入走包络线校验；导出生成可分享的 JSON 文件；
跨世界借人：把另一个世界的 NPC 原样迁移进来（记忆完整保留）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import physics
from .history import (align_memories, materialize_stored_memories,
                      memory_from_value)
from .llm import BaseLLM
from .store import Memory, NPC, World, canonical_targets

CARDS_DIR = Path("worldledger_cards")


def _new_npc_id(world: World) -> str:
    i = len(world.npcs) + 1
    while f"n-{i}" in world.npcs:
        i += 1
    return f"n-{i}"


def _place(world: World, npc: NPC, scene_id: str) -> None:
    npc.state.location = scene_id
    npc.state.mark(world.turn, world.clock)
    world.npcs[npc.id] = npc
    world.scenes[scene_id].npcs.append(npc.id)


def create_npc(world: World, name: str, persona: str,
               traits: dict | None = None, memories: list[str] | None = None,
               scene_id: str | None = None,
               llm: BaseLLM | None = None) -> list[str]:
    """手动创建 NPC（玩家写人设）。返回错误列表，空即成功。"""
    if _name_unsafe(name):
        return ["名字含非法字符（路径分隔符/上级目录）"]
    if any(n.name == str(name).strip() for n in world.npcs.values()):
        return [f"已有同名 NPC：「{name.strip()}」"]
    if _active_count(world) >= physics.MAX_NPCS:
        return [f"活跃人口已达上限（{physics.MAX_NPCS}）"]
    scene_id = scene_id or world.player.get("location", "")
    if scene_id not in world.scenes:
        return ["场景不存在"]
    card = {"name": name, "persona": persona, "traits": traits or {},
            "memories": memories or [], "relationship": 0}
    errors = physics.validate_npc_card(card)
    if errors:
        return errors
    npc = NPC(id=_new_npc_id(world), name=name.strip(), persona=persona.strip(),
              traits=dict(traits or {}))
    for m in memories or []:
        npc.memories.append(memory_from_value(m, world.turn, world.clock))
    if llm is None and any(not m.projections for m in npc.memories):
        return ["自由文本记忆必须经过模型对齐，或随卡提供结构化 projection"]
    _place(world, npc, scene_id)
    alignment = (align_memories(llm, world, npc, npc.memories)
                 if llm is not None else materialize_stored_memories(world, npc))
    if alignment:
        world.scenes[scene_id].npcs.remove(npc.id)
        world.npcs.pop(npc.id, None)
        return alignment
    world.log("npc_created", f"NPC「{npc.name}」登场", "创建NPC命令",
              {"npc": npc.id, "location": scene_id})
    return []


def import_npc(world: World, card: dict, scene_id: str | None = None,
               llm: BaseLLM | None = None) -> list[str]:
    """导入角色卡（外部文件 / 另一个世界 / 玩家手写 JSON）。"""
    if _name_unsafe(card.get("name", "")):
        return ["名字含非法字符（路径分隔符/上级目录）"]
    if any(n.name == str(card.get("name", "")).strip()
           for n in world.npcs.values()):
        return [f"已有同名 NPC：「{str(card.get('name', '')).strip()}」"]
    if _active_count(world) >= physics.MAX_NPCS:
        return [f"活跃人口已达上限（{physics.MAX_NPCS}）"]
    errors = physics.validate_npc_card(card)
    if errors:
        return errors
    scene_id = scene_id or card.get("location") \
        or world.player.get("location", "")
    if scene_id not in world.scenes:
        return [f"场景不存在：{scene_id}"]
    npc = NPC(
        id=_new_npc_id(world),
        name=str(card.get("name", "")).strip(),
        persona=str(card.get("persona", "")).strip(),
        traits=dict(card.get("traits", {})),
        relationship=int(card.get("relationship", 0)),
        goals=[dict(g) for g in card.get("goals", [])],
        memory_gaps=[dict(g) for g in card.get("memory_gaps", [])
                     if isinstance(g, dict)],
    )
    for m in card.get("memories", []):
        npc.memories.append(memory_from_value(m, world.turn, world.clock))
    if llm is None and any(not m.projections for m in npc.memories):
        return ["自由文本记忆必须经过模型对齐，或随卡提供结构化 projection"]
    _place(world, npc, scene_id)
    alignment = (align_memories(llm, world, npc, npc.memories)
                 if llm is not None else materialize_stored_memories(world, npc))
    if alignment:
        world.scenes[scene_id].npcs.remove(npc.id)
        world.npcs.pop(npc.id, None)
        return alignment
    world.log("npc_imported", f"NPC「{npc.name}」导入（记忆 "
              f"{len(npc.memories)} 条）", "导入NPC命令",
              {"npc": npc.id, "location": scene_id})
    return []


def _safe_filename(name: str) -> str:
    """NPC 名 → 安全文件名：路径分隔符与非法字符全部替换。"""
    cleaned = re.sub(r'[\\/:*?"<>|]', '_', str(name)).strip('. _')
    return cleaned or "unnamed"


def _name_unsafe(name: str) -> bool:
    """名字不可用于路径：含分隔符 / 上级目录 / 控制字符。"""
    return (".." in str(name)
            or any(c in str(name) for c in '\\/:*?"<>|')
            or any(ord(c) < 32 for c in str(name)))


def npc_card(npc: NPC) -> dict:
    """导出角色卡：可分享、可再导入的纯数据（驱动力随行）。"""
    return {
        "name": npc.name,
        "persona": npc.persona,
        "traits": dict(npc.traits),
        "memories": [
            {"id": m.id, "turn": m.turn, "content": m.content,
             "importance": m.importance, "keywords": list(m.keywords),
             "occurred_clock": m.occurred_clock,
             "accessible": m.accessible, "access_cause": m.access_cause,
             "embodied_as": m.embodied_as,
             "projections": [dict(p) for p in m.projections]}
            for m in npc.memories],
        "memory_gaps": [dict(g) for g in npc.memory_gaps],
        "goals": [dict(g) for g in npc.goals],
        "relationship": npc.relationship,
    }


def export_npc(npc: NPC, path: str | Path | None = None) -> Path:
    safe = _safe_filename(npc.name)
    p = Path(path) if path else CARDS_DIR / f"{safe}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(npc_card(npc), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def import_from_world(target: World, source: World,
                      npc_name: str, llm: BaseLLM | None = None) -> list[str]:
    """跨世界借人：把 source 世界的 NPC 原样迁入 target（记忆完整）。"""
    src_npc = next((n for n in source.npcs.values()
                    if n.name == npc_name), None)
    if src_npc is None:
        return [f"「{source.name}」里没有 NPC「{npc_name}」"]
    errors = import_npc(target, npc_card(src_npc), llm=llm)
    if errors:
        return errors
    imported = target.npcs[list(target.npcs)[-1]]
    target.log("npc_imported",
               f"NPC「{npc_name}」从「{source.name}」跨世界迁入",
               "导入NPC命令", {"npc": imported.id, "from_world": source.name})
    return []


def inject_memory(world: World, npc: NPC, content: str,
                  llm: BaseLLM | None = None) -> list[str]:
    """玩家直接写入一条记忆（玩家自由导入记忆，最高重要度）。"""
    content = str(content).strip()
    if not (physics.MIN_TEXT <= len(content) <= physics.MAX_TEXT):
        return ["记忆内容为空或超长"]
    if llm is None:
        return ["自由文本记忆必须经过模型对齐"]
    world.remember(npc, content, cause="玩家设定", kind="npc_memory",
                   importance=1.0)
    memory = npc.memories[-1]
    if llm is None:
        return []
    return align_memories(llm, world, npc, [memory])


def _active_count(world: World) -> int:
    """活跃人口：当前可行动且不在雾中的人。"""
    return sum(1 for n in world.npcs.values()
               if n.state.can_act and not n.in_fog)


def emerge_npc(world: World, name: str, persona: str, goal: dict | None,
               scene_id: str, reason: str,
               memories: list | None = None) -> tuple[list[str], str]:
    """涌现 NPC：世界自己长出新角色（事件驱动 / 生活驱动）。

    有因才生：reason 必填（怪谈招来研究者 / 开学季来了新学妹）。
    有界：活跃人口上限（雾中人不占）。有痕：npc_emerged 事件。
    新角色一律是完整实体；是否进入本轮细演由调度器按位置、行动、
    意图、目标和预算决定，而不是由「路人」身份决定。
    返回 (错误列表, 新 NPC id)。
    """
    if _name_unsafe(name):
        return ["名字含非法字符（路径分隔符/上级目录）"], ""
    if not reason or not str(reason).strip():
        return ["缘由必填（有因才生）"], ""
    if any(n.name == str(name).strip() for n in world.npcs.values()):
        return [f"已有同名 NPC：「{str(name).strip()}」"], ""
    if _active_count(world) >= physics.MAX_NPCS:
        return [f"活跃人口已达上限（{physics.MAX_NPCS}）"], ""
    if not (physics.MIN_TEXT <= len(str(name).strip()) <= 20):
        return ["名字为空或超长"], ""
    if not (physics.MIN_TEXT <= len(str(persona).strip())
            <= physics.MAX_TEXT):
        return ["人设为空或超长"], ""
    if scene_id not in world.scenes:
        return [f"场景不存在：{scene_id}"], ""
    npc = NPC(id=_new_npc_id(world), name=str(name).strip(),
              persona=str(persona).strip())
    for memory in memories or []:
        npc.memories.append(memory_from_value(memory, world.turn, world.clock))
    if any(not memory.projections for memory in npc.memories):
        return ["涌现角色的初始记忆必须携带结构化 projection"], ""
    if goal and str(goal.get("text", "")).strip():
        npc.goals.append({
            "id": str(goal.get("id", f"g-1")),
            "text": str(goal["text"]).strip(),
            "progress": max(0.0, min(1.0, float(goal.get("progress", 0)))),
            "targets": canonical_targets(world, goal.get("targets")),
        })
    _place(world, npc, scene_id)
    alignment = materialize_stored_memories(world, npc)
    if alignment:
        world.scenes[scene_id].npcs.remove(npc.id)
        world.npcs.pop(npc.id, None)
        return alignment, ""
    world.log("npc_emerged",
              f"新角色「{npc.name}」登场（{reason}）",
              "世界演化",
              {"npc": npc.id, "name": npc.name, "location": scene_id,
               "reason": str(reason)[:80]})
    return [], npc.id
