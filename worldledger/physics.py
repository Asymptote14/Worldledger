"""法则档案的最小包络线：机械校验与天变。

包络线只管「能不能模拟」——结构合法、参数在界内、自洽可执行；
「好不好玩」是创作自由，不归这里管。
"""
from __future__ import annotations

from .store import Law, LawProfile, World

MAX_LAWS = 8            # 法则条目上限
MAX_TEXT = 200          # 单条触发/后果文本上限
MAX_NPCS = 12           # 同时在场的活跃人口上限（雾中人不占名额）
RELATION_BOUND = 100    # 关系值边界
# 人物经历档案不设固定条数上限；读取窗口在 store.experience_window 控制。
MIN_TEXT = 2            # 触发/后果最短长度
MAX_NPC_NAME = 20       # NPC 名字长度上限
MAX_PERSONA = 300       # 人设文本长度上限


def validate_law(law: Law) -> list[str]:
    errors: list[str] = []
    if not (MIN_TEXT <= len(law.trigger.strip()) <= MAX_TEXT):
        errors.append(f"法则「{law.id}」触发条件长度越界")
    if not (MIN_TEXT <= len(law.effect.strip()) <= MAX_TEXT):
        errors.append(f"法则「{law.id}」后果文本长度越界")
    if not (0.0 <= law.intensity <= 1.0):
        errors.append(f"法则「{law.id}」强度越界（须在 0.0–1.0）")
    return errors


def validate_profile(profile: LawProfile) -> list[str]:
    errors: list[str] = []
    if len(profile.expectation.strip()) < MIN_TEXT:
        errors.append("期望文本为空")
    if not profile.atmosphere.strip():  # 单字氛围合法（雪/夜/雷）
        errors.append("氛围基调为空")
    if len(profile.laws) > MAX_LAWS:
        errors.append(f"法则条目过多（上限 {MAX_LAWS}）")
    ids = [law.id for law in profile.laws]
    if len(ids) != len(set(ids)):
        errors.append("法则 id 重复")
    for law in profile.laws:
        errors.extend(validate_law(law))
    return errors


def apply_law_change(world: World, new_profile: LawProfile) -> list[str]:
    """天变：新档案通过包络线 → 版本 +1 → 事件入日志 → 即时生效。

    返回错误列表；为空表示天变成功。
    """
    errors = validate_profile(new_profile)
    if errors:
        return errors
    new_profile.version = world.law_profile.version + 1
    old = world.law_profile
    world.law_profile = new_profile
    world.log(
        "law_changed",
        f"天变：法则档案 v{old.version} → v{new_profile.version}",
        "改法则命令",
        {
            "old_laws": [law.id for law in old.laws],
            "new_laws": [law.id for law in new_profile.laws],
            "atmosphere": new_profile.atmosphere,
        },
    )
    return []


def clamp_relationship(value: int) -> int:
    return max(-RELATION_BOUND, min(RELATION_BOUND, value))


def adjust_relationship(npc, delta: int) -> int:
    npc.relationship = clamp_relationship(npc.relationship + delta)
    return npc.relationship


def validate_npc_card(card: dict) -> list[str]:
    """NPC 角色卡的包络线：名字 / 人设 / 性格 / 记忆 / 关系值。"""
    errors: list[str] = []
    name = str(card.get("name", "")).strip()
    if not (MIN_TEXT <= len(name) <= MAX_NPC_NAME):
        errors.append(f"NPC 名字长度须在 {MIN_TEXT}-{MAX_NPC_NAME} 字")
    persona = str(card.get("persona", "")).strip()
    if not (MIN_TEXT <= len(persona) <= MAX_PERSONA):
        errors.append(f"人设长度须在 {MIN_TEXT}-{MAX_PERSONA} 字")
    traits = card.get("traits", {})
    if not isinstance(traits, dict):
        errors.append("性格必须是字典")
    else:
        for key in traits:
            if len(str(key)) > MAX_NPC_NAME:
                errors.append(f"性格名过长：{key}")
    memories = card.get("memories", [])
    if not isinstance(memories, list):
        errors.append("记忆必须是列表")
    else:
        for m in memories:
            content = m.get("content", "") if isinstance(m, dict) else m
            if not str(content).strip() or len(str(content)) > MAX_TEXT:
                errors.append("记忆条目为空或超长")
    rel = card.get("relationship", 0)
    if not isinstance(rel, (int, float)) or not (
            -RELATION_BOUND <= rel <= RELATION_BOUND):
        errors.append(f"关系值越界（-{RELATION_BOUND}..{RELATION_BOUND}）")
    if not isinstance(card.get("memory_gaps", []), list):
        errors.append("记忆断档必须是列表")
    return errors
