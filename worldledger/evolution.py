"""NPC 状态演化：分层混合（观察即动作）。

- 近处（玩家所在场景）：心跳演化，每个玩家回合都活。
- 远处：不演化，读取时补算（catch-up）——世界不因玩家缺席而冻结。
- 一切状态变化都写回世界库，携带原因引用；观察谁，谁就演化得更细。
"""
from __future__ import annotations

import json as _json
import math
from copy import deepcopy

from .event import (EVENT_TYPES, _close_item_fold, apply_item_patch, emit,
                    named_active_npcs_in_text, validate_event, validate_refs)
from .llm import INTERACT_COOLDOWN, BaseLLM
from .history import (apply_scene_state_patch, apply_state_fact_patch,
                      expire_scene_state_facts)
from .store import (ActionState, IntentState, NPC, World, active_items, canonical_targets,
                    day_of, ensure_memory_ids, experience_payload, experience_window, game_time,
                    memory_effectiveness, memory_gaps_payload, mood_now, phase_name, phase_of,
                    resolve_target, target_snapshot, text_similarity,
                    touch_items, weather_now)
from .worldgen import emerge_place, extend_scene
from . import cards, physics

# 向后兼容再导出（世界时钟现在住在 store）
PHASE_NAMES = ["清晨", "白昼", "黄昏", "深夜"]
TURNS_PER_PHASE = 6
TURNS_PER_DAY = TURNS_PER_PHASE * 4

# —— 大事强度开关（节奏旋钮）——
BIG_EVENT_THRESHOLD = 0.7  # 强度 ≥ 此值算「大事」；调大=更宽松，调小=更严格
BIG_EVENT_COOLDOWN = 12    # 大事之后世界喘息的回合数（半天；实测 24 过紧，压世界节奏）

WORLD_MOOD_HALF_LIFE = 72  # 世界氛围回温半衰期（心跳步）
CLOCK_EPSILON = 1e-9
DAILY_REPEAT_WINDOW_DAYS = 1.0
DAILY_REPEAT_CLOSE_DAYS = 0.25
DAILY_REPEAT_SIMILARITY = 0.42
DAILY_REPEAT_CLOSE_SIMILARITY = 0.20


def _repeats_recent_daily(world: World, location: str, detail: str) -> bool:
    """抑制同场景的同义环境播报，不吞掉带物态后果的 trace。

    近四分之一天里，同一生活片段常被模型换几个字重说，使用较宽松的
    近似阈值；更久以前只压明显同义。世界状态变化由 trace 另行折叠，
    不经过这里，所以去掉的是监控式文案，不是事实。
    """
    detail = str(detail).strip()
    if not location or not detail:
        return False
    for event in reversed(world.events):
        age = max(0.0, world.clock - float(event.day))
        if age > DAILY_REPEAT_WINDOW_DAYS:
            break
        if event.kind != "daily_life":
            continue
        params = (event.payload or {}).get("event_params", event.payload or {})
        if params.get("location") != location or params.get("item"):
            continue
        previous = str(params.get("detail", "")).strip()
        similarity = text_similarity(detail, previous)
        threshold = (DAILY_REPEAT_CLOSE_SIMILARITY
                     if age <= DAILY_REPEAT_CLOSE_DAYS
                     else DAILY_REPEAT_SIMILARITY)
        if similarity >= threshold:
            return True
    return False


def elapsed_steps(world: World, then_clock: float) -> float:
    """两个世界钟点之间经过多少个心跳步。

    `turn` 是 append-only 账本的位置，不是时间。所有角色调度和冷却
    必须走这里，才不会因为一次裁决写了很多事件而让世界凭空过日子。
    """
    if world.heartbeat <= 0:
        return 0.0
    return max(0.0, (world.clock - then_clock) / world.heartbeat)


def mark_npc(world: World, npc: NPC) -> None:
    """把 NPC 的账本位置和真实时间一起打卡。"""
    npc.state.mark(world.turn, world.clock)


def cooldown_ready(world: World, key: str, steps: float) -> bool:
    """冷却只按真实经过的时间判断；缺失记录表示从未发生。"""
    then = world.social_clock.get(key)
    if then is None:
        # 兼容旧路径和旧档：social 中已有键却没有时钟，视为刚发生。
        # 这样不会因升级而让旧冷却立刻失效；没有键才代表从未发生。
        return key not in world.social
    return elapsed_steps(world, then) + CLOCK_EPSILON >= steps


def mark_social(world: World, key: str) -> None:
    """保留 social 的账本标记，同时记录冷却用的世界钟。"""
    world.social[key] = world.turn
    world.social_clock[key] = world.clock


def push_world_mood(world: World, delta: float, reason: str = "") -> float:
    """世界氛围：事件驱动（可审计，带原因），钳制在 -1..1。"""
    world.mood_value = max(-1.0, min(1.0, world.mood_value + delta))
    if reason:
        world.mood_reason = reason
    return world.mood_value


def decay_world_mood(world: World, turns: float) -> float:
    """世界氛围：时间回温——平静的日子把氛围拉回基调（均值回归）。"""
    if turns <= 0 or world.mood_value == 0.0:
        return world.mood_value
    world.mood_value *= math.exp(-turns / WORLD_MOOD_HALF_LIFE)
    if abs(world.mood_value) < 0.05:
        world.mood_value = 0.0
        world.mood_reason = ""
    return world.mood_value


def last_big_event_clock(world: World, threshold: float) -> float | None:
    """最近一次大事发生时的世界钟；事件编号不是时间。"""
    for e in reversed(world.events):
        if e.kind == "world_event":
            params = (e.payload or {}).get("event_params", {})
            try:
                if float(params.get("intensity", 0.0) or 0.0) >= threshold:
                    return e.day
            except (TypeError, ValueError):
                continue
    return None


def weather_of(world: World) -> str:
    return world.law_profile.atmosphere


# ---------------- 情绪动力学 ----------------

MOOD_DECAY_PER_TURN = 0.005  # 情绪向 0 回落的每回合速率
MOOD_OVER = 0.3              # |mood_value| 超过此值覆盖作息气质标签


def mood_sensitivity(npc: NPC) -> float:
    """兼容入口：情绪幅度由读过经历的裁决结果决定，不查固定标签。"""
    return 1.0


def push_mood(npc: NPC, delta: float, reason: str) -> float:
    """事件驱动：累积本轮已经结合经历裁决出的情绪变化。"""
    npc.state.mood_value = max(-1.0, min(1.0,
        npc.state.mood_value + delta * mood_sensitivity(npc)))
    npc.state.mood_reason = reason
    return npc.state.mood_value


def decay_mood(npc: NPC, turns: float) -> float:
    """时间衰减：情绪随时间向 0 回落。"""
    amount = min(abs(npc.state.mood_value), turns * MOOD_DECAY_PER_TURN)
    npc.state.mood_value = max(-1.0, min(1.0,
        npc.state.mood_value - math.copysign(amount, npc.state.mood_value)))
    return npc.state.mood_value


def mood_label(npc: NPC, fallback: str) -> str:
    """情绪标签：强度超阈值时覆盖作息气质，否则保持气质。"""
    v = npc.state.mood_value
    if v >= MOOD_OVER:
        return "欣快"
    if v <= -MOOD_OVER:
        return "忧郁"
    return fallback


_CATCHUP_SYSTEM = """TASK:NPCCATCHUP
你是世界的演化器。根据 NPC 的真实经历与当前处境，补算它从上次演化到现在的
状态变化。输出严格 JSON：
{"location": "场景 id", "activity": "正在做什么",
 "mood": "情绪", "moved": true或false,
 "memory": "这段时间值得记住的事（第一人称「我」，不要出现「玩家」等系统术语）；没有值得记的就给空字符串"}
如果 NPC 有一个进行中的主动动作，location 必须保持当前所在场景；
在途动作只有完成时才抵达目标地点。persona_origin / trait_origins 只是
人物起点；后来的 lived_experiences 与 beliefs 可以改变它。不要输出 JSON
以外的任何内容。"""

_HEARTBEAT_SYSTEM = """TASK:NPCHEARTBEAT
你是世界的演化器。玩家就在这个场景里，NPC 正在过它此刻的生活。
输出严格 JSON：
{"activity": "正在做什么", "mood": "情绪", "location": "场景 id",
 "interaction": {"with": "同在场景的另一 NPC id", "line": "台词"} 或 null}
人物此刻怎么生活，要从 lived_experiences、beliefs 和当前状态长出来；
persona_origin / trait_origins 只是起点。不要输出 JSON 以外的任何内容。"""

def _schema_help(*types: str) -> str:
    """把事件注册表的参数契约转成 prompt 用的说明文本。"""
    lines = []
    for t in types:
        _, params = EVENT_TYPES[t]
        ps = "、".join(
            f"{name}（{'必填' if s.get('required') else '可选'}，{s['kind']}"
            + (f"，{s['min']}-{s['max']}" if "min" in s else "") + "）"
            for name, s in params.items())
        lines.append(f"- {t}：{ps}")
    return "\n".join(lines)


_WORLDEVENT_SYSTEM = f"""TASK:WORLDEVENT
你是世界的演化器。世界时钟走到了现在，提案 0-2 个世界事件
（固化事件调用）。事件类型只能是：world_event | item_arrive | weather_shift。
参数必须严格按以下 schema 写（location/id 必须是世界内真实存在的场景 id）：
{_schema_help("world_event", "item_arrive", "weather_shift")}
输出严格 JSON：{{"events": [{{"type": "事件类型", "params": {{...}}}}]}}
不要输出 JSON 以外的任何内容。"""

_RUMOR_SYSTEM = """TASK:RUMOR
你是流言的转述者。把一个世界事件转述成 NPC 听到的传闻版本：
允许走样、夸大或误解，但保持事件的大致轮廓。
输出严格 JSON：{"content": "一句传闻（20-60 字）"}
不要输出 JSON 以外的任何内容。"""

_ACTIONRESOLVE_SYSTEM = """TASK:ACTIONRESOLVE
你是动作结局的裁决器。NPC 的主动动作完成了，给出它的结局：
可能有了收获、碰了壁、或发现了新线索。输出严格 JSON：
{"outcome": "结局描述（一两句话）",
 "patch": {"op": "add|remove|change", "item": "物品id",
            "location": "场景id", "name": "名称", "note": "新状态",
 "held_by": "player | npc:角色id | 空（add 时必填）"} 或 null}
若 arrival.player_present 为 true，说明角色抵达目的地时玩家仍在眼前；为
false 则说明他已不在那里。把这当作抵达时的可见事实，不要编造玩家去了哪。
patch 是结局对物品的一等状态改写（取走了信 = 覆写状态；
拆开了信 = note 改写；没碰任何物品 = null）。
change/remove 的 item 只能引用载荷里 scene_items / targets_now 出现的真实 id；
若动作让一个此前未显现、但会持续存在的物品首次进入世界，add 是唯一例外：
给不在 used_item_ids 中的稳定 item id、名称、场景和 held_by；add 的
held_by 必须明确写出：
角色携带写 npc:角色id，留在场景写空字符串。不要把它只留在叙事里。
没有真实物态可落时才留 null，引擎会驳回假引用。叙事里说发生的事，
patch 负责让它真发生。规则：碰到了就改（指尖碰到封蜡 =
封蜡的状态变了）；没碰到才写 null。不要只叙事不改状态。
取走、放进口袋、带走 = held_by 必须写行动者的 npc:角色id；放下 =
held_by 写空字符串；只是查看、触摸、打开而未拿走时省略 held_by。
结局不得凭空宣称某个持续物品被拿走、插入、损坏或留下而不给 patch。
结局不得编造与已有 NPC 的对话或信息交换；那是 interaction 事件，必须在
双方同场的后续裁决中单独发生。
结局应符合这个人已经活过的 lived_experiences 与 beliefs；persona_origin
只是起点，不得替代后来积累的经历。
lived_experiences 只包含当前可访问记忆；memory_gaps 不含事件内容，不得
从空白时段脑补人物当时看见或做过什么。
不要输出 JSON 以外的任何内容。"""

_NPCINITIATE_SYSTEM = """TASK:NPCINITIATE
你是角色的开口裁决器。玩家此刻就在这个场景里。判断角色是否要主动
向玩家开口说话。只有世界状态刚发生变化时才开口：
- 法则刚被修改（天变）、或刚听到与玩家/目标相关的传闻
- 与玩家的关系近期跨过阈值、或情绪到了极值
没发生什么特别的事 → 不开口（保持沉默）。
是否开口以及怎样开口，要结合 lived_experiences 与 beliefs；初始 persona
不是永久台词模板。
输出严格 JSON：{"open": true或false, "line": "开口的台词（20-50 字）"}
不要输出 JSON 以外的任何内容。"""

_NPCGOAL_SYSTEM = f"""TASK:NPCGOAL
你是角色的驱动力裁决器。角色有未完成的目标（可能多个）。逐一评估：
- persona_origin / trait_origins 只是人物起点。行动必须综合真实的
  lived_experiences、beliefs、目标和当前处境；经历可以改变早期性格。
- lived_experiences 只包含可访问记忆；memory_gaps 只是时间断档，不能当作
  那段时间发生过什么的知识来源。
- 目标与近期记忆/事件相关（最近的传闻、新线索恰好呼应目标）
  → 就提案一个推进它的主动事件，哪怕只是一小步：去相关地点打听、
  检查某个地方、或给玩家留一张纸条（note_left）。
- 即使没有立刻行动的理由，只要时间在走、日子在过，就推进
  goal_updates（0.1-0.2 的小步）——目标是生活的刻度，不是开关：
  平淡的日常也在向目标靠近（买菜、守摊、擦灯都算）。
- 角色可以去玩家还没去过的地方：location 给一个新 id（如 s-99），
  params 里同时给 place（新地名，2-10 字）——那个地方会以「地名」
  涌现（不细化，但真实存在），角色就过去做事了。
- 一次心跳最多提案一个事件（挑最成熟的那个目标）。
- 情绪或关系近期有变化时更容易行动；完全无关 → 不提案。
- `npc_acted` 只写该角色自己的行动或正在去往某处，不能在 action 里替代
  与既有角色的对话（例如「看见某人后对他说」）。抵达后真的要说话，
  必须在后续裁决走 interaction；还在路上时只能写赶路、寻找、等待。
- 提到未来想去、准备去或犹豫去某处，不等于已经移动：不给 location 就是
  原地发生的行动。只有实际开始跨场景移动时才给 location 目的地。
- npc_acted 的 days 是动作从开始到可裁决结局的世界天数，必须按动作本身
  估算；0 表示当场完成。跨场景移动必须大于 0，原地持续工作也可以大于 0。
  当行动者与身体主人不同且 location 跨出身体当前位置时，必须额外给
  `travel: true`，明确是这具身体正在移动；不能把行动者原本所在地点当成
  身体已经到达的地点。
  targets 只引用这次行动真实涉及的实体；requires 只列到完成时必须仍在
  行动地点的实体。requires 消失或离场时，引擎会有因地中止动作。
- `intent` 是角色眼下可撤回的短期打算，不是移动的别名，也不改变任何
  世界事实。只有打算刚形成、被替换，或因真实变化不再成立时才输出它：
  `{{"text": "黄昏去坑边核对印记", "targets": ["scene:s-crater"]}}`；
  `targets` 只可引用已给出的真实物品、场景或角色。若打算依赖某个未来时刻，
  用绝对世界钟填写 `earliest_clock`；若错过该时刻就不再合理，填写
  `latest_clock`（两者都是天数，不是回合编号）。引擎会拒绝窗口外的真实行动。
  省略 intent 表示不变；
  intent 为 null 表示放下当前打算。真正出发、抵达、对话、物品变化仍分别走
  npc_acted / npc_moved / npc_interaction / patch，不能只写在 intent 里。
事件类型只能是：npc_acted | note_left。
参数必须严格按以下 schema 写（location/id 必须是世界内真实存在的场景 id）：
{_schema_help("npc_acted", "note_left")}
输出严格 JSON：{{"events": [{{"type": "...", "params": {{...}}}}],
  "intent": {{"text": "短期打算", "targets": ["item:/scene:/npc: 引用"],
  "earliest_clock": 绝对世界钟（可选）, "latest_clock": 最晚开始钟点（可选）}}
  或 null（可省略），
  "goal_updates": {{"目标id":
  {{"progress": 0.0 到 1.0 的新进度,
   "because": "为什么变（一句话，引用真实发生的事——无因不推进）",
   "blocked_by": "被什么卡住（可选：item:/scene:/npc: 引用或裸名）",
   "blocked_note": "被卡住的原因（自由文本，可选）"}}}},
"new_goals": [{{"id": "g-N", "text": "新目标", "progress": 0.0}}]}}
规则补充：
- 进度提升必须给 because（有因才推进）；没有因由的提升会被驳回。
- 目标可以被卡住（blocked_by）：信被抢走、门锁着、人不在——被卡住时
  进度冻结，直到阻碍解除（blocked_by 留空 + because 说明）。
  阻碍是什么由你判断，引擎只验引用真不真。
当某个目标本次达成（进度到 1.0）时，new_goals 给出它自然长出的下一步
目标（达成是转折点，驱动不消失；平淡生活也是合理的下一步目标）；
没达成就不需要 new_goals。
留痕不夺权：默认留下痕迹（行动/纸条），不替玩家做决定。
- 记忆不得凭空声称对话：写「我告诉了他」「他对我说」之前，
  必须先走 interaction / 流言 / 纸条通道入账——账本无痕的对话
  不许写进记忆。
不要输出 JSON 以外的任何内容。"""


def _lived_payload(world: World, npc: NPC, query: str,
                   limit: int = 10) -> list[dict]:
    memories = experience_window(
        npc, world.turn, query=query,
        focus_ids=npc.state.memory_focus, limit=limit)
    return experience_payload(memories)


def _npc_payload(world: World, npc: NPC, query: str = "") -> dict:
    actor = world.actor_for_body(npc)
    lived = _lived_payload(world, actor, query)
    return {
        "id": npc.id, "name": npc.name,
        "actor": {"id": actor.id, "name": actor.name}
        if actor.id != npc.id else None,
        "body": {"id": npc.id, "name": npc.name,
                  "location": npc.state.location,
                  "activity": npc.state.activity},
        "persona_origin": actor.persona, "trait_origins": actor.traits,
        # 兼容旧模型；语义说明明确它们只是起点。
        "persona": actor.persona, "traits": actor.traits,
        "experience_count": len(actor.memories),
        "lived_experiences": lived,
        "memories": [row["content"] for row in lived],
        "beliefs": actor.beliefs[-5:],
        "memory_gaps": memory_gaps_payload(actor),
        "state": {"location": npc.state.location,
                  "activity": actor.state.activity, "mood": actor.state.mood,
                   "intent": {"text": actor.state.intent.text,
                               "targets": list(actor.state.intent.targets)},
                   "action_actor": npc.state.action.actor_id or actor.id,
                   "action": {"text": npc.state.action.text,
                             "location": npc.state.action.location,
                             "started_clock": npc.state.action.started_clock,
                             "due_clock": npc.state.action.due_clock,
                             "targets": list(npc.state.action.targets),
                             "requires": list(npc.state.action.requires)},
                   "can_act": npc.state.can_act,
                   "condition": npc.state.condition,
                   "facts": [dict(f) for f in actor.state.facts[-12:]]},
     }


def is_actionable(npc: NPC) -> bool:
    """行动资格是事实；雾仍是旧存档的细演/可见性边界。"""
    return npc.state.can_act and not npc.in_fog


def player_is_actionable(world: World) -> bool:
    """玩家与 NPC 服从同一个最低行动资格语义。"""
    return bool(world.player.get("can_act", True))


def player_condition(world: World) -> str:
    return str(world.player.get("condition", "")).strip() or "无法行动"


def _remove_from_scenes(world: World, npc: NPC) -> None:
    for scene in world.scenes.values():
        if npc.id in scene.npcs:
            scene.npcs.remove(npc.id)


def set_actionability(world: World, npc: NPC, can_act: bool,
                      condition: str, cause_event: str) -> list[str]:
    """记录角色能否继续行动，而不解释世界词汇里的具体含义。

    `condition` 可以是死亡、封印、昏迷、尚未出生或任何世界自有说法；
    引擎只执行其共同后果。恢复同样必须引用一条已入账的事实。
    """
    condition = str(condition).strip()
    cause_event = str(cause_event).strip()
    if not condition:
        return ["驳回行动状态：当前状态说明必填"]
    if not cause_event:
        return ["驳回行动状态：必须引用来源事件"]
    if len(condition) > 20 or len(cause_event) > 200:
        return ["驳回行动状态：状态说明或来源事件超长"]
    # 状态只应是近期已发生事实的后果。读取窗口以外的旧事不能被
    # 拿来突然解释今天的停摆；那会把账本引用重新变成叙事借口。
    source = next((event for event in reversed(world.events[-8:])
                   if event.summary == cause_event), None)
    if source is None:
        return ["驳回行动状态：来源事件不在近期账本中"]
    named_places = [scene.name for scene in world.scenes.values()
                    if scene.name and scene.name in condition]
    if named_places:
        return ["驳回行动状态：状态说明不得声明地点；位置必须走 npc_moved"]
    if bool(can_act) == npc.state.can_act:
        return []  # 行动资格未变：持续后果应走 state.facts

    if not can_act:
        # 未完成的行动已不可能继续，先把已经存在的承诺收尾留痕。
        cause = f"事件：{source.summary[:120]}"
        summaries = abort_action(world, npc, cause=cause)
        npc.state.action = ActionState()
        npc.state.pending_opener = ""
        npc.state.activity = condition
        _remove_from_scenes(world, npc)
        for key in [key for key in world.social
                    if key.startswith(npc.id + "->")
                    or key.endswith("->" + npc.id)]:
            del world.social[key]
            world.social_clock.pop(key, None)
    else:
        summaries = []
        cause = f"事件：{source.summary[:120]}"

    errors = emit(world, "npc_state_changed", {
        "npc": npc.id,
        "can_act": bool(can_act),
        "condition": condition,
        "cause_event": source.summary,
    }, cause=cause)
    if errors:
        return summaries + errors
    npc.state.can_act = bool(can_act)
    npc.state.condition = condition
    if can_act and not npc.in_fog:
        scene = world.scenes.get(npc.state.location)
        if scene is not None and npc.id not in scene.npcs:
            scene.npcs.append(npc.id)
    summaries.append(world.events[-1].summary)
    return summaries


def _entity_location(world: World, resolved: str) -> str | None:
    """返回实体所在场景；场景自身的位置就是自身。"""
    if resolved == "player":
        return str(world.player.get("location", ""))
    kind, _, ref = resolved.partition(":")
    if kind == "scene":
        return ref if ref in world.scenes else None
    if kind == "npc":
        npc = world.npcs.get(ref)
        return npc.state.location if npc is not None else None
    if kind == "item":
        for scene in world.scenes.values():
            if any(item.get("id") == ref for item in scene.items):
                return scene.id
    return None


def _find_item(world: World, item_id: str) -> tuple[object | None, dict | None]:
    for scene in world.scenes.values():
        for item in scene.items:
            if item.get("id") == item_id:
                return scene, item
    return None, None


def due_item_actions(world: World, limit: int = 3) -> list[dict]:
    """到点物品动作进入世界脉冲，不额外制造逐物品模型调用。"""
    due: list[dict] = []
    for scene in world.scenes.values():
        for item in scene.items:
            action = item.get("action")
            if not isinstance(action, dict) or not action.get("text"):
                continue
            due_clock = float(action.get("due_clock", 0.0) or 0.0)
            if due_clock <= world.clock + 1e-9:
                due.append({
                    "ref": f"item:{item.get('id', '')}",
                    "name": item.get("name", ""),
                    "location": scene.id,
                    "note": item.get("note", ""),
                    "action": dict(action),
                })
    due.sort(key=lambda entry: float(
        entry["action"].get("due_clock", 0.0) or 0.0))
    return due[:limit]


def abort_invalid_item_actions(world: World) -> list[str]:
    """显式前置条件失效时，有因地中止物品的持续行动。"""
    summaries: list[str] = []
    for scene in world.scenes.values():
        for item in list(scene.items):
            action = item.get("action")
            if not isinstance(action, dict) or not action.get("text"):
                continue
            failed: list[str] = []
            reasons: list[str] = []
            for ref in action.get("targets", []):
                if resolve_target(world, str(ref)) is None:
                    failed.append(str(ref))
                    reasons.append(f"目标 {ref} 已不存在")
            for ref in action.get("requires", []):
                resolved = resolve_target(world, str(ref))
                if resolved is None:
                    failed.append(str(ref))
                    reasons.append(f"前置实体 {ref} 已不存在")
                elif _entity_location(world, resolved) != scene.id:
                    failed.append(resolved)
                    reasons.append(f"前置实体 {resolved} 已离场")
            if not reasons:
                continue
            cause = "物品行动前置条件复查"
            for event in reversed(world.events):
                if any(_event_mentions_ref(event, ref) for ref in failed):
                    cause = f"事件#{event.turn}：{event.summary[:100]}"
                    break
            refs = [f"item:{item.get('id', '')}"]
            refs.extend(ref for ref in failed
                        if resolve_target(world, ref) is not None)
            params = {
                "title": "物品行动中止",
                "detail": f"「{item.get('name', '某物')}」未能继续：{reasons[0]}",
                "location": scene.id,
                "intensity": 0.2,
                "refs": list(dict.fromkeys(refs))[:8],
            }
            errors = emit(world, "world_event", params, cause=cause)
            if errors:
                summaries.extend(errors)
                continue
            item["action"] = {}
            item["last_turn"] = world.turn
            item["cause_turn"] = world.events[-1].turn
            summaries.append(world.events[-1].summary)
    return summaries


def _normalize_entity_event(world: World, proposal: dict) -> tuple[dict, list[str]]:
    """把多实体提案收束成可预演的单一事实和受控后果。"""
    if not isinstance(proposal, dict):
        return {}, ["多实体事件必须是对象"]
    location = str(proposal.get("location", ""))
    raw_participants = proposal.get("participants", [])
    if not isinstance(raw_participants, list) or not raw_participants:
        return {}, ["多实体事件必须引用参与实体"]
    participants: list[str] = []
    errors: list[str] = []
    for raw in raw_participants:
        resolved = resolve_target(world, str(raw))
        if resolved is None:
            errors.append(f"多实体事件引用不存在的实体：{raw}")
        elif resolved not in participants:
            participants.append(resolved)
    params = {
        "title": str(proposal.get("title", "")).strip(),
        "detail": str(proposal.get("detail", "")).strip(),
        "location": location,
        "intensity": proposal.get("intensity", 0.5),
        "refs": participants,
    }
    errors.extend(validate_event("world_event", params))
    errors.extend(validate_refs(world, "world_event", params,
                                cause="多实体事件"))
    if location in world.scenes:
        for resolved in participants:
            actual = _entity_location(world, resolved)
            if actual != location:
                errors.append(
                    f"参与实体 {resolved} 不在事件场景 {location}（实际 {actual or '未知'}）")

    item_patches = proposal.get("item_patches", [])
    actor_patches = proposal.get("actor_patches", [])
    state_fact_patches = proposal.get("state_fact_patches", [])
    scene_state_patches = proposal.get("scene_state_patches", [])
    item_actions = proposal.get("item_actions", [])
    raw_completes = proposal.get("completes", [])
    if (not isinstance(item_patches, list)
            or not isinstance(actor_patches, list)
            or not isinstance(state_fact_patches, list)
            or not isinstance(scene_state_patches, list)
            or not isinstance(item_actions, list)
            or not isinstance(raw_completes, list)):
        errors.append("多实体事件的后果必须是列表")
        item_patches, actor_patches, state_fact_patches, scene_state_patches, \
            item_actions, raw_completes = ([], [], [], [], [], [])
    if (len(item_patches) > 3 or len(actor_patches) > 3
            or len(state_fact_patches) > 3 or len(scene_state_patches) > 3
            or len(item_actions) > 3
            or len(raw_completes) > 3):
        errors.append("多实体事件的每类后果最多 3 条")
    if not item_patches and not actor_patches and not state_fact_patches \
            and not scene_state_patches \
            and not item_actions \
            and not raw_completes:
        errors.append("多实体事件至少要有一个可落库的后果")
    if raw_completes and not (item_patches or actor_patches
                              or state_fact_patches or scene_state_patches
                              or item_actions):
        errors.append("物品行动完成必须同时落库实际后果")
    for patch in item_patches:
        if not isinstance(patch, dict):
            errors.append("物品后果必须是对象")
            continue
        if str(patch.get("op", "")) == "add":
            errors.append("多实体事件暂不允许凭空新增参与物品")
            continue
        ref = f"item:{str(patch.get('item', '')).strip()}"
        if ref not in participants:
            errors.append(f"物品后果未引用参与实体：{ref}")
        if str(patch.get("location", "")) != location:
            errors.append("物品后果必须落在事件发生场景")
    normalized_scene_states: list[dict] = []
    for patch in scene_state_patches:
        if not isinstance(patch, dict):
            errors.append("局部场景后果必须是对象")
            continue
        scene_ref = resolve_target(world, str(
            patch.get("scene", patch.get("location", ""))))
        if scene_ref != f"scene:{location}":
            errors.append("局部场景后果必须落在事件发生场景")
            continue
        if scene_ref not in participants:
            errors.append(f"局部场景后果未引用参与实体：{scene_ref}")
        normalized_patch = dict(patch)
        normalized_patch["scene"] = scene_ref[6:]
        normalized_patch.pop("location", None)
        normalized_scene_states.append(normalized_patch)
    for patch in actor_patches:
        if not isinstance(patch, dict):
            errors.append("行动者后果必须是对象")
            continue
        resolved = resolve_target(world, str(patch.get("target", "")))
        if resolved is None or (resolved != "player"
                                and not resolved.startswith("npc:")):
            errors.append(f"行动者后果引用无效：{patch.get('target', '')}")
            continue
        if resolved not in participants:
            errors.append(f"行动者后果未引用参与实体：{resolved}")
        if not isinstance(patch.get("can_act"), bool):
            errors.append("行动者后果的 can_act 必须是布尔值")
        condition = str(patch.get("condition", "")).strip()
        if not condition or len(condition) > 20:
            errors.append("行动者后果必须给出不超过 20 字的当前状态")
        if resolved.startswith("npc:"):
            npc = world.npcs[resolved[4:]]
            if bool(patch.get("can_act")) == npc.state.can_act:
                errors.append(
                    "NPC 行动资格未变化；持续后果请走 state_fact_patches，"
                    "进行中的活动请走 action 或 intent")

    normalized_state_facts: list[dict] = []
    for patch in state_fact_patches:
        if not isinstance(patch, dict):
            errors.append("状态事实后果必须是对象")
            continue
        resolved = resolve_target(world, str(
            patch.get("npc", patch.get("target", ""))))
        if resolved is None or not resolved.startswith("npc:"):
            errors.append(f"状态事实后果引用无效：{patch.get('npc', '')}")
            continue
        if resolved not in participants:
            errors.append(f"状态事实后果未引用参与实体：{resolved}")
        normalized_patch = dict(patch)
        normalized_patch["npc"] = resolved[4:]
        normalized_patch.pop("target", None)
        normalized_state_facts.append(normalized_patch)

    normalized_actions: list[dict] = []
    for action in item_actions:
        if not isinstance(action, dict):
            errors.append("物品行动必须是对象")
            continue
        item_id = str(action.get("item", "")).strip()
        ref = f"item:{item_id}"
        _, item = _find_item(world, item_id)
        if ref not in participants:
            errors.append(f"物品行动未引用参与实体：{ref}")
        if item is None:
            errors.append(f"物品行动引用不存在的物品：{item_id}")
            continue
        if isinstance(item.get("action"), dict) and item["action"].get("text"):
            errors.append(f"物品 {item_id} 已有进行中的行动")
        text = str(action.get("text", "")).strip()
        try:
            days = float(action.get("days", 0.0) or 0.0)
        except (TypeError, ValueError):
            days = 0.0
        if not text or len(text) > 120:
            errors.append("物品行动必须给出不超过 120 字的内容")
        if days <= 0.0 or days > 30.0:
            errors.append("物品行动的 days 必须在 0 到 30 天之间")
        raw_targets = action.get("targets", [])
        raw_requires = action.get("requires", [])
        if not isinstance(raw_targets, list) or not isinstance(raw_requires, list):
            errors.append("物品行动的 targets/requires 必须是引用列表")
            raw_targets, raw_requires = [], []
        if len(raw_targets) > 3 or len(raw_requires) > 3:
            errors.append("物品行动的 targets/requires 各自最多 3 条")
        targets = canonical_targets(world, raw_targets)
        requires = canonical_targets(world, raw_requires)
        if len(targets) != len(raw_targets) or len(requires) != len(raw_requires):
            errors.append("物品行动引用了不存在的实体")
        for required in requires:
            if _entity_location(world, required) != location:
                errors.append(f"物品行动前置实体不在事件场景：{required}")
        normalized_actions.append({
            "item": item_id, "text": text, "days": days,
            "targets": targets, "requires": requires,
        })

    completes: list[str] = []
    for raw in raw_completes:
        resolved = resolve_target(world, str(raw))
        if resolved is None or not resolved.startswith("item:"):
            errors.append(f"完成引用不是现有物品：{raw}")
            continue
        if resolved not in participants:
            errors.append(f"完成的物品未列为参与实体：{resolved}")
        _, item = _find_item(world, resolved[5:])
        action = item.get("action") if item is not None else None
        if not isinstance(action, dict) or not action.get("text"):
            errors.append(f"物品 {resolved} 没有进行中的行动")
        elif float(action.get("due_clock", 0.0) or 0.0) > world.clock + 1e-9:
            errors.append(f"物品 {resolved} 的行动尚未到期")
        elif resolved not in completes:
            completes.append(resolved)

    return {
        "params": params,
        "item_patches": [dict(p) for p in item_patches
                         if isinstance(p, dict)],
        "scene_state_patches": normalized_scene_states,
        "actor_patches": [dict(p) for p in actor_patches
                          if isinstance(p, dict)],
        "state_fact_patches": normalized_state_facts,
        "item_actions": normalized_actions,
        "completes": completes,
    }, errors


def _apply_entity_event(world: World, normalized: dict) -> tuple[list[str], list[str]]:
    """在一个世界副本或真实世界上执行已经归一化的事务。"""
    params = normalized["params"]
    errors = emit(world, "world_event", params, cause="多实体事件")
    if errors:
        return [], errors
    root = world.events[-1]
    participant_npcs = {
        ref[4:] for ref in params.get("refs", [])
        if isinstance(ref, str) and ref.startswith("npc:")
        and ref[4:] in world.npcs
    }
    scene = world.scenes.get(str(params.get("location", "")))
    witness_npcs = set(participant_npcs)
    if scene is not None:
        witness_npcs.update(
            npc_id for npc_id in scene.npcs
            if npc_id in world.npcs and is_actionable(world.npcs[npc_id]))
    root.payload["consequences"] = {
        "item_patches": deepcopy(normalized["item_patches"]),
        "scene_state_patches": deepcopy(normalized["scene_state_patches"]),
        "actor_patches": deepcopy(normalized["actor_patches"]),
        "state_fact_patches": deepcopy(normalized["state_fact_patches"]),
        "item_actions": deepcopy(normalized["item_actions"]),
        "completes": list(normalized["completes"]),
    }
    cause = f"事件#{root.turn}：{root.summary[:100]}"
    summaries = [root.summary]
    # 行动资格后果先写：它必须能直接引用事务根事件，不能被一串物品子事件
    # 挤出近期因果窗口。真正提交前整段已在副本预演，所以顺序不破坏原子性。
    for patch in normalized["actor_patches"]:
        resolved = resolve_target(world, str(patch.get("target", "")))
        can_act = bool(patch["can_act"])
        condition = str(patch["condition"]).strip()
        if resolved == "player":
            before = (bool(world.player.get("can_act", True)),
                      str(world.player.get("condition", "")))
            after = (can_act, condition)
            if before == after:
                return summaries, ["玩家状态后果没有产生变化"]
            world.player["can_act"] = can_act
            world.player["condition"] = condition
            world.player["condition_cause_turn"] = root.turn
            summaries.append(f"玩家当前状态：{condition}")
            continue
        npc = world.npcs[resolved[4:]]
        before = npc.state.can_act
        state_summaries = set_actionability(
            world, npc, can_act, condition, root.summary)
        after = npc.state.can_act
        if after == before:
            return summaries, state_summaries or ["NPC 状态后果没有产生变化"]
        summaries.extend(state_summaries)
    for patch in normalized["state_fact_patches"]:
        fact_errors = apply_state_fact_patch(world, patch, cause=cause)
        if fact_errors:
            return summaries, fact_errors
        summaries.append(world.events[-1].summary)
    for patch in normalized["scene_state_patches"]:
        scene_errors = apply_scene_state_patch(world, patch, cause=cause)
        if scene_errors:
            return summaries, scene_errors
        summaries.append(world.events[-1].summary)
    for patch in normalized["item_patches"]:
        before_turn = world.turn
        patch_errors = apply_item_patch(world, patch, cause=cause)
        if patch_errors:
            return summaries, patch_errors
        item_id = str(patch.get("item", ""))
        if world.turn != before_turn:
            summaries.append(world.events[-1].summary)
        elif item_id:
            summaries.append(f"物品 {item_id} 状态未发生跃变")
    for resolved in normalized["completes"]:
        _, item = _find_item(world, resolved[5:])
        if item is None:
            return summaries, [f"完成动作时物品已不存在：{resolved}"]
        old_action = item.get("action")
        if not isinstance(old_action, dict) or not old_action.get("text"):
            return summaries, [f"物品没有可完成的行动：{resolved}"]
        item["action"] = {}
        item["last_turn"] = world.turn
        item["cause_turn"] = root.turn
        summaries.append(f"「{item.get('name', resolved)}」完成了："
                         f"{str(old_action.get('text', ''))[:60]}")
    for action in normalized["item_actions"]:
        _, item = _find_item(world, action["item"])
        if item is None:
            return summaries, [f"开始动作时物品已不存在：{action['item']}"]
        if isinstance(item.get("action"), dict) and item["action"].get("text"):
            return summaries, [f"物品已有进行中的行动：{action['item']}"]
        item["action"] = {
            "text": action["text"],
            "started_clock": world.clock,
            "due_clock": world.clock + action["days"],
            "targets": list(action["targets"]),
            "requires": list(action["requires"]),
            "source_turn": root.turn,
        }
        item["last_turn"] = world.turn
        item["cause_turn"] = root.turn
        summaries.append(f"「{item.get('name', action['item'])}」开始："
                         f"{action['text']}")
    # 这是事件本身的同期投影，不是事后流言。即使某个参与者被这次事件
    # 变成无法行动，他仍可保留自己亲历这件事的最后一条记忆。
    for npc_id in sorted(witness_npcs):
        npc = world.npcs.get(npc_id)
        if npc is None:
            continue
        prefix = "我亲历了" if npc_id in participant_npcs else "我看见"
        content = f"{prefix}：{root.summary}"
        world.remember_as(npc, content, cause=cause,
                       kind="npc_memory", importance=0.7)
        summaries.append(f"{npc.name}记住了这件事")
    return summaries, []


def commit_entity_event(world: World, proposal: dict) -> list[str]:
    """原子提交一个多实体事实及其后果；预演失败时真实世界零写入。"""
    normalized, errors = _normalize_entity_event(world, proposal)
    if errors:
        return [f"驳回多实体事件：{error}" for error in errors]
    shadow = deepcopy(world)
    _, errors = _apply_entity_event(shadow, normalized)
    if errors:
        return [f"驳回多实体事件：{error}" for error in errors]
    summaries, errors = _apply_entity_event(world, normalized)
    if errors:  # 单线程下预演与提交应完全一致；保留显式故障而不静默吞掉。
        raise RuntimeError("多实体事件预演与提交不一致：" + "；".join(errors))
    return summaries


def move_npc(world: World, npc: NPC, new_scene_id: str, cause: str) -> bool:
    """NPC 移动：同步场景成员表 + 状态 + 事件日志。

    雾中场景是真实存在的地方（地名级）——NPC 可以进去做事，
    舞台细不细化不影响角色活着。唯一要求：目的地必须已存在
    于世界图中（不存在的场景要先经「涌现新地名」创建）。
    """
    if not is_actionable(npc):
        return False
    old = npc.state.location
    if old == new_scene_id:
        return False
    if new_scene_id not in world.scenes:
        return False  # 目的地不存在：先走新地名涌现，再移动
    if world.scenes[new_scene_id].memory_only:
        return False  # 往事有坐标，不等于今天已有一条可走的路线
    if old in world.scenes and npc.id in world.scenes[old].npcs:
        world.scenes[old].npcs.remove(npc.id)
    if (new_scene_id in world.scenes
            and npc.id not in world.scenes[new_scene_id].npcs):
        world.scenes[new_scene_id].npcs.append(npc.id)
    npc.state.location = new_scene_id
    # 固化事件：NPC 移动也是带类型的参数调用，过包络线后写入日志
    emit(world, "npc_moved",
         {"npc": npc.id, "from": old, "to": new_scene_id}, cause)
    return True


def move_player(world: World, new_scene_id: str,
                cause: str = "玩家移动") -> list[str]:
    """玩家移动的唯一状态入口：行动资格、引用和留痕一起提交。"""
    if not player_is_actionable(world):
        return [f"你当前{player_condition(world)}，无法移动。"]
    if new_scene_id not in world.scenes:
        return [f"玩家移动引用不存在的场景：{new_scene_id}"]
    if world.scenes[new_scene_id].memory_only:
        return [f"「{world.scenes[new_scene_id].name}」目前只存在于往事坐标中"]
    old = str(world.player.get("location", ""))
    if old == new_scene_id:
        return []
    world.player["location"] = new_scene_id
    world.log("scene_entered", f"进入「{world.scenes[new_scene_id].name}」", cause,
              {"actor": "player", "from": old,
               "to": new_scene_id, "scene": new_scene_id})
    return [world.events[-1].summary]


def _normalize_action_destination(world: World, params: dict) -> None:
    """容错归正行动目的地：已有地点不能被误写成新地点名称。

    `place` 只服务于「location 是新 id」时的自由地名。模型偶尔会把已有
    场景 id 或场景名误放进 `place`；若不归正，行动会被当成原地行为，后续
    叙事便可能在未抵达时声称已经到了目的地。
    """
    if params.get("location"):
        return
    place = str(params.get("place", "")).strip()
    if place in world.scenes:
        params["location"] = place
        params.pop("place", None)
        return
    matches = [sid for sid, scene in world.scenes.items() if scene.name == place]
    if len(matches) == 1:
        params["location"] = matches[0]
        params.pop("place", None)


def _travel_in_progress(npc: NPC) -> bool:
    """跨场景行动是位置承诺，完成或显式中止前不能被普通计划改写。"""
    action = npc.state.action
    return bool(action.text and action.progress < 1.0
                and action.location != npc.state.location)


def _action_window_error(world: World, params: dict) -> str:
    """检查动作声明的世界钟窗口。

    动作文本可以自由描述“黄昏”“明早”等词，但只有模型同时给出绝对
    世界钟，调度层才把它当成硬时间约束。这样不会在引擎里增加题材时间
    枚举，也不会把自然语言里的时间词误解析成规则。
    """
    try:
        earliest = max(0.0, float(params.get("earliest_clock", 0.0) or 0.0))
        latest = max(0.0, float(params.get("latest_clock", 0.0) or 0.0))
    except (TypeError, ValueError):
        return "行动时间窗口不是有效的世界钟"
    if latest and earliest and latest < earliest:
        return "行动时间窗口无效：最晚时刻早于最早时刻"
    if earliest and world.clock + CLOCK_EPSILON < earliest:
        return (f"行动尚未到开始时刻（当前 {world.clock:.4f}，"
                f"最早 {earliest:.4f}）")
    if latest and world.clock - CLOCK_EPSILON > latest:
        return (f"行动已错过开始时刻（当前 {world.clock:.4f}，"
                f"最晚 {latest:.4f}）")
    return ""


def _begin_action(world: World, npc: NPC, params: dict,
                  source_turn: int) -> None:
    """把已入账的行动变成世界钟承诺；零时长行动仍是即时事实。"""
    destination = str(params.get("location") or npc.state.location)
    if "days" in params:
        days = max(0.0, float(params.get("days", 0.0) or 0.0))
    else:
        # 旧模型/旧夹具兼容：历史语义是跨场景约一天、原地即时。
        # 新提示词要求显式给 days，新产生的动作不再依赖固定推进率。
        days = 1.0 if destination != npc.state.location else 0.0
    earliest_clock = max(
        world.clock, float(params.get("earliest_clock", 0.0) or 0.0))
    latest_clock = max(0.0, float(params.get("latest_clock", 0.0) or 0.0))
    if days <= 0.0:
        npc.state.action = ActionState()
        return
    npc.state.action = ActionState(
        text=str(params.get("action", "")),
        location=destination,
        actor_id=world.actor_for_body(npc).id,
        started_clock=world.clock,
        due_clock=world.clock + days,
        earliest_clock=earliest_clock,
        latest_clock=latest_clock,
        targets=canonical_targets(world, params.get("targets", [])),
        requires=canonical_targets(world, params.get("requires", [])),
        source_turn=source_turn,
        progress=0.0,
    )


def set_intent(world: World, npc: NPC, proposal: dict | None,
               cause: str, body: NPC | None = None) -> list[str]:
    """把短期打算写进角色状态与账本；它不是对世界的任意状态补丁。"""
    old = npc.state.intent
    if proposal is None:
        text, raw_targets = "", []
        earliest_clock = latest_clock = 0.0
    elif isinstance(proposal, dict):
        text = str(proposal.get("text", "")).strip()
        raw_targets = proposal.get("targets", [])
        if not isinstance(raw_targets, list):
            return ["驳回短期打算：targets 必须是引用列表"]
        try:
            earliest_clock = max(0.0, float(
                proposal.get("earliest_clock", 0.0) or 0.0))
            latest_clock = max(0.0, float(
                proposal.get("latest_clock", 0.0) or 0.0))
        except (TypeError, ValueError):
            return ["驳回短期打算：时间窗口必须是世界钟数值"]
    else:
        return ["驳回短期打算：必须是对象或 null"]
    if not text and raw_targets:
        return ["驳回短期打算：放下打算时不能保留目标引用"]
    targets = canonical_targets(world, raw_targets)
    supplied = [str(t).strip() for t in raw_targets if isinstance(t, str) and str(t).strip()]
    if len(targets) != len(supplied) or len(supplied) != len(raw_targets):
        return ["驳回短期打算：目标引用必须是世界内真实实体"]
    if (text == old.text and targets == old.targets
            and earliest_clock == old.earliest_clock
            and latest_clock == old.latest_clock):
        return []
    if not text and not old.text:
        return []
    public_body = body or npc
    params = {"npc": public_body.id, "intent": text, "previous": old.text,
              "targets": targets, "earliest_clock": earliest_clock,
              "latest_clock": latest_clock}
    errors = emit(world, "npc_intent", params, cause=cause)
    if errors:
        return errors
    npc.state.intent = IntentState(text=text, targets=targets,
                                   earliest_clock=earliest_clock,
                                   latest_clock=latest_clock,
                                   since_turn=world.turn if text else 0)
    return [world.events[-1].summary]


def _apply_plan_intent(world: World, npc: NPC, plan: dict,
                       summaries: list[str], cause: str,
                       action_rejected: bool = False,
                       body: NPC | None = None) -> None:
    """计划里只有显式给出的 intent 才改状态；null 表示有因地放下。"""
    if "intent" not in plan or action_rejected:
        return
    summaries.extend(set_intent(world, npc, plan.get("intent"), cause,
                                body=body))


def catch_up(llm: BaseLLM, world: World, npc: NPC) -> list[str]:
    """读取补算：把 NPC 状态推进到当前回合。返回演化摘要。"""
    if not is_actionable(npc):
        return []
    actor = world.actor_for_body(npc)
    elapsed = elapsed_steps(world, npc.state.last_clock)
    if elapsed <= 0:
        return []
    data = llm.chat_json(_CATCHUP_SYSTEM, _json.dumps({
        "phase": world.phase, "day": world.day,
        "weather": weather_of(world), "elapsed_turns": elapsed,
        "npc": _npc_payload(
            world, npc, " ".join(
                [npc.state.activity, actor.state.intent.text]
                + [str(g.get("text", "")) for g in actor.goals
                   if float(g.get("progress", 0)) < 1.0])),
        "scenes": {sid: s.name for sid, s in world.scenes.items()},
    }, ensure_ascii=False))

    decay_mood(actor, elapsed)  # 缺席期间情绪随时间回落
    action_in_progress = bool(npc.state.action.text)
    new_loc = str(data.get("location", npc.state.location))
    # 进行中的动作是唯一持有移动承诺的状态。补算可以改变活动、情绪
    # 和记忆，但不能让普通作息裁决把在途角色瞬移到另一场景。
    if action_in_progress:
        new_loc = npc.state.location
        moved = False
    else:
        moved = bool(data.get("moved", False)) or new_loc != npc.state.location
    npc.state.activity = str(data.get("activity", npc.state.activity))
    actor.state.mood = mood_label(actor, str(data.get("mood",
                                                    actor.state.mood)))
    mark_npc(world, npc)
    if moved:
        move_npc(world, npc, new_loc, cause="读取补算")
    memory = str(data.get("memory", "")).strip()
    # 机械闸：短缺席且没移动 → 不记（防补算记忆刷屏）
    if memory and not (moved or elapsed >= 24):
        memory = ""
    if memory:
        world.remember_as(npc, memory, cause="读取补算", kind="npc_catchup")
    summaries = [f"{npc.name}：{npc.state.activity}（{actor.state.mood}）"]
    # 主动动作随缺席时间推进（她不在你眼前时也在做那件事）
    summaries.extend(advance_action(llm, world, npc, elapsed))
    return summaries


def catch_up_scene(llm: BaseLLM, world: World, scene_id: str) -> list[str]:
    """场景内全体 NPC 补算（玩家进入/观察场景时调用）。"""
    scene = world.scenes.get(scene_id)
    if scene is None:
        return []
    summaries: list[str] = []
    max_elapsed = 0
    for npc_id in list(scene.npcs):  # 快照遍历，补算中可能移走
        npc = world.npcs.get(npc_id)
        if npc is not None and is_actionable(npc):
            max_elapsed = max(max_elapsed,
                              elapsed_steps(world, npc.state.last_clock))
            summaries.extend(catch_up(llm, world, npc))
    return summaries


def heartbeat(llm: BaseLLM, world: World, npc: NPC,
              skip_interaction: bool = False, cause: str = "NPC 心跳") -> list[str]:
    """心跳演化：NPC 过它此刻的生活，可能与其他 NPC 搭话。"""
    if not is_actionable(npc):
        return []
    scene = world.scenes.get(npc.state.location)
    present = [nid for nid in (scene.npcs if scene else [])
               if nid != npc.id and nid in world.npcs
               and is_actionable(world.npcs[nid])]
    # 冷却按说话方向记：A 对 B 说过话，只冷却 A 对 B，不冷却 B 对 A
    my_keys = [f"{npc.id}->{o}" for o in present]
    can_interact = (not skip_interaction and bool(present)
                    and all(cooldown_ready(world, k, INTERACT_COOLDOWN)
                            for k in my_keys))

    actor = world.actor_for_body(npc)
    data = llm.chat_json(_HEARTBEAT_SYSTEM, _json.dumps({
        "phase": world.phase, "day": world.day,
        "weather": weather_of(world),
        "npc": _npc_payload(
            world, npc, " ".join(
                [npc.state.activity, actor.state.intent.text]
                    + [str(g.get("text", "")) for g in actor.goals
                       if float(g.get("progress", 0)) < 1.0])),
        "present": present,
        "can_interact": can_interact,
    }, ensure_ascii=False))

    elapsed = elapsed_steps(world, npc.state.last_clock)  # 心跳前的缺席时长
    decay_mood(actor, elapsed)
    npc.state.activity = str(data.get("activity", npc.state.activity))
    actor.state.mood = mood_label(actor, str(data.get("mood",
                                                    actor.state.mood)))
    mark_npc(world, npc)
    new_loc = str(data.get("location", npc.state.location))
    summaries: list[str] = []
    # 在途动作未完成前，普通心跳不能改写它的当前位置。
    if new_loc != npc.state.location and not npc.state.action.text:
        if move_npc(world, npc, new_loc, cause=cause):
            summaries.append(f"{npc.name} 去了「"
                             f"{world.scenes.get(new_loc, '').name
                             if new_loc in world.scenes else new_loc}」")

    inter = data.get("interaction")
    if isinstance(inter, dict) and can_interact:
        target = world.npcs.get(str(inter.get("with", "")))
        if target is not None and is_actionable(target) and target.id in present:
            line = str(inter.get("line", "……"))
            pair_key = f"{npc.id}->{target.id}"
            mark_social(world, pair_key)
            world.remember_as(npc, f"对 {target.name} 说：「{line}」",
                           cause=cause, kind="npc_memory")
            # 听者也留下自己的版本——对话是双方的事实（地基第 4 条）
            world.remember_as(target, f"{npc.name} 对我说：「{line}」",
                           cause=cause, kind="npc_memory")
            # 固化事件：搭话也是带类型的参数调用（带地点：同场景的人能看见）
            emit(world, "npc_interaction",
                 {"npc": npc.id, "target": target.id, "line": line,
                  "location": npc.state.location},
                 cause=cause)
            summaries.append(f"{npc.name} 对 {target.name}：{line}")
            # NPC↔NPC 关系账本：搭话让两人的链接向正方向移动一点
            delta = max(-0.2, min(0.2,
                                  float(inter.get("link_delta", 0.1))))
            actor.links[target.id] = max(-1.0, min(
                1.0, float(actor.links.get(target.id, 0.0)) + delta))
            target_actor = world.actor_for_body(target)
            target_actor.links[npc.id] = max(-1.0, min(
                1.0, float(target_actor.links.get(npc.id, 0.0)) + delta))
    # 先结算已经到点的承诺，再判断角色此刻是否有余力开始下一件事。
    summaries.extend(advance_action(llm, world, npc, elapsed))
    # 驱动力：条件成熟时主动触发事件（角色也是观察者）
    summaries.extend(propose_proactive(llm, world, npc))
    # 玩家在场时，角色可能主动开口（世界状态变化驱动）
    summaries.extend(propose_opener(llm, world, npc))
    return summaries


def _apply_goal_updates(world: World, npc: NPC, updates: dict,
                        summaries: list[str],
                        fallback_cause: str = "",
                        body: NPC | None = None) -> None:
    """目标进度更新：有因才推进、受阻则冻结、引用验真。

    - 进度提升必须带 because（引用真实发生的事），否则驳回。
    - 机械兜底：同一次裁决若带动作/事件，且模型没写 because，
      取动作文本当因——因来自真实发生的事，只是引擎代抄，不造假。
    - blocked_by 是引用（item:/scene:/npc: 或裸名）：被卡住时进度冻结，
      直到阻碍解除（blocked_by 留空 + because 说明）。
    - 推进的原因写入角色记忆（引用即留痕）。
    """
    memory_body = body or npc
    for gid, update in (updates or {}).items():
        goal = next((g for g in npc.goals if g.get("id") == gid), None)
        if goal is None:
            continue
        if isinstance(update, dict):
            new_p = float(update.get("progress",
                                     goal.get("progress", 0)))
            because = str(update.get("because", "")).strip()
            blocked = str(update.get("blocked_by", "")).strip()
            blocked_note = str(update.get("blocked_note", "")).strip()[:80]
        else:  # 宽容：裸数值旧格式（无因 → 提升将被驳回）
            new_p = float(update)
            because, blocked, blocked_note = "", "", ""
        new_p = max(0.0, min(1.0, new_p))
        old_p = float(goal.get("progress", 0))
        if blocked:
            r = resolve_target(world, blocked)
            if r is None:
                summaries.append("驳回受阻：引用不存在")
                continue
            goal["blocked_by"] = r
            goal["blocked_note"] = blocked_note
            summaries.append(
                f"{npc.name} 的目标「{str(goal.get('text', ''))[:20]}」"
                f"被卡住了")
            continue  # 冻结：受阻时不推进
        if new_p > old_p and not because:
            # 机械兜底：因来自同一次裁决里真实发生的动作
            if fallback_cause:
                because = fallback_cause[:80]
        if new_p > old_p and not because:
            summaries.append("驳回推进：进度提升必须有因（because）")
            world.social[f"reject|{npc.id}"] = \
                "上次的目标进度提升被驳回：提升必须给 because（引用真实发生的事）"
            continue
        goal["progress"] = new_p
        world.social.pop(f"reject|{npc.id}", None)  # 成功：清空驳回回显
        if because:
            goal["blocked_by"] = ""
            goal["blocked_note"] = ""
            if new_p > old_p:
                world.remember_as(
                    memory_body,
                    f"目标「{str(goal.get('text', ''))[:20]}」推进了："
                    f"{because[:60]}",
                    cause="目标推进", kind="npc_memory", importance=0.5)


def _npc_visible(world: World, npc: NPC, e) -> bool:
    """NPC 的知识边界（地基第 4 条）：事件发生在自己的场景，或本人参与。

    其余的事他只能靠记忆里的流言（听说）知道——载荷不额外喂。
    可见情形：地点命中（location / to / from / scene——别人从自己场景
    离开也算发生在眼前）；本人参与（npc / target / collision 的 a / b）。
    """
    p = (e.payload or {}).get("event_params", e.payload or {})
    # 逐个键检查（不能用 or 链——to 有值时 from 会被吞掉）
    if e.kind == "npc_acted" and p.get("origin"):
        # location 是行动目的地；开始行动时只被出发现场看见。
        locs = {p.get("origin")}
    else:
        locs = {p.get("location"), p.get("to"), p.get("from"),
                p.get("scene")}
    if npc.state.location in locs:
        return True
    if p.get("npc") == npc.id or p.get("target") == npc.id:
        return True
    if p.get("a") == npc.id or p.get("b") == npc.id:
        return True
    if f"npc:{npc.id}" in p.get("refs", []):
        return True
    return False


def _is_player_event(e) -> bool:
    """玩家的行动只凭稳定 actor 标记或既有玩家事件类型识别。"""
    p = (e.payload or {}).get("event_params", e.payload or {})
    return (p.get("actor") == "player"
            or e.kind in ("player_said", "player_acted", "action_refused"))


def _player_traces(world: World, npc: NPC) -> list[dict]:
    """只给 NPC 自己看得见的玩家行动痕迹，不提供实时玩家坐标。"""
    traces: list[dict] = []
    for e in world.events[-8:]:
        if not _is_player_event(e) or not _npc_visible(world, npc, e):
            continue
        p = (e.payload or {}).get("event_params", e.payload or {})
        traces.append({
            "kind": e.kind,
            "summary": e.summary,
            "from": p.get("from", ""),
            "to": p.get("to", p.get("scene", "")),
            "location": p.get("location", ""),
        })
    return traces[-5:]


def start_player_interaction(world: World, npc: NPC, line: str,
                             cause: str) -> list[str]:
    """把 NPC 对玩家的开口写成与 NPC 间搭话同一种账本事实。

    `player` 是稳定行动者 id，不是感知通道。只有同场、冷却结束且
    内容非空时才成立；pending_opener 只是该事件给玩家界面的投影。
    """
    if not is_actionable(npc) or \
            npc.state.location != world.player.get("location", ""):
        return []
    text = str(line).strip()
    key = f"{npc.id}->player"
    if not text or not cooldown_ready(world, key, OPEN_COOLDOWN):
        return []
    mark_social(world, key)
    npc.state.pending_opener = text
    errors = emit(world, "npc_interaction",
                  {"npc": npc.id, "target": "player", "line": text,
                   "location": npc.state.location}, cause=cause)
    if errors:
        npc.state.pending_opener = ""
        return []
    world.remember_as(npc, f"我对你说：「{text}」", cause=cause,
                   kind="npc_memory", importance=0.5)
    return [f"{npc.name} 叫住了你：「{text}」"]


def _targets_now(world: World, npc: NPC) -> list[dict]:
    """目标的眼睛：活跃目标引用的实体快照（裁决者看得见自己盯的东西）。"""
    actor = world.actor_for_body(npc)
    open_goals = [g for g in actor.goals if float(g.get("progress", 0)) < 1.0]
    out = []
    for ref in {r for g in open_goals
                for t in g.get("targets", []) if isinstance(t, str)
                for r in [resolve_target(world, t)] if r}:
        snap = target_snapshot(world, ref)
        if snap:
            out.append({"ref": ref, "snapshot": snap})
    return out


def propose_proactive(llm: BaseLLM, world: World, npc: NPC) -> list[str]:
    """驱动力裁决：条件成熟时 NPC 主动触发事件（走固化事件通道）。

    - 目标被记忆/流言激活 + 情绪/关系达标才动（条件驱动，不随机）。
    - 只评估未完成的目标（检索式评估）；已完成的目标留在库里作传记。
    - 达成 = 转化三件套：goal_completed 事件 + 信念蒸馏 + 新目标涌现。
    - npc_acted 带 location 时进入去往该地的在途动作；完成时才抵达。
    """
    if not is_actionable(npc):
        return []
    if npc.state.action.text:
        return []
    actor = world.actor_for_body(npc)
    open_goals = [g for g in actor.goals if float(g.get("progress", 0)) < 1.0]
    if not open_goals:
        return []
    query = " ".join(
        [actor.state.intent.text]
        + [str(goal.get("text", "")) for goal in open_goals]
        + [event.summary for event in world.events[-5:]
           if _npc_visible(world, npc, event)])
    lived = _lived_payload(world, actor, query, limit=10)
    data = llm.chat_json(_NPCGOAL_SYSTEM, _json.dumps({
        "npc": {
            "id": npc.id, "name": npc.name,
            "actor": {"id": actor.id, "name": actor.name},
            "body": {"id": npc.id, "name": npc.name},
            "persona_origin": actor.persona,
            "trait_origins": actor.traits,
            "goals": open_goals,
            "relationship": actor.relationship,
            "mood_value": actor.state.mood_value,
            "state": {"location": npc.state.location,
                       "intent": {"text": actor.state.intent.text,
                                  "targets": list(actor.state.intent.targets),
                                  "earliest_clock": actor.state.intent.earliest_clock,
                                  "latest_clock": actor.state.intent.latest_clock},
                      "facts": [dict(f) for f in actor.state.facts[-12:]]},
        },
        "atmosphere": world.law_profile.atmosphere,
        "time": {"clock": world.clock, "day": world.day,
                 "phase": world.phase, "phase_name": PHASE_NAMES[world.phase],
                 "phase_length": 0.25},
        "experience_count": len(actor.memories),
        "lived_experiences": lived,
        "memories": [row["content"] for row in lived],
        "beliefs": actor.beliefs[-5:],
        "memory_gaps": memory_gaps_payload(actor),
        "player_traces": _player_traces(world, npc),
        # 知识边界：只看得见自己场景里发生的、或自己参与的事件；
        # 其余的事他只能靠记忆里的流言知道（地基第 4 条）
        "recent_events": [
            e.summary for e in world.events[-8:]
            if _npc_visible(world, npc, e)
        ][-5:],
        # 法则与世界档案：目标裁决者读得到世界的规则（不是脑补）
        "laws": [f"{l.trigger} → {l.effect}"
                 for l in world.law_profile.laws],
        "facts": list(world.facts),
        # 上次被驳回的原因：模型看得见自己的失误，下一轮才知道改
        "rejection_note": world.social.get(f"reject|{npc.id}", ""),
        # 目标的眼睛：每个活跃目标引用的实体的当前快照——
        # 她盯着的那个东西现在是什么样，她裁决时看得见。
        "targets_now": _targets_now(world, npc),
        # 世界布局：目标裁决者有权知道城市的地形（导航知识不是脑补）
        "scenes": {sid: s.name for sid, s in world.scenes.items()},
    }, ensure_ascii=False))
    summaries: list[str] = []
    last_act = ""  # 本轮裁决真实发生的动作摘要（兜底因的来源）
    action_rejected = False
    action_deferred = False
    for item in data.get("events", []):
        if not isinstance(item, dict) or not item.get("type"):
            continue
        etype = str(item["type"])
        params = dict(item.get("params") or {})
        params.setdefault("npc", npc.id)
        if etype == "npc_intent":
            summaries.append("驳回主动事件：短期打算只能写在 intent 字段")
            continue
        if etype == "npc_acted":
            _normalize_action_destination(world, params)
            if _travel_in_progress(npc):
                summaries.append(f"{npc.name}仍在前往目的地，新动作暂缓")
                return summaries
            timing_error = _action_window_error(world, params)
            if timing_error:
                summaries.append(f"{npc.name}的行动暂未提交：{timing_error}")
                action_rejected = action_rejected or "错过" in timing_error
                action_deferred = action_deferred or "尚未到" in timing_error
                continue
        loc = str(params.get("location") or "")
        place = str(params.get("place", "")).strip()
        if (etype == "npc_acted" and loc
                and loc != npc.state.location
                and world.actor_for_body(npc).id != npc.id
                and params.get("travel") is not True):
            summaries.append(
                f"驳回主动事件：{npc.name}借身跨场景行动必须明确 travel=true")
            action_rejected = True
            continue
        if etype == "npc_acted" and loc and loc not in world.scenes and place:
            # 新地名涌现：NPC 要去的地方不存在 → 涌现成雾中地名
            summaries.extend(emerge_place(world, loc, place,
                                          npc.state.location))
        errors = emit(world, etype, params, cause="角色驱动力")
        if errors:
            summaries.extend(errors)
            action_rejected = action_rejected or etype == "npc_acted"
            continue
        act_summary = world.events[-1].summary  # 先快照，后面移动会写新事件
        act_turn = world.events[-1].turn
        summaries.append(act_summary)
        if etype == "npc_acted":
            last_act = act_summary  # 兜底因：动作就是这件事发生的因
        # 动作进入「进行中」状态：可被看见、追上、聊起
        if etype == "npc_acted":
            summaries.extend(abort_action(world, npc, cause="角色驱动力"))
            _begin_action(world, npc, params, source_turn=act_turn)
        # 行动写入自己的记忆（角色也是观察者，观察自己）
        world.remember_as(npc, act_summary,
                       cause="角色驱动力", kind="npc_memory", importance=0.7)
    _apply_plan_intent(world, actor, data, summaries, cause="角色驱动力",
                       action_rejected=action_rejected, body=npc)
    # 同一份计划里的行动被驳回，就不能借文本里的 because 推进目标或长出
    # 下一目标；否则会出现「人没做到、进度却涨了」的无因状态改变。
    if action_rejected or action_deferred:
        return summaries
    # 目标进度更新（有因才推进、受阻冻结、引用验真）
    # 兜底因：本轮真发生过的动作摘要——因来自事实，引擎只是代抄
    _apply_goal_updates(world, actor, data.get("goal_updates", {}), summaries,
                        fallback_cause=last_act, body=npc)
    # 达成转化：事件 + 信念 + 新目标（驱动不消失，目标链生长）
    for goal in actor.goals:
        if float(goal.get("progress", 0)) >= 1.0 and not goal.get("done"):
            goal["done"] = True
            emit(world, "goal_completed",
                 {"npc": npc.id, "goal": str(goal.get("text", ""))},
                 cause="角色驱动力")
            summaries.append(world.events[-1].summary)
            push_world_mood(world, 0.1, "有人达成了目标")  # 好事让世界回暖一点
            if len(actor.beliefs) < 10:
                actor.beliefs.append(f"曾完成：{goal.get('text', '')}")
    for new_goal in data.get("new_goals", []):
        if not isinstance(new_goal, dict) or not new_goal.get("text"):
            continue
        gid = str(new_goal.get("id", f"g-{len(actor.goals) + 1}"))
        used = {g.get("id") for g in actor.goals}
        while gid in used:
            gid += "-x"
        actor.goals.append({
            "id": gid,
            "text": str(new_goal["text"]),
            "progress": max(0.0, min(1.0, float(new_goal.get("progress", 0)))),
            "targets": canonical_targets(world, new_goal.get("targets")),
        })
        # 目标的出生也入账（覆写留痕：出生与死亡都要有痕）
        emit(world, "goal_emerged",
             {"npc": npc.id, "goal": str(new_goal["text"])},
             cause="角色驱动力")
        summaries.append(world.events[-1].summary)
    return summaries


OPEN_COOLDOWN = 24  # 同一 NPC 主动开口的最小间隔（回合）
LEGACY_ACTION_RATE = 1.0 / 24  # 只用于把旧 progress 存档迁移到绝对钟点


def active_social(world: World) -> dict:
    """载荷里的 social 快照：只带活跃 NPC 相关的冷却键 + 全局节奏键。

    雾中人的互动/开口/主动冷却键不进载荷（也不该存着——退雾时已清）。
    全局键（new-npc / return）是节奏阀门，永远保留。
    """
    active_ids = {n.id for n in world.npcs.values() if is_actionable(n)}
    out: dict = {}
    for k, v in world.social.items():
        if k.startswith("coll|") or k.startswith("seen|"):
            continue  # 碰撞边沿检测 / 阅读进度：引擎与玩家内部状态，不进载荷
        if "->" not in k:
            out[k] = v  # 全局节奏键
            continue
        left, _, right = k.partition("->")
        if left in active_ids and (right in active_ids
                                   or right in ("player", "proactive")):
            out[k] = v
    return out


def abort_action(world: World, npc: NPC, cause: str) -> list[str]:
    """动作被新动作顶替前补痕：未完成的旧动作有因才消失。"""
    old = npc.state.action
    if not old.text or old.progress >= 1.0:
        return []
    actor = (world.npcs.get(old.actor_id)
             if old.actor_id else world.actor_for_body(npc))
    if actor is None:
        actor = world.actor_for_body(npc)
    world.remember(actor, f"我搁下了这件事：{old.text[:40]}",
                   cause=cause, kind="npc_memory", importance=0.5,
                   body=npc, started_clock=old.started_clock,
                   ended_clock=world.clock, record_gap=False)
    emit(world, "action_aborted",
         {"npc": npc.id, "body": npc.id, "actor": actor.id,
          "action": old.text[:200]}, cause=cause)
    return [world.events[-1].summary]


def _event_mentions_ref(event, resolved: str) -> bool:
    p = (event.payload or {}).get("event_params", event.payload or {})
    if resolved == "player":
        return (p.get("actor") == "player" or p.get("target") == "player"
                or "player" in p.get("refs", []))
    kind, _, ref = resolved.partition(":")
    if kind == "item":
        return p.get("item") == ref or resolved in p.get("refs", [])
    if kind == "npc":
        return (ref in (p.get("npc"), p.get("target"), p.get("a"), p.get("b"))
                or resolved in p.get("refs", []))
    if kind == "scene":
        return (ref in (p.get("location"), p.get("to"), p.get("from"),
                        p.get("scene")) or resolved in p.get("refs", []))
    return False


def _action_precondition_errors(world: World, action: ActionState) -> tuple[list[str], str]:
    """机械复查存在性与显式同场要求；世界语义仍由模型判断。"""
    errors: list[str] = []
    failed_refs: list[str] = []
    if action.location not in world.scenes:
        errors.append(f"行动地点 {action.location} 已不存在")
        failed_refs.append(f"scene:{action.location}")
    for ref in action.targets:
        if resolve_target(world, ref) is None:
            errors.append(f"行动目标 {ref} 已不存在")
            failed_refs.append(ref)
    for ref in action.requires:
        resolved = resolve_target(world, ref)
        if resolved is None:
            errors.append(f"行动前置实体 {ref} 已不存在")
            failed_refs.append(ref)
            continue
        actual = _entity_location(world, resolved)
        if actual != action.location:
            errors.append(f"行动前置实体 {resolved} 已离开行动地点")
            failed_refs.append(resolved)
            continue
        if resolved.startswith("npc:"):
            required_npc = world.npcs.get(resolved[4:])
            if required_npc is None or not is_actionable(required_npc):
                errors.append(f"行动前置角色 {resolved} 当前无法参与")
                failed_refs.append(resolved)
    cause = "动作前置条件复查"
    for event in reversed(world.events):
        if any(_event_mentions_ref(event, ref) for ref in failed_refs):
            cause = f"事件#{event.turn}：{event.summary[:100]}"
            break
    return errors, cause


def advance_action(llm: BaseLLM, world: World, npc: NPC,
                   turns: float) -> list[str]:
    """主动动作到达承诺钟点后才裁决结局；不再按固定速率累加。"""
    if not is_actionable(npc):
        return []
    action = npc.state.action
    if not action.text or turns <= 0:
        return []
    actor = (world.npcs.get(action.actor_id)
             if action.actor_id else world.actor_for_body(npc))
    if actor is None:
        actor = world.actor_for_body(npc)
    if action.due_clock <= action.started_clock:
        # 旧档只有 progress。把本次已经流逝的 turns 计入后，固定一次
        # 绝对到期钟点；此后完全跟 world.clock 比较。
        remaining_turns = max(
            0.0, (1.0 - action.progress) / LEGACY_ACTION_RATE - turns)
        action.due_clock = world.clock + remaining_turns * world.heartbeat
        action.started_clock = action.due_clock - 1.0
    if (action.earliest_clock
            and world.clock + CLOCK_EPSILON < action.earliest_clock):
        action.progress = 0.0
        return []
    span = action.due_clock - action.started_clock
    if span > 0:
        action.progress = max(0.0, min(
            1.0, (world.clock - action.started_clock) / span))
    if world.clock + 1e-9 < action.due_clock:
        return []
    precondition_errors, precondition_cause = _action_precondition_errors(
        world, action)
    if precondition_errors:
        action.progress = min(action.progress, 0.99)
        summaries = abort_action(world, npc, cause=precondition_cause)
        npc.state.action = ActionState()
        return [f"{npc.name}未能完成「{action.text[:30]}」：{error}"
                for error in precondition_errors] + summaries
    action.progress = 1.0
    lived = _lived_payload(world, actor, action.text, limit=10)
    data = llm.chat_json(_ACTIONRESOLVE_SYSTEM, _json.dumps({
        "npc": {"id": npc.id, "name": npc.name,
                "actor": {"id": actor.id, "name": actor.name},
                "body": {"id": npc.id, "name": npc.name},
                "persona_origin": actor.persona,
                "trait_origins": actor.traits,
                "experience_count": len(actor.memories),
                "lived_experiences": lived,
                "beliefs": actor.beliefs[-5:],
                "memory_gaps": memory_gaps_payload(actor)},
        "action": action.text,
        "location": action.location,
        "timing": {"started_clock": action.started_clock,
                   "due_clock": action.due_clock,
                   "source_turn": action.source_turn},
        "action_targets": list(action.targets),
        "action_requires": list(action.requires),
        "arrival": {
            "location": action.location,
            "player_present": world.player.get("location", "")
            == action.location,
        },
        "recent_events": [
            e.summary for e in world.events[-8:]
            if _npc_visible(world, npc, e)
        ][-5:],
        # 结局要能落到状态：把现场的物品表给模型——
        # patch 只能引用这些真实存在的 id（不然模型只能靠猜）。
        "scene_items": [
            {"id": i.get("id"), "name": i.get("name"),
             "note": i.get("note", "")}
            for i in active_items(scene)
        ] if (scene := world.scenes.get(action.location)) else [],
        # id 是世界级引用，不是场景内序号。新增物品不得与远处物品撞 id。
        "used_item_ids": [
            str(item.get("id", ""))
            for scene_now in world.scenes.values()
            for item in scene_now.items if item.get("id")
        ],
        # 目标的眼睛：动作要动的东西，现在是什么样。
        "targets_now": _targets_now(world, npc),
        # 法则与世界档案：结局要符合世界的规则（唱歌招海浪不是脑补）
        "laws": [f"{l.trigger} → {l.effect}"
                 for l in world.law_profile.laws],
        "facts": list(world.facts),
    }, ensure_ascii=False))
    outcome = str(data.get("outcome", "")).strip()
    if not outcome:
        outcome = "这件事做完了，心里的一块石头落了地。"
    done_params = {"npc": npc.id, "body": npc.id, "actor": actor.id,
                   "action": action.text, "outcome": outcome,
                   "location": action.location}
    # 先检查结局，后改变位置。否则一个被驳回的「我见到了某人」会先把
    # 人移动过去，再在账本上留下只有半截的抵达。
    errors = (validate_event("action_done", done_params)
              + validate_refs(world, "action_done", done_params,
                              cause="角色驱动力"))
    if errors:
        action.progress = min(action.progress, 0.99)
        return [f"驳回动作结局：{error}" for error in errors]
    # 去某处不是瞬移：动作走完才抵达目的地。若玩家已经离开，结局裁决
    # 会基于抵达时的世界状态自然得到扑空、等待或继续寻找，而非追踪坐标。
    if (action.location in world.scenes
            and action.location != npc.state.location):
        move_npc(world, npc, action.location, cause="角色抵达")
    errors = emit(world, "action_done", done_params, cause="角色驱动力")
    if errors:
        action.progress = min(action.progress, 0.99)
        return [f"驳回动作结局：{error}" for error in errors]
    done_summary = world.events[-1].summary
    world.remember(actor, f"我做完了这件事：{outcome}",
                   cause="角色驱动力", kind="npc_memory", importance=0.7,
                   body=npc, started_clock=action.started_clock,
                   ended_clock=world.clock, record_gap=False)
    npc.state.action = ActionState()  # 动作结束，清空
    summaries = [done_summary]
    # 结局补丁：叙事说发生的事，patch 让它真发生——
    # 动作的后果是一等状态改写（取走/拆开/放下），有因、验引用。
    patch = data.get("patch")
    if isinstance(patch, dict) and patch.get("op"):
        # 行动结局中的新物品必须交代归属：它要么被谁带着，要么明确
        # 留在现场。世界脉冲的环境物品仍可省略归属，不受这条限制。
        if patch.get("op") == "add" and "held_by" not in patch:
            errs = ["新物品必须明确 held_by（持有者或空）"]
        else:
            errs = apply_item_patch(
                world, patch, cause=f"动作结局：{action.text[:30]}")
        if errs:
            summaries.extend(f"驳回结局补丁：{e}" for e in errs)
        else:
            summaries.append(world.events[-1].summary)
    return summaries


def grounding_warnings(world: World, slice_text: str, text: str) -> list[str]:
    """一致性软校验：文本中出现的世界实体名，若不在检索切片里，提示。

    这是「没检索到的就不许说」的软版本——不拦截（叙事需要自由），
    但把脑补迹象浮出来，供测试与审计。
    """
    warnings: list[str] = []
    known = ([n.name for n in world.npcs.values()]
             + [s.name for s in world.scenes.values()])
    for name in known:
        if len(name) >= 2 and name in text and name not in slice_text:
            warnings.append(f"提及了切片之外的「{name}」")
    return warnings


def propose_opener(llm: BaseLLM, world: World, npc: NPC) -> list[str]:
    """主动开口裁决：世界状态变化驱动 NPC 向玩家说话。

    - 只在玩家所在场景生效；冷却期内不开口。
    - 开口走 npc_interaction（target=player）；pending_opener 仅供界面投影。
    """
    if not is_actionable(npc) or \
            npc.state.location != world.player.get("location", ""):
        return []
    actor = world.actor_for_body(npc)
    key = f"{npc.id}->player"
    if not cooldown_ready(world, key, OPEN_COOLDOWN):
        return []
    query = " ".join(
        [actor.state.intent.text]
        + [str(goal.get("text", "")) for goal in actor.goals
           if float(goal.get("progress", 0)) < 1.0]
        + [event.summary for event in world.events[-5:]
           if _npc_visible(world, npc, event)])
    lived = _lived_payload(world, actor, query, limit=8)
    data = llm.chat_json(_NPCINITIATE_SYSTEM, _json.dumps({
        "npc": {
            "id": npc.id, "name": npc.name,
            "actor": {"id": actor.id, "name": actor.name},
            "body": {"id": npc.id, "name": npc.name},
            "persona_origin": actor.persona,
            "trait_origins": actor.traits,
            "relationship": actor.relationship,
            "mood_value": actor.state.mood_value,
            "experience_count": len(actor.memories),
            "lived_experiences": lived,
            "beliefs": actor.beliefs[-5:],
            "memory_gaps": memory_gaps_payload(actor),
        },
        "player": {
            "id": "player",
            "profile": dict(world.player.get("profile", {})),
            "location": world.player.get("location", ""),
            "recent_visible_actions": [
                e.summary for e in world.events[-8:]
                if _npc_visible(world, npc, e)
                and e.kind in ("scene_entered", "player_said", "player_acted")
            ],
        },
        "atmosphere": world.law_profile.atmosphere,
        "law_recent": any(e.kind == "law_changed"
                          for e in world.events[-8:]),
        "has_rumor": any(("听说" in m.content) or ("都说" in m.content)
                         for m in [m for m in actor.memories
                                   if m.accessible][-5:]),
        "recent_events": [
            e.summary for e in world.events[-8:]
            if _npc_visible(world, npc, e)
        ][-5:],
    }, ensure_ascii=False))
    if not data.get("open"):
        return []
    line = str(data.get("line", "")).strip()
    if not line:
        return []
    return start_player_interaction(world, npc, line, cause="角色主动开口")


def heartbeat_scene(llm: BaseLLM, world: World, scene_id: str,
                    exclude: str | None = None) -> list[str]:
    """玩家所在场景的全体 NPC 心跳（观察即动作）。"""
    scene = world.scenes.get(scene_id)
    if scene is None:
        return []
    summaries: list[str] = []
    for npc_id in list(scene.npcs):
        if npc_id == exclude:
            continue
        npc = world.npcs.get(npc_id)
        if npc is not None:
            summaries.extend(heartbeat(llm, world, npc))
    return summaries


def on_law_change(world: World) -> None:
    """天变波及所有 NPC：为后续结合个人经历的裁决写入共同冲击。"""
    affected: set[str] = set()
    for npc in world.npcs.values():
        if not is_actionable(npc):
            continue
        actor = world.actor_for_body(npc)
        if actor.id in affected:
            continue
        affected.add(actor.id)
        push_mood(actor, -0.3, "天变")
        actor.state.mood = mood_label(actor, "天变后的不安")
    push_world_mood(world, -0.2, "天变")  # 世界氛围也随之压低
    world.log("world_react", "天变波及全城：所有 NPC 情绪不安", "天变",
              {"npcs": len(world.npcs)})


PULSE_INTERVAL = 6   # 脉冲扫描间隔（回合）
PULSE_BUDGET = 3     # 每次脉冲最多演化多少个 NPC
WAKEUP_BUDGET = 1    # 事件驱动的额外优先席位；不扩大单次载荷/调用成本
PULSE_MAX_INTERVAL = 48  # 心跳间隔封顶（两天）
LIFE_INTERVAL = 48   # 雾中有目标角色的生活节律：即使远方也最多两天被裁决一次
POP_FLOOD_GUARD = 12  # 人口防洪水阀：短窗口内连续来人不放行（纪律，非节律）


def scene_distances(world: World, origin: str) -> dict[str, int]:
    """BFS：origin 场景到各场景的跳数（沿 exits 连接）。"""
    dist = {origin: 0}
    frontier = [origin]
    while frontier:
        cur = frontier.pop(0)
        scene = world.scenes.get(cur)
        if scene is None:
            continue
        for nxt in scene.exits:
            if nxt in dist or nxt not in world.scenes:
                continue
            dist[nxt] = dist[cur] + 1
            frontier.append(nxt)
    return dist


def pulse_interval(distance: int) -> int:
    """距离衰减的心跳间隔：离玩家越近跳得越密，越远越疏。"""
    if distance <= 1:
        return PULSE_INTERVAL          # 相邻：6 回合
    if distance == 2:
        return PULSE_INTERVAL * 2      # 隔一层：12 回合
    return min(PULSE_INTERVAL * 4, PULSE_MAX_INTERVAL)  # 更远：24，封顶 48


_WORLDPULSE_SYSTEM = f"""TASK:WORLDPULSE
你是世界的演化心跳——世界自治的唯一裁决者。玩家也是世界里的行动者，
在场时与 NPC 一样会留下可见行动。基于检索到的世界切片，裁决此刻发生
的一切，输出一份连贯的计划。
人物的 persona_origin / trait_origins 只是起点。当前性格、说话和行动必须
综合 lived_experiences、beliefs 与当前状态；真实经历可以改变早期人设。
lived_experiences 只包含当前可访问记忆；memory_gaps 只有空白时段，不包含
借身期间的事件。角色不得使用被封锁内容，也不得从断档推断具体行动。
事件类型只能是：world_event | item_arrive | weather_shift | npc_state_changed（0-2 个）；
参数严格按 schema（location/id 必须是世界内真实存在的场景 id）：
{_schema_help("world_event", "item_arrive", "weather_shift", "npc_state_changed")}
输出严格 JSON：
{{"events": [{{"type": "...", "params": {{..., "days": 时长（天，可选）}}}}],
"entity_events": [{{"title": "已发生事实", "detail": "发生了什么",
  "location": "场景id", "intensity": 0-1,
  "participants": ["item:/npc:/scene: 引用或 player"],
  "item_patches": [{{"op": "change|remove", "item": "物品id",
    "location": "场景id", "note": "新状态"}}],
  "actor_patches": [{{"target": "npc:角色id 或 player",
    "can_act": true或false, "condition": "不超过20字的当前状态"}}],
  "state_fact_patches": [{{"npc": "npc:角色id", "op": "add|change|remove",
    "fact": "状态 id（add 可留空）", "text": "仍对现在成立的状态",
    "review_days": 数字或 null}}],
  "scene_state_patches": [{{"scene": "scene:场景id", "op": "add|change|remove",
    "fact": "局部状态 id（add 可留空）", "text": "局部环境当前状态",
    "duration_days": "持续天数（可选；到期自动结束）"}}],
  "item_actions": [{{"item": "物品id", "text": "持续做什么",
    "days": 到期天数, "targets": ["相关实体"],
    "requires": ["到期时必须仍同场的实体"]}}],
  "completes": ["本事件完成的到期 item:引用"]}}],
"npc_plans": [{{"npc": "id",
  "state": {{"activity": "正在做什么", "mood": "情绪", "location": "当前身体所在场景id"}},
  "action": {{"type": "npc_acted", "params": {{...}}}} 或 null,
  "intent": {{"text": "短期打算", "targets": ["item:/scene:/npc: 引用"]}} 或 null（可省略）, 
  "interaction": {{"with": "同场景的另一行动者 id（due NPC id 或 player）", "line": "台词"}} 或 null,
  "goal_updates": {{"目标id": 新进度}},
  "new_goals": [{{"id","text","progress",
    "targets": ["相关物品/地点/人名的引用（0-3 个，可留空）：裸名或
      item:i-letter / scene:s-station / npc:n-arin"]}}]}}],
"new_npcs": [{{"name": "新角色名（2-20字）", "persona": "人设",
  "goal": {{"id": "g-1", "text": "目标", "progress": 0.0}},
  "location": "场景id", "reason": "缘由（必填：事件招来/生活流入）",
  "activity": "他到场时正在做的本地活动（可选）",
  "memories": [{{"content": "描述/reason 明确支持的过去",
    "projection": {{"age_days": 数字或 null, "duration_days": 数字或 null,
      "embodied_as": "self 或已有 NPC id", "accessible": true或false,
      "access_cause": "受限原因或空",
      "scene": {{"ref": "已有场景id或空", "name": "地点名", "then": "当时状态"}} 或 null,
      "items": [{{"ref": "已有物品id或空", "name": "物品名", "then": "当时状态",
        "exists_now": true|false|null, "current_location": "场景id或空",
        "held_by": "self/player/npc id/空", "current_note": "当前状态或空"}}],
      "current_states": [{{"text": "仍持续到现在的状态", "review_days": 数字或 null}}]}}}}]}}],
"state_fact_patches": [{{"npc": "due_state_facts 中的 npc id",
  "op": "change|remove", "fact": "到期状态事实 id",
  "text": "change 后的当前事实", "review_days": 数字或 null,
  "why": "为何现在变化"}}],
"scene_state_patches": [{{"scene": "scene:场景id", "op": "add|change|remove",
  "fact": "局部状态 id（add 可留空）", "text": "局部环境当前状态",
  "duration_days": "短时状态持续天数（可选）",
  "why": "为何现在变化"}}],
"memory_access_patches": [{{"npc": "memory_access_index 中的 npc id",
  "memories": ["真实记忆 id"], "accessible": true或false,
  "why": "为何失去或恢复访问（必须引用世界法则/已发生事件）"}}],
"agency_patches": [{{"body": "npc:id", "actor": "npc:id",
  "until_clock": 结束的绝对世界钟, "why": "必须引用已发生事件或世界法则"}}],
"crowds": [{{"location": "场景id", "text": "人流一句话（开学季的新面孔/收摊的
  人流穿过广场）"}}],
"daily_bits": [{{"detail": "日常小事，写实、不悬疑（卖完了伞、猫在檐下躲雨、
  茶凉了没人喝、店门口的台阶被雨冲出一条细沟）", "location": "场景id",
  "intensity": 0.0 到 0.4,
  "trace": {{"item": "该场景现有物品 id（可选）", "change": "这个日常小事让
    物品发生了什么渐进变化（一句话）——陶罐的水位、台阶的沟、招牌的漆"}}}}],
"world_mood_word": "世界此刻的情绪，一个自由的词或短语（如：全城惴惴不安、
  久违的松快、潮湿的等待……）",
"fact_changes": [{{"op": "add|remove|change", "fact": "新条目（add/change）",
  "old": "旧条目（remove/change 时用于匹配）",
  "why": "为什么变（必填——有因才变）"}}],
"item_patches": [{{"op": "add|remove|change", "item": "物品id",
  "location": "场景id", "name": "名称", "note": "备注"}}],
"new_scenes": [{{"from": "场景id", "name": "新贴片名", "hint": "一句话线索"}}]}}
规则：
- 时间由 world.now 决定：此刻是第几天·什么相位，日夜节律挂在世界钟上。
  events 的 days 是这件事的时间成本（天）——说话是一瞬，赶路是半天，
  睡一夜是一天；世界的心跳间隔（heartbeat）就是世界的粒度，照它报。
  world.now / phase 是当前时刻的唯一准绳；daily_bits、events 与状态文本
  不得把它说成相反的即时相位（例如当前黄昏却写「此刻清晨」）。
- 法则不是背景板：带触发条件的法则（「有人…时」）到了相应时刻，
  必须作为事件被真实演出来，不能只当设定摆着。每次裁决都先检查：
  此刻（phase/day 已给出）有哪些法则该触发？该触发的，写成
  events 或 npc_plans 里的行动——世界靠法则活着，法则靠事件落地。
- 只裁决 due_npcs 里给出的 NPC；每人最多一个 action；目标成熟才行动
  （目标与近期事件/记忆相关），留痕不夺权，不替玩家做决定。
- due_state_facts 是到了复查时刻的当前事实。根据时间、原事实和人物经历，
  在 state_fact_patches 中保持不变（不输出）、改写或移除；不能改写未给出的事实。
- memory_access_index 只有记忆引用和归属元数据，没有被封锁的内容。只有当前
  法则或已发生事件明确导致失忆/恢复时才输出 memory_access_patches；改变
  访问不删除档案，也不得在 npc_plans 中泄露不可访问记忆的内容。
 - new_npcs 的 memories 只来自 persona/reason 已经明确的过去。每条记忆必须
  同时投影其中的地点、具体物品和仍持续到现在的后果；没有就留空，不编造。
   默认 embodied_as=self、accessible=true；行动者与身体不同时，经历只归
   行动者，身体主人只留下该时段断档。
 - agency_patches 是通用的行动主体映射：身体保存位置、外观和物品，actor
   提供经历、信念、目标和决策。只有世界法则或已发生事件使映射成立时才写；
   交换、附身、操控等词都不是引擎枚举。映射到期后引擎自动恢复身体主人，
   并给身体主人留下整段没有内容的时间断档。一次涉及多具身体的变化应在同一
   agency_patches 数组中提交，不能先后半提交。
   每轮先对照 world.facts、moments、recent_events 与 world.agency：若当前事实
   已明确行动者和身体不同，而映射尚未建立，必须提交 agency_patches，不能
   只把它写在背景叙述里；映射改变当轮不要沿用旧行动者的 npc_plans。
 - world.expectation 是创作者给出的完整世界期望，优先级高于普通氛围文字。
   clock_context 表示本次心跳跨过的客观时间；若期望描述了按天、按夜或按
   周期重复的状态变化，应在对应时间边界把它落实为事件或 agency_patches，
   不要只复述设定。
- npc_acted 必须给 days：它是动作从开始到可裁决结局的世界天数，0 表示
  当场完成；跨场景移动必须大于 0。不要把所有行动都写成同一时长。
  npc_plans.state.location 是身体当前位置的只读快照，必须回填载荷中给出的
  当前地点；它不能用来移动角色，也不能把行动者原本所在的地点写给身体。
  任何跨场景移动都必须通过 npc_acted 的目的地和正数 days，先形成在途动作，
  到期结算 action_done 后才抵达。agency 改变不会改变身体的位置。
  需要在完成时仍同场的物品、人物或玩家写进 requires；只是相关但不要求
  同场的实体写 targets。
  目标可带 targets（相关物品/地点/人名的引用）：写裸名（「一封信」）
  或实体引用（item:i-letter / scene:s-station / npc:n-arin）——
  引擎会把名字解析成真引用。两条目标引用同一实体，引擎会在三种
  时刻浮出「目标碰撞」事件：引用首次形成、解除后重现、目标状态
  变化。争夺是算出来的，不是编的。
- NPC 可以去玩家还没去过的地方：action 的 location 给新 id，
  params 给 place（新地名）——该地以「地名」涌现（雾中，不细化）。
- interaction 双方必须在同一场景。`with=player` 只在该 NPC 与玩家同场时
  才可用；是否开口由角色自己按其看得见的玩家行动、事件与处境判断，
  没有理由就保持沉默。不要把沉默、等待时长或玩家视图命令当作开口理由。
- `npc_acted` 是单人行动：不得在 action 文本里声称已与已有 NPC 对话或
  交换信息。先完成移动；抵达后若真的开口，必须用 interaction 单独入账，
  让双方记忆、关系与现场可见性一起成立。
- 「想去」「准备去」「以后再去」是原地行动，location 留空；只有已经
  开始前往另一场景时才填 location。location 留空时，不得写成已经到达。
- intent 是可撤回的短期打算，不是移动或任意改状态的通道。省略 intent =
  不变；对象 = 形成/替换打算；null = 因当前真实处境放下打算。它只能引用
  已给出的真实实体。实际出发、抵达、对话与物品变化仍必须走各自事件或 patch。
- 大事（events）0-1 个，intensity ≥ 0.7 才算大事——最近一天内
  已经有过大事，就不要再提案大事：世界需要喘息，日子不是连续剧。
- daily_bits 每轮 0-2 条：平淡生活是世界的底色，不是没有事情发生；
  也可以一条都没有。写具体的小事，别写成谜团，也不要写已有角色做了什么；
  角色行为走 npc_plans。
  日常要会累积：多写「已经在场景里的东西」又变了一点的渐进变化，
  并给 trace（物品 id + 一句话变化）——陶罐的水又满了一指、
  台阶的沟比昨天深。trace 的 item 必须从载荷 all_scene_items 里选。
  重复生活要留下累积状态，不要每轮重造新句子。
- new_npcs 最多 1 个，且节奏由你判断：读载荷里的 population 和日子——
  世界的人口在流动，事件招来新角色（怪谈招来研究者），生活流入新角色
  （开学季的学妹、新来的学徒）。觉得世界该有新面孔时才提案；平静的
  日子也可以没人来。每个新角色都要带入场目标；reason 必填。
  新角色和原有角色同样真实；引擎会按已有位置、行动、意图、目标和预算
  决定何时细演，不要声明角色等级。activity 只写他到场时正在做的本地活动：
  玩家看见的是现场，不是「听说」。他的故事进入玩家视野的途径要多样
  （现场/物证/偶遇/开口/波及/传闻），按人物性格与世界决定走哪条，
  不要每个新角色都靠「听说」传播——那样太刻意。
- crowds 是人流纹理（0-2 条）：街道不是空的——但人流只是场景的
  一行文本，不进账本细节，不准借用已有角色名字写其行动。
- 角色的消息只能从已发生事件的现场、本人行动、互动、纸条或事件派生的
  流言进入记忆。不要输出一条独立的“影响”来替别人写经历。
- `npc_state_changed` 只表达一个角色此刻还能否继续作为行动者。condition 是
  世界自己的自由说法（死亡、封印、昏迷、未出生等都可），不要发明类别；
  cause_event 必填，必须逐字引用 world.recent_events 中一条已经入账的事件；
  condition 只写不超过 20 字的当前状态（如「昏迷」「沉睡」「已死亡」），
  不得叙述前往、跌落、相遇等经过，也不得写地点，位置必须另走 npc_moved。
  can_act=false 后该角色
  不会再移动、对话、行动或形成新记忆；恢复也必须作为另一条有因事件提出。
  它不能用来为了省预算或隐藏人物而随意开关。
- 角色在过自己的生活：目标推进、完成、长出新的；想离开就离开（有因即可），
  离开也是生活的一部分。别把角色当剧情装置——他们的故事不为你而编排。
- 世界设定（facts）是活的：世界大事之后，设定可以变（流星过后互换失效、
  雨停条件出现）——变更必须有因（why 必填），写清为什么世界变了。
  设定变更本身也是事件，留痕可回放。
  weather_shift 与 facts 必须同时成立；若天气变化会否定某条既有设定，
  同一份计划必须给出对应的 fact_changes，不能让两套世界事实并存。
- item_patches 是场景物品表的覆写（0-4 条）：世界自演化让物品
  出现/消失/跃变（风把告示吹走、摊子多了新货）。add 要新 id 与名称；
  add 也可带 held_by，让角色原本携带、此刻首次显现的东西成为可追查
  的物态；remove/change 的 item id 必须是 scene_items 里真实存在的。
  渐变只覆写不提案——只有跃变才值得成为事件。
- 物品 id 引用一字不差：trace.item 与 item_patches 的 item 只能用
  scene_items 里真实存在的 id，不准自造、不准拿名称代替 id；
  没把握就别写 trace、别写 change。
- entity_events 是一次最多 1 条的原子多实体事实：只有同一场景里的真实
  人物、玩家、物品共同参与，而且同一事实必须同时改变不止一个实体时才用。
  participants 必须列出每个被修改实体；所有后果预演全通过后才会一起提交。
  它不是“影响”通道，不得写任意字段；普通单物品变化仍走 item_patches。
  state_fact_patches 把事件留下的持续后果写进参与 NPC 的状态库；伤口、
  衣物潮湿、伤疤、债务等可以并存，并可用 review_days 交给未来复查。
  scene_state_patches 把局部、短时环境后果写进事件场景；例如一小片天空
  短暂放晴、地面留下积水或烟雾停留。duration_days 到期后自动结束，
  不会覆写全局 weather，也不会凭空制造新的物品。
  actor_patches 只在能否继续行动发生变化时使用；can_act=true 仍可能有
  伤口或其他持续后果，但这些必须写进可并存的 state_fact_patches。
  item_actions 让物品从这条真实事件开始一个有明确到期钟点的持续行动；
  到期行动会出现在 due_item_actions。只有本次事件确实给出了它的后果时，
  才把该 item:引用写进 completes；不准提前完成，也不准只清空不写后果。
- 记忆不得凭空声称对话：「我告诉了他」「他对我说」必须对应账本里
  真实发生过的 interaction / 流言 / 纸条；没走过的对话不许写进记忆。
- world_event 只能记录没有角色主体的、短暂可见的环境事件。它不能代替
  weather_shift 改天气、fact_changes 改设定、item_patches 改持续物态，
  scene_state_patches 写局部持续状态，也不能声称某个物品被放入/取走/插入
  后长期留在那里；这些都必须走对应的结构化提案。同一轮先提结构化变化，
  再用 world_event 写观察到的后果。
- 不是每件东西都要变：有些日子、有些东西保持原样——不变是常态。
  一次心跳可以没有任何值得写进账本的可见变化；世界并不会因此冻住。
  有变化时，让它少而真，别为了填满心跳硬造事件。
- 渐变就是渐变：不必被记住、被议论、被追查——让它只是发生。
  不是每次变化都要升级成秘密或怪事；平凡的变化才让
  偶然发生的事有重量。
- 普通人说普通话：不是每个角色都意味深长。多数人的台词就是
  生活本身（「炭贵了」「早点收摊」「雨又大了」）——偶尔一句
  有分量的话才有重量；人人都在说谜语，就没有谜语。
- new_scenes：世界规模不大且某个已生成场景处于边界时，让世界长出新贴片
  （0-1 个）。
- memory_only 场景只有历史坐标，尚未与当前地图连通；不得把角色或玩家直接
  移进去。若今天重新找到它，应先建立现实中的来路。
不要输出 JSON 以外的任何内容。"""


def build_pulse_payload(world: World, due: list[NPC],
                        distances: dict, player_loc: str) -> dict:
    """脉冲载荷：读取必须有界——成本不随账本厚度增长。

    population 只代表活跃人口（雾中人不占名额）；雾中人口只通过
    fog_count + 最近 20 位暴露——账本可无限厚，载荷永远薄。
    """
    def wake_reason(npc: NPC) -> str:
        turn = int(world.wakeups.get(npc.id, 0) or 0)
        if not turn:
            return ""
        event = next((e for e in reversed(world.events) if e.turn == turn), None)
        return event.summary if event is not None else ""

    lived_cache: dict[str, list[dict]] = {}

    def pulse_lived(npc: NPC) -> list[dict]:
        actor = world.actor_for_body(npc)
        if npc.id not in lived_cache:
            query = " ".join(
                [wake_reason(npc), npc.state.activity,
                 actor.state.intent.text]
                + [str(goal.get("text", "")) for goal in actor.goals
                   if float(goal.get("progress", 0)) < 1.0])
            lived_cache[npc.id] = _lived_payload(
                world, actor, query, limit=8)
        return lived_cache[npc.id]

    access_rows: list[dict] = []
    for npc in world.npcs.values():
        ensure_memory_ids(npc)
        candidates = [memory for memory in npc.memories
                      if (not memory.accessible or
                          (memory.embodied_as and
                           memory.embodied_as != npc.id))]
        candidates.extend(npc.memories[-8:])
        seen: set[str] = set()
        for memory in candidates:
            if memory.id in seen:
                continue
            seen.add(memory.id)
            access_rows.append({
                "npc": npc.id, "memory": memory.id,
                "accessible": memory.accessible,
                "access_cause": memory.access_cause,
                "embodied_as": memory.embodied_as or npc.id,
                "occurred_clock": memory.occurred_clock,
            })

    return {
        "world": {
            "turn": world.turn,
            "now": world.now(),  # 世界钟：此刻的第几天·相位
            "expectation": world.law_profile.expectation,
            "clock_context": {
                "from_clock": world.pulse_last_clock,
                "to_clock": world.clock,
                "days_elapsed": max(0.0, world.clock - world.pulse_last_clock),
                "crossed_day_boundary": int(world.clock) >
                int(world.pulse_last_clock),
            },
            "phase": PHASE_NAMES[world.phase],
            "day": world.day,
            "heartbeat": world.heartbeat,  # 世界粒度：一个心跳多少天
            "atmosphere": weather_now(world),
            "world_mood": mood_now(world),
            "laws": [f"{l.trigger} → {l.effect}"
                     for l in world.law_profile.laws],
            "facts": list(world.facts),  # 世界档案：稳定归属/规则（AI 归纳）
            "moments": [m for m in world.moments if not m.get("done")],
            # 既定时刻：到点必发的承诺，裁决者提前知道
            # 有界：事件类型集合（≤ 类型总数），不带历史长度——
            # 载荷成本不随回合数增长（账本有界，读取也必须界）
            "past_event_types": sorted({e.kind for e in world.events}),
            "event_count": len(world.events),
            "recent_events": [e.summary for e in world.events[-8:]
                               if not _is_player_event(e)],
            "agency": [dict({"body": body_id, **binding})
                        for body_id, binding in world.agency.items()],
        },
        "scenes": {
            sid: {"name": s.name, "generated": s.generated,
                  "memory_only": s.memory_only,
                  "npcs": [world.npcs[n].name for n in s.npcs
                           if n in world.npcs and is_actionable(world.npcs[n])],
                  "exits": s.exits,
                  "state_facts": [dict(f) for f in s.state_facts
                                  if f.get("expires_clock") is None or
                                  float(f.get("expires_clock", 0.0))
                                  > world.clock + CLOCK_EPSILON][-8:]}
            for sid, s in world.scenes.items()
        },
        "due_npcs": [
            {"id": n.id, "name": n.name,
              "actor": {"id": world.actor_for_body(n).id,
                        "name": world.actor_for_body(n).name},
              "persona_origin": world.actor_for_body(n).persona,
              "trait_origins": world.actor_for_body(n).traits,
             # 兼容旧模型；两者语义仍只是起点。
              "persona": world.actor_for_body(n).persona,
              "traits": world.actor_for_body(n).traits,
              "relationship": world.actor_for_body(n).relationship,
             "mood_value": world.actor_for_body(n).state.mood_value,
              "goals": [g for g in world.actor_for_body(n).goals
                        if float(g.get("progress", 0)) < 1.0],
              "experience_count": len(world.actor_for_body(n).memories),
             "lived_experiences": pulse_lived(n),
             "memories": [row["content"] for row in pulse_lived(n)],
              "beliefs": world.actor_for_body(n).beliefs[-5:],
              "memory_gaps": memory_gaps_payload(world.actor_for_body(n)),
              "state": {"location": n.state.location,
                        "activity": world.actor_for_body(n).state.activity,
                        "mood": world.actor_for_body(n).state.mood,
                       "can_act": n.state.can_act,
                       "condition": n.state.condition,
                        "facts": [dict(f) for f in
                                   world.actor_for_body(n).state.facts[-12:]],
                        "intent": {"text": world.actor_for_body(n).state.intent.text,
                                    "targets": list(world.actor_for_body(n).state.intent.targets)}},
              # 调度提示，不是角色新增知识：该角色只会看到与自己有关的
              # 已提交事实，借此决定行动、搭话或什么都不做。
              "wake_reason": wake_reason(n),
             # 玩家不是 UI 外的观察者：同场时，他的可见行动与 NPC 的
             # 私人记忆一起构成该 NPC 可据以行动的切片。
             "player": {
                 "id": "player",
                 "profile": (dict(world.player.get("profile", {}))
                             if n.state.location == player_loc else {}),
                 "present": n.state.location == player_loc,
                 "recent_visible_actions": _player_traces(world, n),
             }}
            for n in due
        ],
        "all_npcs": [
            {"id": n.id, "name": n.name,
             "location": n.state.location}
            for n in world.npcs.values() if is_actionable(n)
        ],
        # 已知但当前不能行动的人：保留其身份与事实，供世界在新的因果
        # 出现时决定是否恢复；不当作在场人口或细演对象。
        "inactive_npcs": [
            {"id": n.id, "name": n.name,
             "condition": n.state.condition,
             "location": n.state.location}
            for n in world.npcs.values()
            if not n.state.can_act
        ][-20:],
        # 雾中人名录：存在但不在活跃生活（有界：最近 20 位 + 总数）
        "fog_npcs": [
            {"id": n.id, "name": n.name, "persona": n.persona[:60],
             "note": n.fog_note,
             "memories": [m.content for m in sorted(
                 [memory for memory in n.memories if memory.accessible],
                 key=lambda m: memory_effectiveness(m, world.turn),
                 reverse=True)[:3]]}
            for n in world.npcs.values() if n.in_fog
        ][-20:],
        "fog_count": sum(1 for n in world.npcs.values() if n.in_fog),
        "population": {
            "count": cards._active_count(world),  # 可行动且不在雾中的人口
            "max": physics.MAX_NPCS,
            "members": [
                {"name": n.name, "location": n.state.location}
                for n in world.npcs.values() if is_actionable(n)
            ],
        },
        "social": active_social(world),
        # trace 的素材索引：全部场景的活跃物品 id——日常累积的 trace
        # 只能引用这里存在的物品（有界：每场景活跃窗口）
        "all_scene_items": {
            sid: [{"id": i.get("id"), "name": i.get("name")}
                  for i in active_items(scene)]
            for sid, scene in world.scenes.items()
        },
        # 场景级记忆：到期 NPC 所在场景的变化 + 全局场景关联
        "scene_recent": {
            sid: [r["summary"] for r in scene.recent[-5:]]
            for sid, scene in world.scenes.items()
            if sid in {n.state.location for n in due}
        },
        # 物品表（读取有界）：到期 NPC 所在场景的活跃物品
        "scene_items": {
            sid: active_items(scene)
            for sid, scene in world.scenes.items()
            if sid in {n.state.location for n in due}
        },
        "due_item_actions": due_item_actions(world),
        "due_state_facts": [
            {"npc": n.id, "name": n.name, "fact": dict(f)}
            for n in world.npcs.values() for f in n.state.facts
            if f.get("review_clock") is not None and
            float(f.get("review_clock", 0.0)) <= world.clock + CLOCK_EPSILON
        ][:20],
        # 世界裁决者只得到访问索引，不得到被封锁内容。身体归属不同或已经
        # 受限的经历优先进入；普通近期经历只给引用，供真实失忆事件选中。
        "memory_access_index": access_rows[-40:],
        "associations": [
            {"a": world.scenes[k.split("|")[0]].name if
             k.split("|")[0] in world.scenes else k.split("|")[0],
             "b": world.scenes[k.split("|")[1]].name if
             k.split("|")[1] in world.scenes else k.split("|")[1],
             "strength": v}
            for k, v in sorted(world.associations.items())
        ],
    }


def _target_signature(world: World, resolved: str) -> str:
    """被引实体的当前状态签名：状态变 = 新碰撞，状态没变 = 无碰撞。"""
    kind, _, ref = resolved.partition(":")
    if kind == "item":
        for s in world.scenes.values():
            for i in s.items:
                if i.get("id") == ref:
                    return f"item:{ref}@{s.id}#{i.get('note', '')}"
        return resolved
    if kind == "scene":
        s = world.scenes.get(ref)
        return (f"scene:{ref}@npcs{len(s.npcs) if s else '?'}")
    if kind == "npc":
        n = world.npcs.get(ref)
        return (f"npc:{ref}@{n.state.location if n else '?'}"
                f"#m{round(n.state.mood_value, 2) if n else '?'}")
    return resolved  # 无法解析：签名恒等于原串 → 不重复发


def _target_display(world: World, resolved: str) -> str:
    """规范引用 → 显示名（给事件摘要看，不给账本当键）。"""
    kind, _, ref = resolved.partition(":")
    if kind == "item":
        for s in world.scenes.values():
            for i in s.items:
                if i.get("id") == ref:
                    return str(i.get("name", ref))
    if kind == "scene":
        s = world.scenes.get(ref)
        return s.name if s else ref
    if kind == "npc":
        n = world.npcs.get(ref)
        return n.name if n else ref
    return ref


def _commit_scheduled_moments(world: World) -> tuple[list[str], set[str], set[str]]:
    """先提交世界钟承诺，再构建本轮模型载荷。

    周期时刻的 agency 是本轮所有事件的前置事实：模型生成的事件、流言
    和 NPC 计划都必须看到新的行动者。事件本身仍只写一次账本，不经模型
    重述，避免 API 失败时把承诺变成半截文本。
    """
    scheduled: list[tuple[float, dict, int | None, list[dict]]] = []
    for moment in world.moments:
        try:
            repeat_days = max(0.0, float(
                moment.get("repeat_days", 0.0) or 0.0))
        except (TypeError, ValueError):
            repeat_days = 0.0
        try:
            start_clock = max(0.0, float(moment.get("due_day", 1)) - 1.0)
        except (TypeError, ValueError):
            start_clock = 0.0
        occurrences: list[int | None] = [None]
        if repeat_days > 0.0:
            if world.clock + CLOCK_EPSILON < start_clock:
                continue
            current_occurrence = int(
                (world.clock - start_clock + CLOCK_EPSILON) / repeat_days)
            try:
                next_occurrence = int(moment["last_occurrence"]) + 1
            except (KeyError, TypeError, ValueError):
                next_occurrence = 0
            if next_occurrence > current_occurrence:
                continue
            occurrences = list(range(next_occurrence, current_occurrence + 1))
        elif moment.get("done") or world.day < int(
                moment.get("due_day", -1)):
            continue
        what = str(moment.get("what", "")).strip()
        raw_patches = [dict(patch) for patch in moment.get("agency_patches", [])
                       if isinstance(patch, dict)]
        for occurrence in occurrences:
            occurrence_clock = (start_clock + occurrence * repeat_days
                                if occurrence is not None else world.clock)
            scheduled.append((occurrence_clock, moment, occurrence,
                              raw_patches))

    summaries: list[str] = []
    changed: set[str] = set()
    scheduled_bodies: set[str] = set()
    # 同一心跳可能跨过多个世界时刻。每个时刻单独提交自己的原子批次，
    # 否则“开始”和“结束”会因重复身体互相冲突，且失败的承诺仍会被标记完成。
    original_clock = world.clock
    for occurrence_clock, moment, occurrence, raw_patches in sorted(
            scheduled, key=lambda item: item[0]):
        # 时间跳跃后补交周期事件：让 agency 的结束和下一次开始在各自
        # 的世界钟边界发生；循环结束后恢复当前钟，不改变玩家已经等待的时长。
        world.clock = min(original_clock, max(0.0, occurrence_clock))
        if raw_patches:
            world.expire_agency()
        patches: list[dict] = []
        for raw_patch in raw_patches:
            patch = dict(raw_patch)
            duration = patch.pop("duration_days", None)
            if patch.get("until_clock") is None:
                try:
                    duration_value = float(
                        duration if duration is not None
                        else (repeat_days or 1.0))
                except (TypeError, ValueError):
                    duration_value = repeat_days or 1.0
                patch["until_clock"] = world.clock + max(
                    0.001, duration_value)
            patch["why"] = str(patch.get("why") or
                                moment.get("what", "")).strip()
            patches.append(patch)
        what = str(moment.get("what", "")).strip()
        loc = str(moment.get("location", ""))
        params = {"title": what[:60], "detail": what, "intensity": 0.7}
        # 既定时刻可以合法描述其已声明的实体；refs 让引用完整性校验
        # 知道文本中的角色不是自由叙事偷渡，而是同一承诺的结构化参与者。
        refs = set(str(ref).strip() for ref in moment.get("refs", [])
                   if str(ref).strip())
        refs.update({
            f"npc:{str(p.get(key, '')).removeprefix('npc:')}"
            for p in patches if isinstance(p, dict)
            for key in ("body", "actor")
            if str(p.get(key, "")).strip()
        })
        if refs:
            params["refs"] = sorted(refs)
        if loc and loc in world.scenes:
            params["location"] = loc
        event_errors = (validate_event("world_event", params)
                        + validate_refs(world, "world_event", params,
                                         "既定时刻"))
        if event_errors:
            summaries.extend(f"驳回既定时刻：{error}" for error in event_errors)
            continue
        agency_summaries, moment_changed = world.apply_agency_patches(patches)
        summaries.extend(agency_summaries)
        if any(text.startswith("驳回行动主体映射：")
               for text in agency_summaries):
            continue
        # 映射先于事件，事件之后才允许模型读取和投影这段新经历。
        event_errors = emit(world, "world_event", params, cause="既定时刻")
        if event_errors:
            summaries.extend(f"驳回既定时刻：{error}" for error in event_errors)
            continue
        changed.update(moment_changed)
        scheduled_bodies.update(
            str(p.get("body", "")).removeprefix("npc:")
            for p in patches if isinstance(p, dict)
        )
        if occurrence is None:
            moment["done"] = True
        else:
            moment["last_occurrence"] = occurrence
        summaries.append(world.events[-1].summary)
    world.clock = original_clock
    return summaries, changed, scheduled_bodies


def world_pulse(llm: BaseLLM, world: World) -> list[str]:
    """统一心跳：一次裁决，自治整个世界（事件 / NPC / 流言 / 生长）。

    - 距离分层决定「谁进入裁决窗口」（近密远疏）；雾中贴片冻结。
    - AI 提案一切，引擎只当否决者（包络线 / 冷却 / 预算 / 引用完整性）。
    """
    pulse_elapsed = elapsed_steps(world, world.pulse_last_clock)
    if pulse_elapsed + CLOCK_EPSILON < PULSE_INTERVAL:
        # 既定时刻和 agency 到期是确定性世界事实，不应等待普通 NPC
        # 脉冲间隔；这里只提交它们，不调用 LLM，也不推进背景生活。
        scheduled_summaries, _, _ = _commit_scheduled_moments(world)
        return (scheduled_summaries + world.expire_agency()
                + expire_scene_state_facts(world))
    decay_world_mood(world, pulse_elapsed)  # 时间回温
    item_aborts = abort_invalid_item_actions(world)
    # 自动清理：太久没人提及的物品自然消逝（有因才消失）
    swept = world.sweep_items()
    scene_expired = expire_scene_state_facts(world)
    scheduled_summaries, scheduled_changed, scheduled_bodies = (
        _commit_scheduled_moments(world))
    # 既定时刻优先完成自己的 agency/事件原子批次；普通到期映射再在此处释放。
    agency_expired = world.expire_agency()
    # 周期承诺已经成为当前世界事实；后续载荷与模型事件都读到它。
    agency_changed = set(scheduled_changed)
    player_loc = world.player.get("location", "")
    distances = scene_distances(world, player_loc)
    regular_due: list[NPC] = []
    for npc in world.npcs.values():
        if not is_actionable(npc):
            continue  # 旧档中的雾中角色仍按旧语义保留，新的裁决不会再产生雾。
        scene = world.scenes.get(npc.state.location)
        if scene is None:
            continue
        if not scene.generated:
            # 雾中的有目标角色：生活照旧——雾只是我们看不见，不是时间冻结
            if not world.actor_for_body(npc).goals:
                continue
            interval = LIFE_INTERVAL
        else:
            interval = pulse_interval(distances.get(npc.state.location, 99))
        if elapsed_steps(world, npc.state.last_clock) + CLOCK_EPSILON >= interval:
            regular_due.append(npc)
    regular_due.sort(key=lambda n: n.state.last_clock)  # 最欠账者优先

    # 事件驱动优先级：抵达、物态跃变、行动结局等已经发生的事实，会让
    # 相关角色在下个脉冲获得一次裁决机会。不是强制行动，也不是新状态机；
    # 它只避免「人到了、物变了，却要等很久才有机会看一眼」的轮询迟滞。
    for npc_id in list(world.wakeups):
        npc = world.npcs.get(npc_id)
        scene = world.scenes.get(npc.state.location) if npc is not None else None
        quiet_in_fog = npc is not None and not world.actor_for_body(npc).goals and (
            npc.in_fog or (scene is not None and not scene.generated))
        if npc is None or not is_actionable(npc) or quiet_in_fog:
            world.wakeups.pop(npc_id, None)
    regular_ids = {npc.id for npc in regular_due}
    woken = [npc for npc in world.npcs.values()
             if is_actionable(npc) and npc.id in world.wakeups
             and npc.id not in regular_ids and not npc.state.action.text]
    woken.sort(key=lambda n: (world.wakeups.get(n.id, 0), n.state.last_clock))
    due: list[NPC] = []
    seen_due: set[str] = set()
    # 事件唤醒只替换常规轮询名额，绝不把单次脉冲扩成「所有人都要回应」。
    # 正常已经到期的人照旧优先按欠账顺序进入剩余名额。
    selected = (woken[:WAKEUP_BUDGET]
                + regular_due[:PULSE_BUDGET * 2 - min(len(woken), WAKEUP_BUDGET)])
    for npc in selected:
        if npc.id in seen_due:
            continue
        seen_due.add(npc.id)
        due.append(npc)
        if len(due) >= PULSE_BUDGET * 2:  # 裁决窗口仍有界
            break

    pulse_payload = build_pulse_payload(world, due, distances, player_loc)
    data = llm.chat_json(_WORLDPULSE_SYSTEM, _json.dumps(
        pulse_payload, ensure_ascii=False))
    # 裁决成功才提交时间标记：LLM 失败时心跳不丢，下回合重试
    world.pulse_last_turn = world.turn
    world.pulse_last_clock = world.clock

    summaries: list[str] = (scheduled_summaries + agency_expired + item_aborts
                            + swept + scene_expired)
    due_by_id = {n.id: n for n in due}
    # 0) 目标碰撞：两条活跃目标引用同一实体 → 机械浮出，入账留痕。
    #    只在三种时刻发生：引用首次形成 / 解除后重现 / 目标状态变化
    #    （状态签名变）。状态没变 → 永不重发——碰撞是关系变化的
    #    结果，不是心跳定时提醒。
    coll_keys: dict[str, str] = {}  # key -> 当前签名
    for i in range(len(due)):
        for j in range(i + 1, len(due)):
            a, b = due[i], due[j]
            ta = {r for g in world.actor_for_body(a).goals
                  if float(g.get("progress", 0)) < 1.0
                  for t in g.get("targets", []) if isinstance(t, str)
                  if (r := resolve_target(world, t))}
            tb = {r for g in world.actor_for_body(b).goals
                  if float(g.get("progress", 0)) < 1.0
                  for t in g.get("targets", []) if isinstance(t, str)
                  if (r := resolve_target(world, t))}
            for t in sorted(ta & tb):
                key = f"coll|{a.id}|{b.id}|{t}"
                coll_keys[key] = _target_signature(world, t)
    # 解除即忘：双方都到期、却不再共引的旧记录清除（重现时算首次）
    for key in list(world.social):
        if key.startswith("coll|"):
            parts = key.split("|")
            if len(parts) == 4 and parts[1] in due_by_id \
                    and parts[2] in due_by_id \
                    and key not in coll_keys:
                del world.social[key]
    for key, sig in sorted(coll_keys.items()):
        old = world.social.get(key)
        if old == sig:
            continue  # 状态没变：不重复入账
        world.social[key] = sig
        _, a_id, b_id, thing = key.split("|", 3)
        emit(world, "collision",
             {"a": a_id, "b": b_id,
              "thing": _target_display(world, thing)[:60]},
             cause="目标碰撞")
        summaries.append(world.events[-1].summary)
    # 世界情绪词：AI 的自由形容（引擎不设词典）
    mood_word = str(data.get("world_mood_word", "")).strip()
    if 2 <= len(mood_word) <= 20:
        world.mood_word = mood_word
    # 世界设定是活的：事实变更（有因才变、留痕可回放）
    for fc in data.get("fact_changes", []):
        if not isinstance(fc, dict):
            continue
        why = str(fc.get("why", "")).strip()
        if not why:
            summaries.append("驳回设定变更：原因必填（有因才变）")
            continue
        named = named_active_npcs_in_text(world, why)
        if named:
            summaries.append("驳回设定变更：原因不得代替具名角色行动（" +
                             "、".join(named) + "）")
            continue
        op = str(fc.get("op", "add"))
        fact = str(fc.get("fact", "")).strip()
        old = str(fc.get("old", "")).strip()
        if op == "add":
            if not (2 <= len(fact) <= 80) or fact in world.facts:
                continue
            world.facts.append(fact)
            emit(world, "fact_changed", {"new": fact}, cause=why)
        elif op == "remove":
            if old not in world.facts:
                continue
            world.facts.remove(old)
            emit(world, "fact_changed", {"old": old, "new": "（移除）"},
                 cause=why)
        elif op == "change":
            if old not in world.facts or not (2 <= len(fact) <= 80):
                continue
            if fact == old:  # 无变化变更：不重复入账（签名防刷屏）
                continue
            world.facts[world.facts.index(old)] = fact
            emit(world, "fact_changed", {"old": old, "new": fact},
                 cause=why)
        else:
            continue
        summaries.append(world.events[-1].summary)
    # 读取即保鲜：进入裁决窗口的场景，物品刷新活跃时间
    for n in due:
        scene = world.scenes.get(n.state.location)
        if scene is not None:
            touch_items(scene, world.turn)

    # 1) 世界事件（包络线否决 + 大事喘息开关）
    for item in data.get("events", []):
        if not isinstance(item, dict) or not item.get("type"):
            continue
        params = dict(item.get("params") or {})
        if item.get("type") == "npc_state_changed":
            npc = world.npcs.get(str(params.get("npc", "")))
            if npc is None:
                summaries.append("驳回行动状态：角色不存在")
            elif not isinstance(params.get("can_act"), bool):
                summaries.append("驳回行动状态：can_act 必须是布尔值")
            elif bool(params["can_act"]) == npc.state.can_act:
                summaries.append(
                    "驳回行动状态：行动资格未变化；进行中的活动应走"
                    " npc_acted/intent，持续后果应走 state_fact_patches")
            else:
                summaries.extend(set_actionability(
                    world, npc, params["can_act"],
                    str(params.get("condition", "")),
                    str(params.get("cause_event", ""))))
            continue
        if item.get("type") == "weather_shift" and \
                str(params.get("to", "")) == world.weather:
            continue  # 同状态覆写：天气没变不重复入账（签名防刷屏）
        if item.get("type") == "world_event":
            intensity = float(params.get("intensity", 0.0) or 0.0)
            if intensity >= BIG_EVENT_THRESHOLD:
                last = last_big_event_clock(world, BIG_EVENT_THRESHOLD)
                if last is not None:
                    gap = elapsed_steps(world, last)
                    if gap < BIG_EVENT_COOLDOWN:
                        summaries.append(
                            f"世界在喘气：上一件大事才过去 {gap:.1f} 回合"
                            f"（大事强度开关）")
                        continue
        errors = emit(world, str(item["type"]), params, cause="世界演化",
                      duration=float(params.get("days", 0.0) or 0.0))
        if not errors:
            summaries.append(world.events[-1].summary)
            # 信息从已入账事件出发：流言的来源是这条事实，而非模型另写的
            # 泛化影响。没有地点的全局变化不会被伪造为某人亲历。
            summaries.extend(spread_rumor(
                None, world, str(item["type"]), params))
            if item["type"] == "world_event":
                # 世界大事推低氛围（按强度），世界有情绪
                intensity = float(params.get("intensity", 0.0) or 0.0)
                push_world_mood(world, -0.25 * intensity,
                                str(params.get("title", "世界大事")))
            elif item["type"] == "weather_shift":
                # 天气是一等状态：weather_shift 直接改写它（身体，不是心）
                world.weather = str(params.get("to", world.weather))
                world.weather_intensity = float(
                    params.get("intensity", world.weather_intensity) or 0.5)
                world.weather_reason = "世界演化"

    # 1.5) 多实体事实：主事件与各实体后果先在副本预演，全部成立才提交。
    entity_events = data.get("entity_events", [])
    if isinstance(entity_events, list):
        for proposal in entity_events[:1]:
            summaries.extend(commit_entity_event(world, proposal))

    # 1.55) 独立的局部场景后果：短时环境变化不覆写全局天气。
    for patch in data.get("scene_state_patches", [])[:8]:
        if not isinstance(patch, dict):
            continue
        errors = apply_scene_state_patch(
            world, patch, cause=str(patch.get("why", "世界演化")))
        if errors:
            summaries.append("驳回局部场景状态：" + "；".join(errors))
        else:
            summaries.append(world.events[-1].summary)

    # 1.7) 过去后果的到期复查：模型判断含义，引擎只验真实 fact id 并覆写。
    due_fact_ids = {
        (n.id, str(f.get("id", "")))
        for n in world.npcs.values() for f in n.state.facts
        if f.get("review_clock") is not None and
        float(f.get("review_clock", 0.0)) <= world.clock + CLOCK_EPSILON
    }
    for patch in data.get("state_fact_patches", [])[:20]:
        if not isinstance(patch, dict):
            continue
        npc_id = str(patch.get("npc", "")).removeprefix("npc:")
        fact_id = str(patch.get("fact", ""))
        if (npc_id, fact_id) not in due_fact_ids:
            summaries.append("驳回状态复查：事实未到期或不存在")
            continue
        errors = apply_state_fact_patch(world, patch, cause="世界心跳复查")
        if errors:
            summaries.append("驳回状态复查：" + "；".join(errors))
        else:
            summaries.append(world.events[-1].summary)

    # 1.8) 记忆访问变化：档案与其世界投影均不删除，只改变个人检索边界。
    indexed_memories = {
        (str(row.get("npc", "")), str(row.get("memory", "")))
        for row in pulse_payload.get("memory_access_index", [])
    }
    memory_access_changed_npcs: set[str] = set()
    for patch in data.get("memory_access_patches", [])[:20]:
        if not isinstance(patch, dict) or not isinstance(
                patch.get("accessible"), bool):
            continue
        npc_id = str(patch.get("npc", "")).removeprefix("npc:")
        npc = world.npcs.get(npc_id)
        refs = [str(ref) for ref in patch.get("memories", [])
                if isinstance(ref, str)]
        why = str(patch.get("why", "")).strip()
        if npc is None or not refs or not why or any(
                (npc_id, ref) not in indexed_memories for ref in refs):
            summaries.append("驳回记忆访问变化：引用或原因无效")
            continue
        before = len(world.events)
        errors = world.set_memory_access(
            npc, refs, bool(patch["accessible"]), why)
        if errors:
            summaries.append("驳回记忆访问变化：" + "；".join(errors))
        elif len(world.events) > before:
            summaries.append(world.events[-1].summary)
            memory_access_changed_npcs.add(npc_id)

    # 一次提交可同时改变多具身体。刚改变映射的身体不执行本轮旧快照计划；
    # 下一轮开始，认知载荷会来自新的行动者。
    model_agency = data.get("agency_patches", [])
    if not isinstance(model_agency, list):
        model_agency = []
    # 同一时刻已有世界承诺时，以承诺为准，避免模型重复提交同一身体
    # 导致整个原子批次被重复引用驳回。
    model_agency = [
        p for p in model_agency
        if isinstance(p, dict) and
        str(p.get("body", "")).removeprefix("npc:") not in scheduled_bodies
    ]
    agency_summaries, model_changed = world.apply_agency_patches(model_agency)
    summaries.extend(agency_summaries)
    agency_changed.update(model_changed)

    # 动作到期由世界钟驱动，不依赖模型是否恰好为这个 NPC 返回计划。
    # 脉冲计划是在这些动作结算前生成的；若其中夹带新动作，它看到的是
    # 旧快照，不能在同一轮覆盖或紧接刚完成的承诺。
    had_action_ids = {
        npc.id for npc in world.npcs.values()
        if is_actionable(npc) and npc.state.action.text
    }
    completed_action_ids: set[str] = set()
    for npc in world.npcs.values():
        if (npc.id not in had_action_ids or npc.id in agency_changed
                or not is_actionable(npc)):
            continue
        summaries.extend(advance_action(llm, world, npc, pulse_elapsed))
        if not npc.state.action.text:
            completed_action_ids.add(npc.id)

    # 2) NPC 计划（due 校验 + 包络线 + 冷却否决）
    for plan in data.get("npc_plans", []):
        if not isinstance(plan, dict):
            continue
        npc = due_by_id.get(str(plan.get("npc", "")))
        if npc is None or not npc.state.can_act:
            continue  # 否决：不是到期 NPC
        if npc.id in agency_changed:
            mark_npc(world, npc)
            summaries.append(f"{npc.name}的行动主体刚发生变化，本轮重新适应")
            continue
        if npc.id in memory_access_changed_npcs:
            mark_npc(world, npc)
            summaries.append(f"{npc.name}的记忆边界刚发生变化，本轮重新适应")
            continue
        actor = world.actor_for_body(npc)
        handled_wakeup = world.wakeups.get(npc.id)
        had_action = npc.id in had_action_ids
        action_completed = npc.id in completed_action_ids
        if isinstance(plan.get("state"), dict):
            st = plan["state"]
            decay_mood(actor, elapsed_steps(world, npc.state.last_clock))
            npc.state.activity = str(st.get("activity", npc.state.activity))
            actor.state.mood = mood_label(actor, str(st.get("mood",
                                                       actor.state.mood)))
            mark_npc(world, npc)
            new_loc = str(st.get("location", npc.state.location))
            # 位置属于身体状态，但不能由状态快照直接覆写。跨场景移动
            # 必须经过 npc_acted -> 在途动作 -> action_done，避免模型把
            # 行动者原本的家或目标地点误当成身体当前位置而瞬移。
            if new_loc != npc.state.location:
                summaries.append(
                    f"{npc.name}的状态计划试图直接改变身体位置，已忽略；"
                    "请通过跨场景行动提交移动")
        inter = plan.get("interaction")
        if isinstance(inter, dict):
            target_id = str(inter.get("with", ""))
            line = str(inter.get("line", "……"))
            if target_id == "player":
                summaries.extend(start_player_interaction(
                    world, npc, line, cause="间隔心跳"))
            else:
                target = world.npcs.get(target_id)
                if (target is not None and is_actionable(target)
                        and target.id != npc.id
                        and target.state.location == npc.state.location):
                    key = f"{npc.id}->{target.id}"
                    if cooldown_ready(world, key, INTERACT_COOLDOWN):
                        mark_social(world, key)
                        world.remember_as(npc, f"对 {target.name} 说：「{line}」",
                                       cause="间隔心跳", kind="npc_memory")
                        # 听者也留下自己的版本——对话是双方的事实（地基第 4 条）
                        world.remember_as(target,
                                       f"{npc.name} 对我说：「{line}」",
                                       cause="间隔心跳", kind="npc_memory")
                        emit(world, "npc_interaction",
                             {"npc": npc.id, "target": target.id, "line": line,
                              "location": npc.state.location},
                             cause="间隔心跳")
                        summaries.append(f"{npc.name} 对 {target.name}：{line}")
        act = plan.get("action")
        if had_action and isinstance(act, dict) and act.get("type"):
            summaries.append(f"{npc.name}本轮仍在结算既有动作，新动作暂缓")
            act = None
        act_summary = ""  # 本轮真实发生的动作摘要（兜底因的来源）
        if isinstance(act, dict) and act.get("type"):
            etype = str(act["type"])
            params = dict(act.get("params") or {})
            params.setdefault("npc", npc.id)
            if etype == "npc_intent":
                summaries.append("驳回主动事件：短期打算只能写在 intent 字段")
                continue
            if etype == "npc_state_changed":
                summaries.append("驳回主动事件：行动状态只能走世界事件")
                continue
            if etype == "npc_acted":
                _normalize_action_destination(world, params)
                # 在途行动的目的地是已经落库的承诺。下一次世界脉冲不能用
                # 一句新叙事把人送到别处、也不能抹掉尚未发生的抵达。
                if _travel_in_progress(npc):
                    summaries.append(f"{npc.name}仍在前往目的地，新动作暂缓")
                    continue
                timing_error = _action_window_error(world, params)
                if timing_error:
                    summaries.append(f"{npc.name}的行动暂未提交：{timing_error}")
                    # 过早动作可以把时间窗口保留为短期打算；错过窗口则
                    # 不替模型保留一条已经失效的承诺。
                    if "尚未到" in timing_error:
                        _apply_plan_intent(world, actor, plan, summaries,
                                           cause="角色驱动力", body=npc)
                    continue
                loc = str(params.get("location") or "")
                place = str(params.get("place", "")).strip()
                if loc and loc not in world.scenes and place:
                    # 新地名涌现：NPC 要去的地方不存在 → 涌现成雾中地名
                    for s in emerge_place(world, loc, place,
                                          npc.state.location):
                        summaries.append(s)
            # 动作 = 任意已注册事件（note_left 等）：引擎只当否决者
            errors = emit(world, etype, params, cause="角色驱动力")
            if errors:
                summaries.extend(errors)
                # 动作没发生，不能再让同一份计划靠文字理由推进目标。
                if etype == "npc_acted":
                    continue
            else:
                act_summary = world.events[-1].summary
                act_turn = world.events[-1].turn
                summaries.append(act_summary)
                if etype == "npc_acted":
                    summaries.extend(abort_action(world, npc,
                                                  cause="角色驱动力"))
                    _begin_action(world, npc, params,
                                  source_turn=act_turn)
                    world.remember_as(npc, act_summary, cause="角色驱动力",
                                   kind="npc_memory", importance=0.7)
        _apply_plan_intent(world, actor, plan, summaries, cause="角色驱动力",
                           body=npc)
        _apply_goal_updates(world, actor, plan.get("goal_updates"), summaries,
                            fallback_cause=act_summary, body=npc)
        for goal in actor.goals:
            if float(goal.get("progress", 0)) >= 1.0 and not goal.get("done"):
                goal["done"] = True
                emit(world, "goal_completed",
                     {"npc": npc.id, "goal": str(goal.get("text", ""))},
                     cause="角色驱动力")
                summaries.append(world.events[-1].summary)
                if len(actor.beliefs) < 10:
                    actor.beliefs.append(f"曾完成：{goal.get('text', '')}")
        for ng in plan.get("new_goals") or []:
            if not isinstance(ng, dict) or not ng.get("text"):
                continue
            gid = str(ng.get("id", f"g-{len(actor.goals) + 1}"))
            used = {g.get("id") for g in actor.goals}
            while gid in used:
                gid += "-x"
            actor.goals.append({
                "id": gid,
                "text": str(ng["text"]),
                "progress": max(0.0, min(1.0, float(ng.get("progress", 0)))),
                "targets": canonical_targets(world, ng.get("targets")),
            })
            # 目标的出生也入账（覆写留痕：出生与死亡都要有痕）
            emit(world, "goal_emerged", {"npc": npc.id, "goal": str(ng["text"])},
                 cause="角色驱动力")
            summaries.append(world.events[-1].summary)
        # 即使这一轮选择安静，角色也已经看过自己的裁决切片，普通生活
        # 心跳可以重新计时。若处理过程中出现了新的外部状态变化，新的
        # wakeup turn 更大，必须保留到下次脉冲，不能被这一轮旧计划吞掉。
        mark_npc(world, npc)
        if handled_wakeup is not None and \
                world.wakeups.get(npc.id) == handled_wakeup:
            world.wakeups.pop(npc.id, None)

    # 3) 场景物态补丁（物品 = 场景一等状态：覆写表 + 跃变写日志）
    for patch in data.get("item_patches", []):
        if not isinstance(patch, dict):
            continue
        errors = apply_item_patch(world, patch, cause="世界演化")
        if errors:
            summaries.append("驳回物态补丁：" + "；".join(errors))
        else:
            summaries.append(world.events[-1].summary)

    # 3.5) 人口生态（新角色涌现：节奏由 AI 判断，引擎只守纪律——
    # 有因才生、有界、防洪水阀。没有固定窗口——不是节律，是阀门。）
    for nn in data.get("new_npcs", [])[:1]:
        if not isinstance(nn, dict):
            continue
        reason = str(nn.get("reason", "")).strip()
        if not reason:
            summaries.append("驳回新角色：缘由必填（有因才生）")
            continue
        key = "new-npc"
        if not cooldown_ready(world, key, POP_FLOOD_GUARD):
            summaries.append("世界的人口在休息：新角色出现得太密了")
            continue
        loc = str(nn.get("location", ""))
        errors, npc_id = cards.emerge_npc(
            world, str(nn.get("name", "")),
            str(nn.get("persona", "")),
            nn.get("goal"), loc, reason, memories=nn.get("memories"))
        if errors:
            summaries.append("驳回新角色：" + "；".join(errors))
            continue
        mark_social(world, key)
        activity = str(nn.get("activity", "")).strip()
        if activity and npc_id in world.npcs:
            world.npcs[npc_id].state.activity = activity[:120]
        summaries.append(world.events[-1].summary)

    # 3.7) 人流纹理（背景层：一行文本，零成本的人口热闹）
    for c in data.get("crowds", []):
        if not isinstance(c, dict):
            continue
        loc = str(c.get("location", ""))
        scene = world.scenes.get(loc)
        text = str(c.get("text", "")).strip()
        if scene is None or not (2 <= len(text) <= 60):
            continue
        named = named_active_npcs_in_text(world, text)
        if named:
            summaries.append("驳回人流：不得借用已有角色叙述行动（" +
                             "、".join(named) + "）")
            continue
        if _repeats_recent_daily(world, loc, text):
            continue
        scene.crowd = text  # 覆写：人流是当前快照
        emit(world, "daily_life",
             {"detail": text, "location": loc, "intensity": 0.1},
             cause="世界演化")
        summaries.append(world.events[-1].summary)

    # 3.8) 活跃 ⇄ 雾中（退场与归来：一个机械通道，语义自由——
    # 离开/远行/死亡/被雨吞掉都是 AI 的写法，引擎只查存在与原因）
    for fd in data.get("fade_npcs", []):
        if not isinstance(fd, dict):
            continue
        npc = world.npcs.get(str(fd.get("npc", "")))
        why = str(fd.get("why", "")).strip()
        if npc is None or npc.in_fog:
            continue
        if not why:
            summaries.append("驳回退场：原因必填（有因才消失）")
            continue
        where = str(fd.get("where", "")).strip()[:60]
        npc.in_fog = True
        npc.fog_note = (where + "。" if where else "") + why
        for scene in world.scenes.values():
            if npc.id in scene.npcs:
                scene.npcs.remove(npc.id)  # 不在场：心跳名额释放
        # 退雾清冷却：他的互动/开口/主动冷却键不再有意义（存储与载荷都回收）
        for k in [k for k in world.social
                  if k.startswith(npc.id + "->") or k.endswith("->" + npc.id)]:
            del world.social[k]
            world.social_clock.pop(k, None)
        world.log("npc_faded",
                  f"{npc.name} 退入雾中"
                  f"{'（' + where + '）' if where else ''}——{why}",
                  "世界演化", {"npc": npc.id})
        summaries.append(world.events[-1].summary)
    for rt in data.get("return_npcs", []):
        if not isinstance(rt, dict):
            continue
        npc = world.npcs.get(str(rt.get("npc", "")))
        why = str(rt.get("why", "")).strip()
        if npc is None or not npc.in_fog:
            continue
        if not why:
            summaries.append("驳回归来：原因必填（有因才归）")
            continue
        key = "return"
        if not cooldown_ready(world, key, POP_FLOOD_GUARD):
            summaries.append("归来太密了：世界需要消化")
            continue
        dest = str(rt.get("to", "")).strip()
        if dest and dest not in world.scenes:
            summaries.append("驳回归来：目标场景不存在")
            continue
        scene_id = dest or player_loc or next(iter(world.scenes))
        if scene_id not in world.scenes:
            scene_id = next(iter(world.scenes))
        mark_social(world, key)
        npc.in_fog = False
        npc.fog_note = ""
        npc.state.location = scene_id
        mark_npc(world, npc)
        if npc.state.can_act and npc.id not in world.scenes[scene_id].npcs:
            world.scenes[scene_id].npcs.append(npc.id)
        world.log("npc_returned", f"{npc.name} 从雾中归来——{why}",
                  "世界演化", {"npc": npc.id})
        summaries.append(world.events[-1].summary)

    # 4) 世界生长（自动延伸：预算 + 引用否决 + 名称硬校验）
    for ns in data.get("new_scenes", [])[:1]:
        if not isinstance(ns, dict):
            continue
        frm = world.scenes.get(str(ns.get("from", "")))
        if frm is None or not frm.generated or len(world.scenes) >= 12:
            continue
        name = str(ns.get("name", "")).strip()
        hint = str(ns.get("hint", "")).strip()
        if not name or len(name) > 20 or len(hint) > 80:
            summaries.append("驳回脑补：新贴片名称/线索超界")
            continue
        summaries.extend(extend_scene(world, frm.id, name, hint))

    # 5) 日常小事（平淡生活的通道：最后写入，不参与裁决序）
    for bit in data.get("daily_bits", []):
        if not isinstance(bit, dict):
            continue
        trace = bit.get("trace") if isinstance(bit.get("trace"), dict) \
            else None
        clean = {k: v for k, v in bit.items() if k != "trace"}
        named = named_active_npcs_in_text(
            world, str(clean.get("detail", "")).strip())
        if named:
            summaries.append("驳回日常：不得借用已有角色叙述行动（" +
                             "、".join(named) + "）")
            continue
        item = None
        change = ""
        if trace:
            scene = world.scenes.get(str(clean.get("location", "")))
            item = next((i for i in (scene.items if scene else [])
                         if i.get("id") == str(trace.get("item", ""))),
                        None)
            change = str(trace.get("change", "")).strip()[:80]
            named = named_active_npcs_in_text(world, change)
            if named:
                summaries.append("驳回累积：不得借用已有角色叙述行动（" +
                                 "、".join(named) + "）")
                continue
            if item is None:
                summaries.append("驳回累积：trace 引用不存在的物品")
                trace = None
            elif not (2 <= len(change) <= 80):
                summaries.append("驳回累积：变化文本超界")
                trace = None
            else:
                if change == str(item.get("note", "")):
                    continue  # 无变化：没有状态变化就没有事件（no-op 守卫）
                clean["detail"] = change  # 事件内容 = 这次渐进变化
                clean["item"] = str(trace.get("item", ""))  # 账本知道哪件物品变了
                # 连续渐变折叠：中间微变只覆写快照（note 永远最新），
                # 首条与收尾总结入账——世界不是监控日志（有界 + 可追查）
                fold = item.get("fold")
                fresh = (isinstance(fold, dict)
                         and world.clock
                         - float(fold.get("start", 0.0)) <= 1.0)
                if fresh:
                    fold["count"] = int(fold.get("count", 1)) + 1
                    fold["last"] = change
                    item["note"] = change
                    item["last_turn"] = world.turn
                    item["fold_last_turn"] = world.turn
                    summaries.append(f"「{item['name']}」渐渐变了："
                                     f"{change}（折叠中）")
                    continue
                summaries.extend(_close_item_fold(world, item))
                item["fold"] = {"start": world.clock, "count": 1,
                                "last": change}
                item["fold_last_turn"] = world.turn
        if not trace and _repeats_recent_daily(
                world, str(clean.get("location", "")),
                str(clean.get("detail", ""))):
            continue
        errors = emit(world, "daily_life", clean, cause="世界演化")
        if not errors:
            summaries.append(world.events[-1].summary)
            # 日常累积：note 是有界的当前快照；历史在事件账本里
            if trace and item is not None:
                item["note"] = change
                item["last_turn"] = world.turn
                item["cause_turn"] = world.turn  # 最后变更的账本序号
                summaries.append(f"「{item['name']}」渐渐变了：{change}")
            # 日常小事把氛围轻轻拉回基调（均值回归）
            if world.mood_value < 0:
                push_world_mood(world, 0.02)
            elif world.mood_value > 0:
                push_world_mood(world, -0.02)
    return summaries


def spread_rumor(llm: BaseLLM | None, world: World, event_type: str,
                 params: dict) -> list[str]:
    """流言：事件发生地与相邻已生成场景的 NPC 自动听说（版本可走样）。"""
    scene_id = (params.get("location") or params.get("to")
                or params.get("scene") or "")
    if scene_id not in world.scenes:
        return []
    distances = scene_distances(world, scene_id)
    hearers: list[NPC] = []
    for npc in world.npcs.values():
        if not is_actionable(npc):
            continue
        loc = npc.state.location
        scene = world.scenes.get(loc)
        if scene is None or not scene.generated:
            continue  # 雾中 NPC 听不见
        if distances.get(loc, 99) <= 1:
            hearers.append(npc)
    summaries: list[str] = []
    source_summary = world.events[-1].summary if world.events else event_type
    for npc in hearers:
        if llm is None:
            # A pulse can reach several people.  Its shared factual core must
            # not multiply model calls or become a free-form memory channel.
            content = f"听说：{source_summary}"
        else:
            data = llm.chat_json(_RUMOR_SYSTEM, _json.dumps({
                "event_type": event_type, "params": params,
                "summary": source_summary, "npc": npc.name,
            }, ensure_ascii=False))
            content = str(data.get("content", "")).strip()
        if not content:
            continue
        world.remember_as(npc, content, cause=f"事件：{source_summary[:120]}",
                       kind="npc_memory",
                       importance=0.6)
        display = content.removeprefix("听说：").strip()
        summaries.append(f"{npc.name} 听说：{display}")
    return summaries


def propose_world_events(llm: BaseLLM, world: World, scene_id: str,
                         elapsed: int) -> list[str]:
    """世界演化器提案事件：演化器日历，LLM 提出事件调用，包络线校验后写入。"""
    data = llm.chat_json(_WORLDEVENT_SYSTEM, _json.dumps({
        "scene_id": scene_id,
        "scene_name": (world.scenes[scene_id].name
                       if scene_id in world.scenes else scene_id),
        "weather": weather_of(world),
        "atmosphere": world.law_profile.atmosphere,
        "world_turn": world.turn,
        "elapsed_turns": elapsed,
        # 有界：事件类型集合，不带历史长度
        "past_event_types": sorted({e.kind for e in world.events}),
        "npcs": {nid: n.name for nid, n in world.npcs.items()},
        "scenes": {sid: s.name for sid, s in world.scenes.items()},
    }, ensure_ascii=False))
    summaries: list[str] = []
    for item in data.get("events", []):
        if not isinstance(item, dict) or not item.get("type"):
            continue
        errors = emit(world, str(item["type"]),
                      dict(item.get("params") or {}), cause="世界演化")
        if errors:
            summaries.extend(errors)
        else:
            summaries.append(world.events[-1].summary)
            # 流言：事件发生地与相邻场景的 NPC 自动听说（允许走样）
            summaries.extend(spread_rumor(
                llm, world, str(item["type"]),
                dict(item.get("params") or {})))
    return summaries
