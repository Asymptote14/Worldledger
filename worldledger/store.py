"""世界库：世界状态的外置存储（写入即记忆原则的载体）。

所有世界状态以显式结构持久化为 JSON；所有变化以 append-only
事件日志记录，每条事件携带原因引用。NPC 的记忆存在这里，
而不是存在任何上下文中——读取取代重述。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

def _default_save_path() -> Path:
    override = os.environ.get("WORLDLEDGER_SAVE_PATH")
    if override:
        return Path(override).expanduser()

    data_override = os.environ.get("WORLDLEDGER_DATA_DIR")
    if data_override:
        return Path(data_override).expanduser() / "universe.json"

    if os.name == "nt":
        root = Path(os.environ.get(
            "LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "WorldLedger" / "universe.json"

    root = Path(os.environ.get(
        "XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "worldledger" / "universe.json"


DEFAULT_SAVE_PATH = _default_save_path()

# ---------------- 世界时钟：回合 → 游戏内时间 ----------------

TURNS_PER_PHASE = 6
TURNS_PER_DAY = TURNS_PER_PHASE * 4
PHASE_NAMES = ["清晨", "白昼", "黄昏", "深夜"]


def phase_of(turn: int) -> int:
    return (turn // TURNS_PER_PHASE) % 4


def phase_name(turn: int) -> str:
    return PHASE_NAMES[phase_of(turn)]


def day_of(turn: int) -> int:
    return turn // TURNS_PER_DAY + 1


def game_time(turn: int) -> str:
    """游戏内时间戳：第X天·相位。"""
    return f"第{day_of(turn)}天·{phase_name(turn)}"


# ---------------- 记忆动力学 ----------------

MEMORY_HALF_LIFE = 48  # 记忆衰减半衰期（回合）；衰减只发生在读取时


def memory_effectiveness(mem: "Memory", now: int) -> float:
    """有效重要度 = 重要度 × 时间衰减。读取时计算，存储保持原文。"""
    age = max(0, now - mem.turn)
    return mem.importance * math.exp(-age / MEMORY_HALF_LIFE)


def text_similarity(a: str, b: str) -> float:
    """字符二元组 Jaccard——离线、零依赖的中文语义近似。

    「我叫什么」与「玩家的真名是黑猫」共享「我/的/名」等字符，
    得分高于与「今天雨好大」——语义检索的确定性兜底。
    """
    def grams(s: str) -> set:
        t = "".join(s.split())
        return set(t) | {t[i:i + 2] for i in range(len(t) - 1)}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def retrieval_window(npc: "NPC", now: int, recent: int = 3,
                     fill: int = 3, query: str = "",
                     embedder=None) -> list["Memory"]:
    """拟人化记忆窗口 + 语义检索：最近的最清楚，重要的最深刻，
    问得准的能被捞出来。

    - 最近 recent 条全取（话头接得上）。
    - 其余按「有效重要度 + 4×语义相关」排序补 fill 条：
      语义命中压过时间优势——「我叫什么」能捞出尘封的
      「名字是黑猫」。无 query / 无 embedder 时退化为旧行为。
    """
    memories = [memory for memory in npc.memories if memory.accessible]
    recents = memories[-recent:]
    rest = [m for m in memories if m not in recents]
    if query:
        def text_of(m: "Memory") -> str:
            """检索文本 = 内容 + 写入时打的关键词——不赌记忆措辞。"""
            return (m.content + " " + " ".join(m.keywords)).strip()
        texts = [text_of(m) for m in rest]
        sims = (embedder.similarities(query, texts) if embedder is not None
                else [text_similarity(query, text) for text in texts])
        sim_of = {id(m): s for m, s in zip(rest, sims)}
        def score(m: "Memory") -> float:
            return memory_effectiveness(m, now) + 4.0 * sim_of.get(id(m), 0.0)
    else:
        def score(m: "Memory") -> float:
            return memory_effectiveness(m, now)
    fillers = sorted(rest, key=score, reverse=True)[:fill]
    return recents + fillers


def ensure_memory_ids(npc: "NPC") -> None:
    """给旧档与导入卡中的经历补稳定 id；经历只追加，所以 id 不漂移。"""
    used = {m.id for m in npc.memories if m.id}
    for index, memory in enumerate(npc.memories, 1):
        if memory.id:
            continue
        stem = f"m-{memory.turn or 'origin'}-{index}"
        candidate = stem
        suffix = 2
        while candidate in used:
            candidate = f"{stem}-{suffix}"
            suffix += 1
        memory.id = candidate
        used.add(candidate)


def experience_window(npc: "NPC", now: int, query: str = "",
                      focus_ids: list[str] | None = None,
                      limit: int = 12) -> list["Memory"]:
    """人物的一次有限注意力：最近、当前深挖链和话题相关经历。

    档案本身不裁剪。focus 的前后各带一条，玩家连续追问同一段过去时，
    可以沿时间邻域继续深入；剩余名额按话题相关度与记忆有效度补齐。
    """
    if limit <= 0:
        return []
    ensure_memory_ids(npc)
    memories = [memory for memory in npc.memories if memory.accessible]
    if not memories:
        return []
    selected: list[Memory] = []
    selected_ids: set[str] = set()

    def add(memory: Memory) -> None:
        if memory.id not in selected_ids and len(selected) < limit:
            selected.append(memory)
            selected_ids.add(memory.id)

    for memory in memories[-3:]:
        add(memory)
    by_id = {memory.id: index for index, memory in enumerate(memories)}
    for focus_id in focus_ids or []:
        index = by_id.get(str(focus_id))
        if index is None:
            continue
        for nearby in range(max(0, index - 1), min(len(memories), index + 2)):
            add(memories[nearby])

    def score(memory: Memory) -> float:
        searchable = (memory.content + " " + " ".join(memory.keywords)).strip()
        relevance = text_similarity(query, searchable) if query else 0.0
        return memory_effectiveness(memory, now) + 4.0 * relevance

    for memory in sorted(memories, key=score, reverse=True):
        add(memory)
        if len(selected) >= limit:
            break
    return selected


def experience_payload(memories: list["Memory"]) -> list[dict]:
    """人物经历进入裁决载荷的稳定形状；id 可被模型引用为回忆焦点。"""
    return [{"id": memory.id, "turn": memory.turn,
             "occurred_clock": memory.occurred_clock,
             "embodied_as": memory.embodied_as,
             "content": memory.content, "importance": memory.importance,
             "keywords": list(memory.keywords)}
            for memory in memories]


def memory_gaps_payload(npc: "NPC", limit: int = 8) -> list[dict]:
    """断档只有时间与原因，没有被借用期间的事件内容。"""
    return [{"id": str(gap.get("id", "")),
             "started_clock": gap.get("started_clock"),
             "ended_clock": gap.get("ended_clock")}
            for gap in npc.memory_gaps[-max(0, limit):]]

# ---------------- 场景级记忆（变化 + 关联） ----------------

SCENE_RECENT_CAP = 8      # 场景变化记忆窗口上限
ASSOC_CAP = 9.0           # 场景关联强度上限（个位数，够用）

# 事件触达的场景（按 payload 取位置）
_SCENE_TRACE_KINDS = {
    "scene_generated": lambda p: [p.get("scene")],
    "scene_extended": lambda p: [p.get("from"), p.get("scene")],
    "item_arrive": lambda p: [p.get("location")],
    "world_event": lambda p: [p.get("location")],
    "npc_moved": lambda p: [p.get("from"), p.get("to")],
    "npc_interaction": lambda p: [p.get("location")],
    # npc_acted.location 是承诺的目的地；动作开始只触达出发现场。
    # 旧事件没有 origin 时回退到历史 location，保持旧档可读。
    "npc_acted": lambda p: [p.get("origin") or p.get("location")],
    "action_done": lambda p: [p.get("location")],
    "note_left": lambda p: [p.get("location")],
    "item_added": lambda p: [p.get("location")],
    "item_removed": lambda p: [p.get("location")],
    "item_changed": lambda p: [p.get("location")],
    "scene_state_changed": lambda p: [p.get("scene")],
    "daily_life": lambda p: [p.get("location")],
}

# 这些不是世界观的事件分类，而是调度层的唤醒条件：一条已经提交的
# 外部状态变化，应让真正被它碰到的人有机会在下一次脉冲重新判断。
# 其余事件（记忆、目标进度、日常文字）不唤醒，避免角色彼此递归惊动。
_WAKEUP_KINDS = {
    "npc_moved", "action_done", "action_aborted", "npc_state_changed",
    "item_arrive", "item_added", "item_removed", "item_transfer",
    "item_changed", "world_event", "weather_shift", "scene_entered",
    "player_acted",
    "npc_state_fact_changed",
    "agency_changed",
}

# 事件把两个场景扯上关系的配对
_LINK_PAIR_KINDS = {
    "scene_extended": lambda p: (p.get("from"), p.get("scene")),
    "npc_moved": lambda p: (p.get("from"), p.get("to")),
}


def scene_associations(world: "World", sid: str, top: int = 3
                       ) -> list[tuple["Scene", float]]:
    """场景的关联记忆：与它被事件扯上关系的场景，按强度排序。"""
    pairs: list[tuple[Scene, float]] = []
    for key, strength in world.associations.items():
        a, b = key.split("|")
        other = b if a == sid else (a if b == sid else None)
        if other is not None and other in world.scenes:
            pairs.append((world.scenes[other], strength))
    pairs.sort(key=lambda x: -x[1])
    return pairs[:top]


ACTIVE_ITEMS = 10  # 物品读取有界：活跃物品窗口
ITEM_MAX_IDLE = 96  # 物品无人提及 96 回合（4 天）后自然消逝


def active_items(scene: "Scene", top: int = ACTIVE_ITEMS) -> list[dict]:
    """物品表读取有界：取最近活跃且未被持有的物品（表尾 = 最近写入）。

    存储不设上限（世界不丢东西），读取只带活跃窗口——
    其余聚合为「以及一些杂物」，按需再展开。
    被持有的物品（held_by 非空）跟着持有者走，不在场景物品表露面。
    """
    items = [i for i in scene.items if not i.get("held_by")]
    if len(items) <= top:
        return items
    return items[-top:]


def touch_items(scene: "Scene", turn: int) -> None:
    """读取即保鲜：物品被带进检索切片时，刷新它的活跃时间。"""
    for item in scene.items:
        item["last_turn"] = turn


def mood_now(world: "World") -> str:
    """世界的情绪：AI 的自由形容（空 = 平静）。

    情绪词不是引擎的词典——由 AI 在心跳里按世界状态自由提案。
    """
    return world.mood_word or "平静"


def weather_now(world: "World") -> str:
    """世界的天气：纯天气层，与情绪无关。

    天气状态由 weather_shift 自由改写（雷雨/雾/晴）；强度是数值，
    不带固定形容词——质感交给世界自己的文本。
    """
    return world.weather or world.law_profile.atmosphere


def resolve_target(world: "World", t: str) -> str | None:
    """把目标引用解析成规范实体引用（ids 是语法，名字是词汇）。

    item:i-letter / scene:s-station / npc:n-arin 直接验真；
    裸名按 物品名 → 场景名 → NPC名 回退解析。
    解析不到 = 无效引用 → 返回 None。
    """
    t = str(t).strip()
    if t == "player":
        return "player"
    if ":" in t:
        kind, _, ref = t.partition(":")
        if kind == "item":
            for s in world.scenes.values():
                if any(i.get("id") == ref for i in s.items):
                    return f"item:{ref}"
            return None
        if kind == "scene":
            return f"scene:{ref}" if ref in world.scenes else None
        if kind == "npc":
            return f"npc:{ref}" if ref in world.npcs else None
        return None
    for s in world.scenes.values():
        for i in s.items:
            if i.get("name") == t:
                return f"item:{i.get('id')}"
    for s in world.scenes.values():
        if s.name == t:
            return f"scene:{s.id}"
    for n in world.npcs.values():
        if n.name == t:
            return f"npc:{n.id}"
    return None


def canonical_targets(world: "World", targets: list) -> list[str]:
    """写入时规范化：能解析成 id 的引用落成 id 形式；无效引用直接丢弃。

    无效引用运行时也被忽略（不参与碰撞与快照），落库时就不该存着。
    """
    out: list[str] = []
    for t in targets or []:
        if not isinstance(t, str):
            continue
        resolved = resolve_target(world, t)
        if resolved:
            out.append(resolved)
    return out[:3]


def target_snapshot(world: "World", resolved: str) -> str | None:
    """被引实体的当前快照（一句话）：裁决者读得到自己盯着的那个东西。

    引用不只是碰撞的键——也是 NPC 的眼睛。物品给 note 与所在场景，
    场景给在场人数，NPC 给位置与情绪。
    """
    if resolved == "player":
        return (f"玩家位于 {world.player.get('location', '?')}，"
                f"状态：{world.player.get('condition', '') or '正常'}")
    kind, _, ref = resolved.partition(":")
    if kind == "item":
        for s in world.scenes.values():
            for i in s.items:
                if i.get("id") == ref:
                    note = str(i.get("note", "")).strip()
                    return (f"「{i.get('name', ref)}」在「{s.name}」"
                            + (f"，{note}" if note else ""))
        return None
    if kind == "scene":
        s = world.scenes.get(ref)
        return (f"「{s.name}」在场 {len(s.npcs)} 人") if s else None
    if kind == "npc":
        n = world.npcs.get(ref)
        return (f"「{n.name}」在「"
                f"{world.scenes.get(n.state.location).name
                 if n.state.location in world.scenes else '某处'}」"
                f"，{n.state.mood}") if n else None
    return None


def scene_changes(world: "World", scene_id: str, since_turn: int,
                  limit: int = 5) -> list[dict]:
    """重返场景视图：从事件账本机械导出 since_turn 以来的事实变化。

    零 LLM、零编造——每条变化就是账本里的事件原文，
    附事件 id（turn:序号）可回放。玩家离开再回来，看见的是
    具体什么变了、为什么变，而不是更多文本。
    """
    scene = world.scenes.get(scene_id)
    if scene is None or limit <= 0:
        return []
    out: list[dict] = []
    for idx in range(len(world.events) - 1, -1, -1):
        e = world.events[idx]
        if e.turn <= since_turn:
            break  # 再往前全是 since 之前的事件
        params = e.payload.get("event_params", {})
        hit = False
        if e.kind in ("world_event", "item_arrive", "item_removed",
                      "item_changed", "scene_extended", "scene_generated"):
            hit = (params.get("location") == scene_id
                   or params.get("scene") == scene_id
                   or params.get("from") == scene_id)
        elif e.kind == "npc_moved":
            hit = (params.get("from") == scene_id
                   or params.get("to") == scene_id)
        elif e.kind in ("daily_life", "scene_state_changed"):
            hit = params.get("location") == scene_id
            if e.kind == "scene_state_changed":
                hit = params.get("scene") == scene_id
        elif e.kind == "collision":
            hit = (params.get("a") in scene.npcs
                   or params.get("b") in scene.npcs)
        if hit:
            out.append({
                "event_id": f"{e.turn}:{idx}",
                "kind": e.kind,
                "fact": e.summary,
                "cause": e.cause,
            })
            if len(out) >= limit:
                break  # 已凑齐最近 limit 条：从尾部扫，天然是最新的
    return list(reversed(out))


# ---------------- 事件日志 ----------------


@dataclass
class Event:
    turn: int
    kind: str  # world_created | law_changed | dialogue | door_crossed | scene_entered | ...
    summary: str
    cause: str  # 原因引用：哪次命令 / 交互触发
    payload: dict = field(default_factory=dict)
    day: float = 0.0       # 发生时的钟点戳（世界钟，天）
    duration: float = 0.0  # 这件事让世界走了多久（天）——时间成本


# ---------------- 法则档案 ----------------


@dataclass
class Law:
    id: str
    trigger: str  # 触发条件（自然语言）
    effect: str   # 触发后果（自然语言）
    intensity: float  # 0.0 - 1.0

    @staticmethod
    def from_dict(d: dict) -> "Law":
        # 宽容解析：真模型偶尔漏字段，缺省值兜底，包络线随后把关
        return Law(id=str(d.get("id", "")),
                   trigger=str(d.get("trigger", "")),
                   effect=str(d.get("effect", "")),
                   intensity=float(d.get("intensity", 0.5)))


@dataclass
class LawProfile:
    expectation: str  # 期望文本——法则的「为什么」
    atmosphere: str   # 氛围基调（如「雨·永续」）
    laws: list[Law]   # 法则条目
    version: int = 0  # 天变计数

    @staticmethod
    def from_dict(d: dict) -> "LawProfile":
        return LawProfile(
            expectation=d["expectation"],
            atmosphere=d["atmosphere"],
            laws=[Law.from_dict(x) for x in d.get("laws", [])],
            version=int(d.get("version", 0)),
        )


# ---------------- 场景 / NPC ----------------


@dataclass
class Memory:
    turn: int
    content: str
    importance: float = 0.5  # 0.0-1.0：玩家设定 > 情感对话 > 日常
    keywords: list[str] = field(default_factory=list)  # 写入时的检索关键词
    id: str = ""  # 人物档案内稳定引用；旧档读取时自动补齐
    # 记住它的时刻不等于它发生的时刻；未知就保留 None，不伪造年份。
    occurred_clock: float | None = None
    # 这段叙述已经投影到哪些账本事实；内容是来源索引，不是另一份状态。
    projections: list[dict] = field(default_factory=list)
    # 档案存在与当前能想起是两回事。失忆只关闭访问，不删除过去。
    accessible: bool = True
    access_cause: str = ""
    # 经历发生时使用的身体实体；空表示自己的身体（旧档兼容）。
    embodied_as: str = ""

    @staticmethod
    def from_dict(d: dict) -> "Memory":
        return Memory(turn=int(d.get("turn", 0)), content=d["content"],
                      importance=float(d.get("importance", 0.5)),
                      keywords=list(d.get("keywords", [])),
                      id=str(d.get("id", "")),
                      occurred_clock=(float(d["occurred_clock"])
                                      if d.get("occurred_clock") is not None
                                      else None),
                      projections=[dict(p) for p in
                                   (d.get("projections") or
                                    ([d["projection"]]
                                     if isinstance(d.get("projection"), dict)
                                     else []))
                                   if isinstance(p, dict)],
                      accessible=bool(d.get("accessible", True)),
                      access_cause=str(d.get("access_cause", "")),
                      embodied_as=str(d.get("embodied_as", "")))


@dataclass
class Scene:
    id: str
    name: str
    description: str  # 贴片描述（空 = 雾中未生成）
    atmosphere: str
    npcs: list[str] = field(default_factory=list)   # 在场的 NPC id
    exits: list[str] = field(default_factory=list)  # 相邻场景 id
    hint: str = ""               # DNA 里的一句话线索（贴片生成用）
    generated: bool = True       # 贴片是否已生成
    memory_only: bool = False    # 只在往事中有坐标；尚未声明与当前地图连通
    recent: list[dict] = field(default_factory=list)  # 场景变化记忆（有界窗口）
    items: list[dict] = field(default_factory=list)  # 物品表：场景一等状态
    state_facts: list[dict] = field(default_factory=list)  # 局部环境状态
    crowd: str = ""              # 人流（背景层）：一行文本，零成本的人口纹理
    # 个人往事对这个地点的历史声明；不直接冒充场景今天的 description。
    history: list[dict] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "Scene":
        # 宽容解析：真模型漏字段给缺省值，包络线/调用点随后把关
        return Scene(
            id=str(d.get("id", "")), name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            atmosphere=d.get("atmosphere", ""),
            npcs=list(d.get("npcs", [])), exits=list(d.get("exits", [])),
            hint=d.get("hint", ""),
            generated=bool(d.get("generated", True)),
            memory_only=bool(d.get("memory_only", False)),
            recent=[dict(r) for r in d.get("recent", [])],
            items=[dict(i) for i in d.get("items", [])],
            state_facts=[dict(f) for f in d.get("state_facts", [])
                         if isinstance(f, dict)],
            crowd=str(d.get("crowd", "")),
            history=[dict(h) for h in d.get("history", [])
                     if isinstance(h, dict)],
        )


@dataclass
class ActionState:
    """NPC 进行中的主动动作：可被看见、追上、聊起，随时间推进。"""
    text: str = ""          # 动作描述（如「在车站打听那封信的下落」）
    location: str = ""      # 动作发生地（场景 id）
    # 动作开始时的行动者。身体可以在动作完成前归位或再次换人，
    # 物理后果仍落在身体上，但经历必须回到启动者；空值兼容旧存档。
    actor_id: str = ""
    # 新动作以绝对世界钟为准。progress 只保留为旧存档迁移和界面投影，
    # 不再是调度依据。
    started_clock: float = 0.0
    due_clock: float = 0.0
    # 可选的世界钟窗口：动作可以被提前承诺，但只能在窗口内开始/完成。
    # 0 表示未声明，兼容旧存档与旧模型。
    earliest_clock: float = 0.0
    latest_clock: float = 0.0
    targets: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    source_turn: int = 0
    progress: float = 0.0

    @staticmethod
    def from_dict(d: dict) -> "ActionState":
        return ActionState(
            text=str(d.get("text", "")),
            location=str(d.get("location", "")),
            actor_id=str(d.get("actor_id", "")),
            started_clock=float(d.get("started_clock", 0.0)),
            due_clock=float(d.get("due_clock", 0.0)),
            earliest_clock=float(d.get("earliest_clock", 0.0)),
            latest_clock=float(d.get("latest_clock", 0.0)),
            targets=[str(t) for t in d.get("targets", [])
                     if isinstance(t, str)],
            requires=[str(t) for t in d.get("requires", [])
                      if isinstance(t, str)],
            source_turn=int(d.get("source_turn", 0)),
            progress=float(d.get("progress", 0.0)),
        )


@dataclass
class IntentState:
    """NPC 当前短期打算：可改变、可放下，但本身不改变世界事实。"""
    text: str = ""
    targets: list[str] = field(default_factory=list)
    earliest_clock: float = 0.0
    latest_clock: float = 0.0
    since_turn: int = 0

    @staticmethod
    def from_dict(d: dict) -> "IntentState":
        return IntentState(
            text=str(d.get("text", "")),
            targets=[str(t) for t in d.get("targets", []) if isinstance(t, str)],
            earliest_clock=float(d.get("earliest_clock", 0.0)),
            latest_clock=float(d.get("latest_clock", 0.0)),
            since_turn=int(d.get("since_turn", 0)),
        )


@dataclass
class NPCState:
    """NPC 状态快照：位置、活动、情绪与能否继续作为行动者。"""
    location: str = ""     # 所在场景 id
    activity: str = "待机"  # 正在做什么
    mood: str = "平静"      # 情绪标签（作息气质 + 情绪覆盖）
    mood_value: float = 0.0  # 情绪极性强度 -1..1（事件驱动，时间衰减）
    mood_reason: str = ""    # 最近一次情绪变化的原因（可审计）
    last_turn: int = 0     # 最后演化到的回合
    last_clock: float = 0.0  # 最后演化到的世界钟（角色调度唯一时间基准）
    last_time: str = ""    # 游戏内时间戳（人可读）
    pending_opener: str = ""  # 主动开口的台词（待玩家回应，聊过即清空）
    memory_focus: list[str] = field(default_factory=list)  # 正在深挖的经历 id
    # 仍对现在成立的自由文本事实：伤口、伤疤、债务、身份等共用一个语法。
    # review_clock 只表示何时应重新裁决，不预设它届时如何变化。
    facts: list[dict] = field(default_factory=list)
    action: ActionState = field(default_factory=ActionState)  # 进行中的主动动作
    intent: IntentState = field(default_factory=IntentState)  # 短期打算，不等于已发生
    # 这是操作事实，不是「死亡/失踪/封印」的枚举。世界用 condition 写
    # 具体含义；引擎只据此决定该实体还能不能产生新的行动。
    can_act: bool = True
    condition: str = ""

    @staticmethod
    def from_dict(d: dict) -> "NPCState":
        return NPCState(
            location=d.get("location", ""),
            activity=d.get("activity", "待机"),
            mood=d.get("mood", "平静"),
            mood_value=float(d.get("mood_value", 0.0)),
            mood_reason=d.get("mood_reason", ""),
            last_turn=int(d.get("last_turn", 0)),
            last_clock=float(d.get("last_clock", 0.0)),
            last_time=d.get("last_time", ""),
            pending_opener=d.get("pending_opener", ""),
            memory_focus=[str(x) for x in d.get("memory_focus", [])][:4],
            facts=[dict(f) for f in d.get("facts", [])
                   if isinstance(f, dict)],
            action=ActionState.from_dict(d["action"]) if d.get("action")
            else ActionState(),
            intent=IntentState.from_dict(d["intent"]) if d.get("intent")
            else IntentState(),
            can_act=bool(d.get("can_act", True)),
            condition=str(d.get("condition", "")),
        )

    def mark(self, turn: int, clock: float | None = None) -> None:
        """演化打卡：回合 + 游戏内时间戳一起写。"""
        self.last_turn = turn
        if clock is None:
            self.last_time = game_time(turn)
            return
        self.last_clock = clock
        day = int(clock) + 1
        phase = int((clock % 1.0) * 4) % 4
        self.last_time = f"第{day}天·{PHASE_NAMES[phase]}"


@dataclass
class NPC:
    id: str
    name: str
    persona: str  # 出生/导入时的性格与背景起点，不是永久行为指令
    traits: dict = field(default_factory=dict)  # 初始气质线索；经历可以覆盖它
    memories: list[Memory] = field(default_factory=list)  # 不删除的个人经历档案
    # 身体被另一行动者使用时，身体主人只得到时间断档，不得到行动内容。
    memory_gaps: list[dict] = field(default_factory=list)
    beliefs: list[str] = field(default_factory=list)  # 蒸馏出的执念/信念
    goals: list[dict] = field(default_factory=list)  # 驱动力：{id, text, progress}
    relationship: int = 0  # 与玩家的关系值 -100..100
    state: NPCState = field(default_factory=NPCState)  # 状态快照
    in_fog: bool = False    # 退入雾中：存在但不在活跃生活（可归来）
    fog_note: str = ""      # 淡出时的去向与缘由（AI 自由文本）
    links: dict = field(default_factory=dict)  # 与其他 NPC 的关系值 -1..1

    @staticmethod
    def from_dict(d: dict) -> "NPC":
        # 宽容解析：真模型漏字段给缺省值
        return NPC(
            id=str(d.get("id", "")), name=str(d.get("name", "")),
            persona=d.get("persona", ""),
            traits=dict(d.get("traits", {})),
            memories=[Memory(turn=0, content=str(m))
                      if isinstance(m, str) else Memory.from_dict(m)
                      for m in d.get("memories", [])],
            memory_gaps=[dict(g) for g in d.get("memory_gaps", [])
                         if isinstance(g, dict)],
            beliefs=list(d.get("beliefs", [])),
            goals=[dict(g) for g in d.get("goals", [])],
            relationship=int(d.get("relationship", 0)),
            in_fog=bool(d.get("in_fog", False)),
            fog_note=str(d.get("fog_note", "")),
            links={k: float(v) for k, v in (d.get("links") or {}).items()},
            state=NPCState.from_dict(d["state"]) if d.get("state") else
            NPCState(),
        )


@dataclass
class Door:
    id: str
    to_world: str
    to_scene: str
    note: str = ""

    @staticmethod
    def from_dict(d: dict) -> "Door":
        return Door(id=d["id"], to_world=d["to_world"],
                    to_scene=d["to_scene"], note=d.get("note", ""))


# ---------------- 世界 ----------------


@dataclass
class World:
    name: str
    description: str  # 创建时的原始描述（期望文本）
    law_profile: LawProfile
    scenes: dict[str, Scene] = field(default_factory=dict)
    npcs: dict[str, NPC] = field(default_factory=dict)
    doors: dict[str, Door] = field(default_factory=dict)
    player: dict = field(default_factory=dict)  # {location, notes}
    events: list[Event] = field(default_factory=list)  # append-only
    social: dict = field(default_factory=dict)  # NPC 对的最近互动回合
    social_clock: dict[str, float] = field(default_factory=dict)
    # social 的值还有碰撞签名等非时间信息；冷却时间单独存，不能混用。
    # 调度提示，不是角色状态或世界设定。值是尚未处理的触发事件回合。
    # 事件本身仍是唯一事实来源；队列丢失时只会延后一次裁决，不会改写世界。
    wakeups: dict[str, int] = field(default_factory=dict)
    pulse_last_turn: int = 0  # 上次间隔心跳的回合
    pulse_last_clock: float = 0.0  # 上次统一脉冲的世界钟
    turn: int = 0
    associations: dict[str, float] = field(default_factory=dict)  # 场景关联强度
    mood_value: float = 0.0  # 世界氛围值 -1..1（事件驱动，时间回温）
    mood_reason: str = ""    # 最近一次氛围变化的原因（可审计）
    weather: str = ""           # 天气状态（如「雨·永续」；空 = 回落到基调）
    weather_intensity: float = 0.3  # 天气强度 0..1（缓 → 烈）
    weather_reason: str = ""   # 最近一次天气变化的原因
    mood_word: str = ""        # 世界情绪的自由形容（AI 提案；空 = 平静）
    facts: list = field(default_factory=list)  # 世界档案：AI 生成时归纳的
    # 稳定归属/规则条目（自由文本，如「衣服跟着身体走」）——给裁决的锚
    moments: list = field(default_factory=list)  # 既定时刻：{"due_day": N,
    # "what": 到那时必然发生的事, "done": false}——时刻锚，到点必发
    clock: float = 0.0        # 世界钟（天）：客观时间的唯一账本，只随事件累加
    heartbeat: float = 1.0 / 24.0  # 心跳间隔（天）：隐性地但恒定的时间线
    # 记忆提到的物品档案。当前确实在场的物品仍以 Scene.items 为现状；
    # 丢失或去向未知的物品也保留身份与历史，不被伪装成眼前可用物品。
    past_items: dict[str, dict] = field(default_factory=dict)
    # 当前行动主体映射：body npc id -> {actor, started_clock, until_clock, cause}。
    # 身体保存位置与物态，行动者提供经历、信念和决策。空表表示每个人
    # 通过自己的身体行动；它不是互换枚举，也不包含任何题材词汇。
    agency: dict[str, dict] = field(default_factory=dict)
    # 世界的粒度：快世界大、慢世界小——语法一个数，词汇交给世界

    # ---- 世界钟：时间由运动决定 ----

    @property
    def day(self) -> int:
        """第几天（1 起）——由钟算，不由回合算。"""
        return int(self.clock) + 1

    @property
    def phase(self) -> int:
        """0=清晨 1=白昼 2=黄昏 3=深夜——由钟的小数部分定。"""
        return int((self.clock % 1.0) * 4) % 4

    def now(self) -> str:
        """人可读的当前时刻。"""
        return f"第{self.day}天·{PHASE_NAMES[self.phase]}"

    # ---- 事件与记忆的唯一入口：一切变化必须经过这里 ----

    def log(self, kind: str, summary: str, cause: str,
            payload: dict | None = None, duration: float = 0.0) -> int:
        """写入事件日志并推进世界钟。返回新 turn。

        duration 是这件事的时间成本（天）：事件发生时盖钟点戳，
        然后钟往前走 duration——时间由运动决定，无事发生则钟不走。

        同时维护两块场景级记忆（世界状态不存历史快照，
        只随事件节点做部分更新）：
        - 场景变化记忆：事件触达的场景记下有界窗口。
        - 场景关联记忆：事件把两个场景扯上关系时强化关联强度。
        """
        self.turn += 1
        day_stamp = self.clock
        self.clock += max(0.0, float(duration))
        self.events.append(Event(turn=self.turn, kind=kind, summary=summary,
                                 cause=cause, payload=payload or {},
                                 day=day_stamp, duration=duration))

        p = payload or {}
        if "event_params" in p:  # emit 包了一层，索引用里层
            p = p["event_params"]
        locs = [l for l in _SCENE_TRACE_KINDS.get(kind, lambda q: [])(p)
                if l]
        for sid in locs:
            scene = self.scenes.get(sid)
            if scene is not None:
                scene.recent.append({"turn": self.turn, "kind": kind,
                                     "summary": summary})
                if len(scene.recent) > SCENE_RECENT_CAP:
                    del scene.recent[:len(scene.recent) - SCENE_RECENT_CAP]
        pair = _LINK_PAIR_KINDS.get(kind, lambda q: None)(p)
        if pair and pair[0] and pair[1] and pair[0] != pair[1]:
            key = "|".join(sorted((pair[0], pair[1])))
            self.associations[key] = min(
                ASSOC_CAP, self.associations.get(key, 0.0) + 1.0)
        self._queue_wakeups(kind, p, locs)
        return self.turn

    def _queue_wakeups(self, kind: str, params: dict,
                       locations: list[str]) -> None:
        """把已提交事实的直接参与者、现场者和关注者排入下次裁决。

        这只决定谁更早获得一次思考机会，不决定他们必须采取什么行动。
        因此「抵达后沉默」「物品变化但无人理会」仍是合法结果。
        """
        if kind not in _WAKEUP_KINDS:
            return
        candidates: set[str] = set()
        for key in ("npc", "target", "a", "b"):
            value = str(params.get(key, ""))
            if value in self.npcs:
                candidates.add(value)
        refs = (list(params.get("refs") or [])
                + list(params.get("targets") or [])
                + list(params.get("requires") or []))
        for ref in refs:
            resolved = resolve_target(self, str(ref))
            if resolved and resolved.startswith("npc:"):
                candidates.add(resolved[4:])
        for sid in locations:
            scene = self.scenes.get(sid)
            if scene is not None:
                candidates.update(scene.npcs)
        # 无地点的天气变化是全局事实；其余无地点事件只唤醒直接参与者。
        if kind == "weather_shift":
            candidates.update(self.npcs)

        changed_items = {
            f"item:{params.get('item')}" for _ in [0]
            if params.get("item")
        }
        for npc in self.npcs.values():
            actor = self.actor_for_body(npc)
            watched = {
                str(target) for goal in actor.goals
                if float(goal.get("progress", 0.0)) < 1.0
                for target in goal.get("targets", [])
            }
            watched.update(actor.state.intent.targets)
            watched.update(npc.state.action.targets)
            watched.update(npc.state.action.requires)
            if watched & changed_items:
                candidates.add(npc.id)

        for npc_id in candidates:
            npc = self.npcs.get(npc_id)
            # 雾中且没有任何目标的人是背景，不因远处事件被强行拉回细演。
            scene = self.scenes.get(npc.state.location) if npc is not None else None
            quiet_in_fog = npc is not None and not npc.goals and (
                npc.in_fog or (scene is not None and not scene.generated))
            if npc is not None and not quiet_in_fog:
                self.wakeups[npc_id] = self.turn

    def remember(self, npc: NPC, content: str, cause: str,
                 kind: str = "dialogue", importance: float = 0.5,
                 cap: int | None = None,
                 keywords: list[str] | None = None,
                 body: NPC | str | None = None,
                 started_clock: float | None = None,
                 ended_clock: float | None = None,
                 record_gap: bool = True) -> int:
        """NPC 写入式经历：只追加、不删除；cap 仅为旧调用兼容保留。

        人物一生可以持续变厚。每次裁决通过 experience_window 限制注意力，
        不能用删除过去来控制提示词成本。
        """
        body_id = body.id if isinstance(body, NPC) else str(body or npc.id)
        if body_id not in self.npcs:
            raise ValueError(f"记忆归属引用不存在的身体实体：{body_id}")
        start = self.clock if started_clock is None else float(started_clock)
        end = self.clock if ended_clock is None else float(ended_clock)
        if end < start:
            raise ValueError("记忆归属的结束时间不能早于开始时间")
        turn = self.log(kind, f"{npc.name} 记住了：{content}", cause,
                        {"npc": npc.id, "actor": npc.id, "body": body_id,
                         "started_clock": start, "ended_clock": end})
        npc.memories.append(Memory(turn=turn, content=content,
                                   importance=importance,
                                   keywords=list(keywords or []),
                                   id=f"m-{turn}", embodied_as=body_id))
        if body_id != npc.id and record_gap:
            self.record_memory_gap(
                self.npcs[body_id], actor=npc.id, body=body_id,
                started_clock=start, ended_clock=end, cause=cause,
                source_memory=f"m-{turn}")
        return turn

    def actor_for_body(self, body: NPC | str) -> NPC:
        """返回此刻通过该身体行动的人；无映射时就是身体主人。"""
        body_id = body.id if isinstance(body, NPC) else str(body)
        owner = self.npcs.get(body_id)
        if owner is None:
            raise ValueError(f"行动身体不存在：{body_id}")
        binding = self.agency.get(body_id)
        actor_id = str((binding or {}).get("actor", body_id))
        return self.npcs.get(actor_id, owner)

    def remember_as(self, body: NPC, content: str, cause: str,
                    kind: str = "dialogue", importance: float = 0.5,
                    keywords: list[str] | None = None) -> int:
        """把经由身体发生的经历写给当前行动者，不泄给身体主人。"""
        actor = self.actor_for_body(body)
        binding = self.agency.get(body.id)
        started = float((binding or {}).get("started_clock", self.clock))
        return self.remember(
            actor, content, cause, kind=kind, importance=importance,
            keywords=keywords, body=body, started_clock=started,
            ended_clock=self.clock,
            # 运行态在结束时写一段完整断档，不按每条经历切碎。
            record_gap=not bool(binding and actor.id != body.id),
        )

    def _release_agency(self, body_id: str, cause: str,
                        ended_clock: float | None = None) -> list[str]:
        binding = self.agency.pop(body_id, None)
        if not isinstance(binding, dict):
            return []
        actor_id = str(binding.get("actor", body_id))
        started = float(binding.get("started_clock", self.clock))
        ended = self.clock if ended_clock is None else float(ended_clock)
        ended = max(started, ended)
        owner = self.npcs.get(body_id)
        summaries: list[str] = []
        if owner is not None and actor_id != body_id:
            self.record_memory_gap(
                owner, actor=actor_id, body=body_id,
                started_clock=started, ended_clock=ended, cause=cause,
                source_memory=f"agency:{body_id}:{started:.6f}")
            summaries.append(self.events[-1].summary)
        self.log(
            "agency_changed",
            f"{owner.name if owner else body_id} 的行动主体恢复为身体主人",
            cause,
            {"npc": body_id, "body": body_id, "actor": body_id,
             "previous_actor": actor_id, "started_clock": started,
             "ended_clock": ended},
        )
        summaries.append(self.events[-1].summary)
        # 归属变化本身也是行动者的经历。身体主人只保留上面的断档，
        # 不复制这条私人记忆或借身期间的具体内容。
        if owner is not None and actor_id != body_id:
            actor = self.npcs.get(actor_id)
            if actor is not None:
                self.remember(
                    actor,
                    f"我结束了通过{owner.name}的身体行动，回到自己的身体。",
                    cause=cause, kind="npc_memory", importance=0.7,
                    body=owner, started_clock=ended, ended_clock=ended,
                    record_gap=False)
                summaries.append(self.events[-1].summary)
        return summaries

    def apply_agency_patches(self, patches: list[dict]
                             ) -> tuple[list[str], set[str]]:
        """原子校验后提交行动主体映射；同轮可表达交换或单向借用。"""
        if not isinstance(patches, list):
            return ["驳回行动主体映射：映射批次必须是数组"], set()
        normalized: list[tuple[str, str, float, str]] = []
        errors: list[str] = []
        seen_bodies: set[str] = set()
        for patch in patches[:8]:
            if not isinstance(patch, dict):
                errors.append("行动主体映射必须是对象")
                continue
            body_id = str(patch.get("body", "")).removeprefix("npc:")
            actor_id = str(patch.get("actor", "")).removeprefix("npc:")
            why = str(patch.get("why", "")).strip()
            try:
                until = float(patch.get("until_clock", 0.0) or 0.0)
            except (TypeError, ValueError):
                until = -1.0
            if body_id not in self.npcs or actor_id not in self.npcs:
                errors.append("行动主体映射引用了不存在的角色")
            elif body_id in seen_bodies:
                errors.append(f"同一身体被重复映射：{body_id}")
            elif not why:
                errors.append("行动主体映射必须说明原因")
            elif actor_id != body_id and until <= self.clock:
                errors.append("行动主体映射的结束钟点必须晚于当前时刻")
            else:
                seen_bodies.add(body_id)
                normalized.append((body_id, actor_id, until, why))
        if errors:
            return [f"驳回行动主体映射：{error}" for error in errors], set()

        summaries: list[str] = []
        changed: set[str] = set()
        for body_id, actor_id, until, why in normalized:
            current_actor = self.actor_for_body(body_id).id
            if current_actor == actor_id:
                continue
            if body_id in self.agency:
                summaries.extend(self._release_agency(body_id, why))
            changed.add(body_id)
            if actor_id == body_id:
                continue
            self.agency[body_id] = {
                "actor": actor_id, "started_clock": self.clock,
                "until_clock": until, "cause": why[:160],
            }
            owner = self.npcs[body_id]
            actor = self.npcs[actor_id]
            # 旧行动和短期承诺属于上一位行动者，不能被新行动者继承。
            # 位置、伤势、持有物等身体状态仍原样保留。
            old_action = owner.state.action
            if old_action.text and old_action.progress < 1.0:
                old_actor = (self.npcs.get(old_action.actor_id)
                             if old_action.actor_id else owner)
                if old_actor is None:
                    old_actor = owner
                self.remember(
                    old_actor, f"我搁下了这件事：{old_action.text[:40]}",
                    cause=why, kind="npc_memory", importance=0.5,
                    body=owner, started_clock=old_action.started_clock,
                    ended_clock=self.clock, record_gap=False)
                self.log(
                    "action_aborted",
                    f"{owner.name} 中止了动作「{old_action.text[:200]}」",
                    why,
                    {"event_params": {
                        "npc": owner.id, "body": owner.id,
                        "actor": old_actor.id,
                        "action": old_action.text[:200],
                    }})
                summaries.append(self.events[-1].summary)
            owner.state.action = ActionState()
            owner.state.intent = IntentState()
            owner.state.pending_opener = ""
            self.log(
                "agency_changed",
                f"{actor.name} 开始通过 {owner.name} 的身体行动",
                why,
                {"npc": body_id, "body": body_id, "actor": actor_id,
                 "started_clock": self.clock, "until_clock": until},
            )
            summaries.append(self.events[-1].summary)
            # 映射建立本身不只是公共账本事实，也是行动者知道的经历。
            # 具体行动仍由 remember_as 按实际事件继续归属给 actor。
            self.remember(
                actor,
                f"我开始通过{owner.name}的身体行动。",
                cause=why, kind="npc_memory", importance=0.7,
                body=owner, started_clock=self.clock,
                ended_clock=self.clock, record_gap=False)
            summaries.append(self.events[-1].summary)
        return summaries, changed

    def expire_agency(self) -> list[str]:
        """世界钟到点后结束映射，并给身体主人留下整段时间断档。"""
        summaries: list[str] = []
        due = [body_id for body_id, binding in self.agency.items()
               if float(binding.get("until_clock", 0.0) or 0.0)
               <= self.clock + 1e-9]
        for body_id in due:
            binding = self.agency.get(body_id, {})
            ended = float(binding.get("until_clock", self.clock) or self.clock)
            summaries.extend(self._release_agency(
                body_id, "行动主体映射到期", ended_clock=ended))
        return summaries

    def record_memory_gap(self, owner: NPC, actor: str, body: str,
                          started_clock: float, ended_clock: float,
                          cause: str, source_memory: str = "") -> str:
        """给身体主人记一段无内容断档；相同来源重复物化时不重复写。"""
        if source_memory and any(g.get("source_memory") == source_memory
                                 for g in owner.memory_gaps):
            return next(str(g.get("id", "")) for g in owner.memory_gaps
                        if g.get("source_memory") == source_memory)
        gap_id = f"gap-{owner.id}-{len(owner.memory_gaps) + 1}"
        owner.memory_gaps.append({
            "id": gap_id, "started_clock": float(started_clock),
            "ended_clock": float(ended_clock), "actor": str(actor),
            "body": str(body), "cause": str(cause)[:160],
            "source_memory": str(source_memory),
        })
        self.log("memory_gap_recorded",
                 f"{owner.name} 的经历出现一段时间断档", cause,
                 {"npc": owner.id, "actor": str(actor), "body": str(body),
                  "gap": gap_id, "started_clock": float(started_clock),
                  "ended_clock": float(ended_clock)})
        return gap_id

    def set_memory_access(self, npc: NPC, memory_ids: list[str],
                          accessible: bool, cause: str) -> list[str]:
        """改变记忆可访问性而不删除档案；每次变化有因、有痕。"""
        wanted = {str(memory_id) for memory_id in memory_ids}
        existing = {memory.id: memory for memory in npc.memories}
        missing = sorted(wanted - set(existing))
        if missing:
            return [f"记忆引用不存在：{memory_id}" for memory_id in missing]
        changed = [existing[memory_id] for memory_id in wanted
                   if existing[memory_id].accessible != bool(accessible)]
        if not changed:
            return []
        for memory in changed:
            memory.accessible = bool(accessible)
            memory.access_cause = str(cause)[:160]
        if not accessible:
            npc.state.memory_focus = [memory_id for memory_id in
                                      npc.state.memory_focus
                                      if memory_id not in wanted]
        self.log("memory_access_changed",
                 f"{npc.name} 的 {len(changed)} 段经历"
                 f"{'重新可被想起' if accessible else '变得无法直接想起'}",
                 cause, {"npc": npc.id,
                         "memories": [m.id for m in changed],
                         "accessible": bool(accessible)})
        return []

    def _scheduled_boundary_crossed(self, start_clock: float,
                                    end_clock: float) -> bool:
        """映射将在本段时间内到期且有归位 moment 时延后释放。"""
        pending_restores = {
            str(patch.get("body", "")).removeprefix("npc:")
            for moment in self.moments if not moment.get("done")
            for patch in (moment.get("agency_patches") or [])
            if isinstance(patch, dict)
            and str(patch.get("body", "")).removeprefix("npc:")
            == str(patch.get("actor", "")).removeprefix("npc:")
        }
        for body_id, binding in self.agency.items():
            try:
                until = float(binding.get("until_clock", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if (body_id in pending_restores
                    and start_clock - 1e-9 < until <= end_clock + 1e-9):
                return True
        return False

    def pass_time(self, turns: int, cause: str = "时间流逝") -> int:
        """时间流逝是世界的一种动作：推进时钟并记入事件日志。

        每 turn 前进一个心跳（隐性地恒定的时间线）：
        钟走 turns × heartbeat 天，日夜节律挂在钟上。
        """
        self.turn += turns
        days = turns * self.heartbeat
        day_stamp = self.clock
        self.clock += days
        self.events.append(Event(turn=self.turn, kind="time_passed",
                                 summary=f"时间流逝 {turns} 回合（{days:.2f} 天）",
                                 cause=cause,
                                 payload={"turns": turns, "days": days},
                                 day=day_stamp, duration=days))
        # 跨过既定时刻时不要在这里抢先释放 agency；否则归位 moment
        # 的 world_event 只能等下一次脉冲才入账，和恢复时间错开。
        if not self._scheduled_boundary_crossed(day_stamp, self.clock):
            self.expire_agency()
        return self.turn

    def sweep_items(self, max_idle: int = ITEM_MAX_IDLE) -> list[str]:
        """自动清理：太久没人提及的物品自然消逝（有因才消失）。

        物品的出生→跃变→消逝闭环：被世界遗忘的东西被时间扫走，
        且消逝本身是一条可回放的事件——不是内存爆炸，也不是凭空消失。
        """
        summaries: list[str] = []
        # 豁免：被世界档案（facts）或活跃目标引用的物品是世界的锚点——
        # 钥匙不会被时间扫走（引用完整性延伸：引用即保鲜）
        protected: set[str] = set()
        for fact in self.facts:
            for scene in self.scenes.values():
                for item in scene.items:
                    if str(item.get("name", "")) in str(fact):
                        protected.add(item.get("id", ""))
        for npc in self.npcs.values():
            for goal in npc.goals:
                if float(goal.get("progress", 0)) >= 1.0:
                    continue
                for t in goal.get("targets", []):
                    r = resolve_target(self, t)
                    if r and r.startswith("item:"):
                        protected.add(r[5:])
        # 持有中的东西仍在角色的生活里；即使一时没有被提及，也不能被
        # 场景清扫误当成无人问津的环境物。
        for scene in self.scenes.values():
            for item in scene.items:
                if item.get("held_by"):
                    protected.add(item.get("id", ""))
                if isinstance(item.get("action"), dict) \
                        and item["action"].get("text"):
                    protected.add(item.get("id", ""))
        for sid, scene in self.scenes.items():
            keep = []
            for item in scene.items:
                idle = self.turn - int(item.get("last_turn", self.turn))
                if idle <= max_idle or item.get("id", "") in protected:
                    keep.append(item)
                    continue
                name = str(item.get("name", item.get("id", "某物")))
                note = "太久没人提起，被时间扫走了"
                self.log("item_removed",
                         f"「{name}」从「{scene.name}」消失（{note}）",
                         "时间流逝",
                         {"event_params": {"item": item.get("id", ""),
                                           "name": name,
                                           "location": sid, "note": note}})
                summaries.append(f"「{name}」从「{scene.name}」消失（{note}）")
            scene.items = keep
        return summaries

    # ---- 持久化 ----

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scenes"] = {k: asdict(v) for k, v in self.scenes.items()}
        d["npcs"] = {k: asdict(v) for k, v in self.npcs.items()}
        d["doors"] = {k: asdict(v) for k, v in self.doors.items()}
        d["law_profile"] = asdict(self.law_profile)
        d["events"] = [asdict(e) for e in self.events]
        return d

    @staticmethod
    def from_dict(d: dict) -> "World":
        turn = int(d.get("turn", 0))
        hb = float(d.get("heartbeat", 1.0 / 24.0))
        # 兼容迁移：旧档没有 clock 时，按旧换算重建——
        # turn / 24 天 = 旧 day_of(turn) 的关系，时间线不丢
        legacy_clock = turn * (1.0 / 24.0)
        world = World(
            name=d["name"], description=d["description"],
            law_profile=LawProfile.from_dict(d["law_profile"]),
            scenes={k: Scene.from_dict(v) for k, v in d.get("scenes", {}).items()},
            npcs={k: NPC.from_dict(v) for k, v in d.get("npcs", {}).items()},
            doors={k: Door.from_dict(v) for k, v in d.get("doors", {}).items()},
            player=dict(d.get("player", {})),
            events=[Event(**e) for e in d.get("events", [])],
            social=dict(d.get("social", {})),
            social_clock={k: float(v) for k, v in
                          (d.get("social_clock") or {}).items()},
            wakeups={k: int(v) for k, v in (d.get("wakeups") or {}).items()},
            pulse_last_turn=int(d.get("pulse_last_turn", 0)),
            pulse_last_clock=float(d.get("pulse_last_clock", 0.0)),
            turn=turn,
            associations={k: float(v)
                          for k, v in d.get("associations", {}).items()},
            mood_value=float(d.get("mood_value", 0.0)),
            mood_reason=str(d.get("mood_reason", "")),
            weather=str(d.get("weather", "")),
            weather_intensity=float(d.get("weather_intensity", 0.3)),
            weather_reason=str(d.get("weather_reason", "")),
            mood_word=str(d.get("mood_word", "")),
            facts=[str(f) for f in d.get("facts", [])][:8],
            moments=[dict(m) for m in d.get("moments", [])][:4],
            clock=float(d.get("clock", legacy_clock)),
            heartbeat=hb,
            past_items={str(k): dict(v) for k, v in
                        (d.get("past_items") or {}).items()
                        if isinstance(v, dict)},
            agency={str(k): dict(v) for k, v in
                    (d.get("agency") or {}).items()
                    if isinstance(v, dict)},
        )
        # 旧档只有事件回合号。不能把它误当真实时间，读档后从当前钟
        # 重新开始调度，避免首次读取凭空补算一整段生活。
        if "pulse_last_clock" not in d:
            world.pulse_last_clock = world.clock
        for npc_id, raw_npc in d.get("npcs", {}).items():
            if "last_clock" not in (raw_npc.get("state") or {}):
                npc = world.npcs.get(npc_id)
                if npc is not None:
                    npc.state.last_clock = world.clock
                    npc.state.mark(world.turn, world.clock)
        if "social_clock" not in d:
            world.social_clock = {k: world.clock for k in world.social}
        return world


# ---------------- 多世界容器 ----------------


@dataclass
class Universe:
    worlds: dict[str, World] = field(default_factory=dict)
    current: str = ""

    @property
    def here(self) -> World:
        return self.worlds[self.current]

    def save(self, path: str | Path = DEFAULT_SAVE_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "worlds": {k: w.to_dict() for k, w in self.worlds.items()},
            "current": self.current,
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    @staticmethod
    def load(path: str | Path = DEFAULT_SAVE_PATH) -> "Universe":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        worlds = {k: World.from_dict(v) for k, v in data["worlds"].items()}
        # worldgen 依赖 store，故在运行时导入，避免模块初始化循环。
        # 读档只验证，绝不替旧档猜测或改写世界事实。
        from .worldgen import validate_world
        for name, world in worlds.items():
            problems = validate_world(world)
            if problems:
                raise ValueError(
                    f"存档世界「{name}」结构校验失败：" + "；".join(problems))
        current = data.get("current", "")
        if worlds and current not in worlds:
            raise ValueError(f"存档当前世界不存在：{current}")
        return Universe(worlds=worlds, current=current)
