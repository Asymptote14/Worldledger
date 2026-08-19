"""引擎基准自检：批量世界描述 → 生成 + 模拟运转 → 不变量报告。

用法：python -m tools.bench
全 Mock 确定性模式，零成本；真模型验证请跑 python -m tools.demo。
"""
from __future__ import annotations

from worldledger import evolution
from worldledger.llm import MockLLM
from worldledger.worldgen import generate_world

DATASET = [
    "一座永远在下雨的小城，人们撒谎时会漏出真心话",
    "一座永不落日的沙漠圣城，火焰只会照亮不会灼伤",
    "一座被大雪封住的钟表之城，人们听到钟声就会忘记时间",
    "一座漂浮在云海上的灯笼镇，说谎的人影子会先离开",
    "普通的小镇，日子平淡",
    "一座建在鲸背上的渔村，歌声能招来海浪",
    "一座图书馆模样的城市，每本书里都住着一个人",
    "一座只有夜晚的城市，星光落在地上会结霜",
]


def simulate(world, llm, turns: int = 96) -> dict:
    """模拟一段无人游玩的时间：世界心跳 + 收尾补算，收集统计。"""
    stats: dict = {"events": {}, "vetoes": 0}
    for _ in range(turns // 6):
        world.pass_time(6)
        for s in evolution.world_pulse(llm, world):
            if "驳回" in s or s.startswith("事件"):
                stats["vetoes"] += 1
    loc = world.player.get("location", "")
    evolution.catch_up_scene(llm, world, loc)
    for e in world.events:
        stats["events"][e.kind] = stats["events"].get(e.kind, 0) + 1
    return stats


def check_invariants(world) -> list[str]:
    problems: list[str] = []
    # 场景成员表 ⇄ NPC 状态位置，双向一致
    for sid, scene in world.scenes.items():
        for nid in scene.npcs:
            npc = world.npcs.get(nid)
            if npc is None or npc.state.location != sid:
                problems.append(f"{nid} 成员表与状态不一致")
    for npc in world.npcs.values():
        scene = world.scenes.get(npc.state.location)
        if scene is not None and npc.id not in scene.npcs:
            problems.append(f"{npc.name} 不在其场景成员表")
    # 事件日志单调递增
    turns = [e.turn for e in world.events]
    if turns != sorted(turns):
        problems.append("事件日志非单调")
    # 数值有界
    for npc in world.npcs.values():
        if not (-1.0 <= npc.state.mood_value <= 1.0):
            problems.append(f"{npc.name} 情绪越界")
        if not (-100 <= npc.relationship <= 100):
            problems.append(f"{npc.name} 关系越界")
        for g in npc.goals:
            if not (0.0 <= float(g.get("progress", 0)) <= 1.0):
                problems.append(f"{npc.name} 目标进度越界")
    return problems


def main() -> None:
    llm = MockLLM()
    print("世界引擎基准自检（Mock 确定性 · 8 个世界 × 96 回合模拟）")
    print("=" * 72)
    all_scene_names: set[str] = set()
    all_npc_names: set[str] = set()
    event_kinds: set[str] = set()
    report = []
    for desc in DATASET:
        a = generate_world(llm, "W", desc)
        b = generate_world(llm, "W", desc)
        deterministic = a.to_dict() == b.to_dict()
        stats = simulate(a, llm)
        problems = check_invariants(a)
        grew = any(e.kind == "scene_extended" for e in a.events)
        names = {s.name for s in a.scenes.values()}
        npcs = {n.name for n in a.npcs.values()}
        all_scene_names |= names
        all_npc_names |= npcs
        event_kinds |= set(stats["events"].keys())
        report.append({
            "desc": desc,
            "atmo": a.law_profile.atmosphere,
            "laws": len(a.law_profile.laws),
            "scenes": len(names),
            "det": deterministic,
            "events": len(a.events),
            "kinds": len(stats["events"]),
            "vetoes": stats["vetoes"],
            "grew": grew,
            "problems": problems,
        })
        print(f"\n■ {desc}")
        print(f"  氛围 {a.law_profile.atmosphere}｜法则 {len(a.law_profile.laws)} 条｜"
              f"场景 {len(names)} 个｜NPC {len(npcs)} 名｜"
              f"生长 {'✓' if grew else '✗'}｜确定性 {'✓' if deterministic else '✗'}")
        print(f"  场景：{'、'.join(sorted(names))}")
        print(f"  NPC：{'、'.join(sorted(npcs))}")
        print(f"  96 回合：事件 {len(a.events)} 条 / {len(stats['events'])} 类｜"
              f"否决 {stats['vetoes']} 次｜不变量问题 {len(problems)} 个")
        if problems:
            for p in problems[:3]:
                print(f"    ⚠ {p}")

    print("\n" + "=" * 72)
    print("总体评估")
    print("=" * 72)
    total_problems = sum(len(r["problems"]) for r in report)
    all_det = all(r["det"] for r in report)
    all_grew = all(r["grew"] for r in report)
    print(f"确定性：{'全部通过' if all_det else '存在失败'}")
    print(f"世界生长：{'全部触发' if all_grew else '存在未触发'}")
    print(f"不变量问题总数：{total_problems}")
    print(f"跨世界场景名去重：{len(all_scene_names)} 个不同地点")
    print(f"跨世界 NPC 名去重：{len(all_npc_names)} 个不同角色")
    print(f"事件类型覆盖：{sorted(event_kinds)}")
    print("\n结论：Mock 夹具按描述关键字落三套模板（雨/沙漠/中性）——")
    print("基准只证明确定性与生长、不变量；描述的多样性由真实模型运行验证")
    print("（同一句话多次生成长出不同世界），不在 Mock 基准范围之内。")


if __name__ == "__main__":
    main()
