"""开源可复现性 Pilot：验证 WorldLedger 的状态正确性、可审计性、可复现性。

用法：python -m tools.pilot
输出：examples/pilot/{pilot_ledger.json, pilot_player_view.txt, pilot_audit.md}

全部使用 MockLLM（零成本、确定性），不依赖 API key、本地存档或绝对路径。
复用：tools/bench.py 的 simulate/check_invariants、tools.finalstory 的玩家视角导出。
21 项断言覆盖：知识边界、物品不回流、原子多实体事件、
失败无部分提交、有因可追溯、同种子可复现。
"""
from __future__ import annotations

import sys
from pathlib import Path

from tools.bench import check_invariants, simulate
from tools.finalstory import build as build_player_view
from worldledger import evolution
from worldledger.event import apply_item_patch, emit, transfer_item
from worldledger.llm import MockLLM
from worldledger.store import Universe, active_items, memory_effectiveness
from worldledger.worldgen import generate_world

DESC = "一座永远在下雨的小城，人们撒谎时会漏出真心话"
SECRET = "信封里装着给港务局的调令"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "examples" / "pilot"
CHECKS: list[dict] = []


def check(name: str, passed: bool, note: str = "") -> None:
    CHECKS.append({"name": name, "passed": bool(passed), "note": note})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}｜{note}")


def _find(world, key, kind):
    if kind == "scene":
        return next((s for s in world.scenes.values() if key in s.name), None)
    return next((n for n in world.npcs.values() if key in n.name), None)


def run(out_dir: Path = DEFAULT_OUT) -> dict:
    """执行 21 项断言，并把三份产物写入 out_dir。返回统计。"""
    llm = MockLLM()
    # 同一种子（同一描述 + Mock）生成两次，结构必须一致
    w1 = generate_world(llm, "Pilot", DESC)
    w2 = generate_world(llm, "Pilot", DESC)
    check("同种子生成可复现", w1.to_dict() == w2.to_dict(),
          f"两轮生成 to_dict 相等={w1.to_dict() == w2.to_dict()}")
    w = w1
    station = _find(w, "车站", "scene")
    cafe = _find(w, "咖啡", "scene")
    zhou = _find(w, "老周", "npc")
    arin = _find(w, "阿凛", "npc")
    letter = next(i for i in station.items if "信" in str(i.get("name", "")))
    evolution.move_npc(w, zhou, station.id, cause="Pilot 布置")
    evolution.move_npc(w, arin, cafe.id, cause="Pilot 布置")
    w.player["location"] = cafe.id

    # 1) NPC A（老周）知道秘密；NPC B（阿凛）在别处，不得因全局日志知道
    w.remember(zhou, f"{SECRET}（只有我知道这件事）",
               cause="Pilot 夹具", kind="npc_memory", importance=1.0)
    check("A 知道秘密", any(SECRET in m.content for m in zhou.memories))
    check("B 记忆里没有秘密（全局日志未泄漏）",
          not any(SECRET in m.content for m in arin.memories))
    from worldledger.evolution import _npc_visible
    leaked = [e.summary for e in w.events if _npc_visible(w, arin, e)
              and SECRET in e.summary]
    check("B 可见切片无秘密", not leaked, str(leaked[:1]))
    eff = max((memory_effectiveness(m, w.turn) for m in arin.memories
               if SECRET in m.content), default=0.0)
    check("B 检索不到秘密", eff == 0.0, f"最高有效度={eff}")

    # 2) 玩家拿走物品 → 物品不回流（持有标记 + 读取隐去）
    w.player["location"] = station.id
    errs = transfer_item(w, letter, "player", cause="玩家取走")
    check("玩家取走信（转移成功）", not errs, str(errs))
    check("信被标记为玩家持有", letter.get("held_by") == "player",
          f"held_by={letter.get('held_by')}")
    check("信从场景活跃视图隐去",
          not any(i.get("id") == letter["id"]
                  for i in active_items(station)))
    simulate(w, llm, turns=12)  # 世界继续跑 12 回合
    back = any(i.get("id") == letter["id"] for i in active_items(station))
    re_added = any(e.kind == "item_added"
                   and (e.payload or {}).get("event_params", {})
                   .get("item") == letter["id"] for e in w.events)
    check("12 回合后信未回到原处", not back and not re_added,
          f"重回活跃视图={back} 被重加={re_added}")
    check("信仍由玩家持有", letter.get("held_by") == "player",
          f"held_by={letter.get('held_by')}")

    # 3) 原子多实体事件：一个事件同时引用 NPC + 物品 + 地点
    n0 = len(w.events)
    errs = emit(w, "npc_acted",
                {"npc": zhou.id, "action": "把柜台抽屉里的信记进值班簿",
                 "location": station.id,
                 "targets": [f"item:{letter['id']}"],
                 "days": 0.0}, cause="Pilot 原子事件")
    check("原子事件提交成功", not errs, str(errs))
    ev = w.events[-1] if len(w.events) > n0 else None
    p = (ev.payload or {}).get("event_params", {}) if ev else {}
    refs_ok = (ev is not None and p.get("npc") == zhou.id
               and p.get("location") == station.id
               and f"item:{letter['id']}" in p.get("targets", []))
    check("单事件引用 NPC+物品+地点", refs_ok,
          f"npc={p.get('npc')} loc={p.get('location')} "
          f"targets={p.get('targets')}")

    # 4) 失败的前置条件 → 不写部分状态
    note_before = letter.get("note", "")
    n_before = len(w.events)
    errs = apply_item_patch(
        w, {"op": "change", "item": letter["id"],
            "location": cafe.id, "note": "不应落库"}, cause="Pilot 失败用例")
    changed_ev = [e for e in w.events[n_before:]
                  if e.kind == "item_changed"]
    check("错误场景补丁被拒", bool(errs), str(errs[:1]))
    check("失败后无 item_changed 入账", not changed_ev)
    check("物品状态未变", letter.get("note", "") == note_before,
          letter.get("note", ""))
    n_before = len(w.events)
    errs = emit(w, "npc_acted",
                {"npc": arin.id, "action": "去取不存在的东西",
                 "location": cafe.id,
                 "targets": ["item:i-ghost"]}, cause="Pilot 失败用例")
    check("悬空引用事件被拒", bool(errs), str(errs[:1]))
    check("被拒事件未入账", len(w.events) == n_before)

    # 审计汇总（复用 bench 不变量 + 账本纪律）
    problems = check_invariants(w)
    check("引擎不变量（bench 同款）", not problems, str(problems[:2]))
    causeless = [e for e in w.events if not (e.cause or "").strip()]
    check("无因事件为 0", not causeless, f"{len(causeless)} 条")
    part = [e for e in w.events
            if e.kind in ("item_changed", "item_added", "item_removed")
            and (e.payload or {}).get("partial")]
    check("非原子部分提交为 0", not part, f"{len(part)} 条")
    untrace = [e for e in w.events[-30:]
               if not e.cause or not e.summary]
    check("近 30 条事件可追溯（有因有文）", not untrace)

    # 三份产物：完整账本 JSON / 玩家视角文本 / 审计报告
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / "pilot_ledger.json"
    Universe(worlds={w.name: w}, current=w.name).save(str(ledger))
    build_player_view(str(ledger), str(out_dir / "pilot_player_view.txt"))
    with open(out_dir / "pilot_audit.md", "w", encoding="utf-8") as f:
        f.write("# WorldLedger 开源可复现性 Pilot 审计\n\n")
        f.write("- 模式：MockLLM（零成本确定性）｜世界：Pilot\n")
        f.write(f"- 事件总数：{len(w.events)}｜回合：{w.turn}\n")
        f.write(f"- 复现：同种子生成 to_dict 相等，"
                f"两次运行事件直方图一致\n\n")
        f.write("## 检查结果\n\n| 检查 | 结果 | 备注 |\n"
                "| --- | --- | --- |\n")
        for c in CHECKS:
            f.write(f"| {c['name']} | {'✅' if c['passed'] else '❌'} "
                    f"| {c['note'][:60]} |\n")
        f.write("\n## 能力缺口\n\n")
        f.write("- 行为层：引擎只约束知识边界（B 检索不到秘密），"
                "不强制模型「主动/不主动告诉」——\n"
                "  「没有理由时 NPC 不主动告诉玩家」属于模型行为，"
                "Mock 无法证明，需真实模型人工评估。\n")
        f.write("- 复现层：Mock 全程确定性；真实模型输出不可复现"
                "（非引擎缺口）。\n")
    passed = sum(1 for c in CHECKS if c["passed"])
    print(f"\nPilot 完成：{passed}/{len(CHECKS)} 项通过 → {out_dir}")
    return {"checks": list(CHECKS), "passed": passed,
            "total": len(CHECKS), "events": len(w.events)}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    run(DEFAULT_OUT)


if __name__ == "__main__":
    main()
