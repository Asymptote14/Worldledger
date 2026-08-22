"""Deterministic bridge pilot for WorldLedger and state-runtime.

Run with the sibling state-runtime checkout available on PYTHONPATH:

    $env:PYTHONPATH="C:/path/to/state-runtime"
    python -m tools.state_runtime_pilot

Outputs are written to examples/state_runtime_pilot/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import state_runtime  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on installation
    raise SystemExit(
        "state-runtime is required for this pilot; install it or set PYTHONPATH"
    ) from exc

from worldledger import evolution
from worldledger.event import apply_item_patch
from worldledger.llm import MockLLM
from worldledger.store import Universe
from worldledger.worldgen import ensure_scene, generate_world
from worldledger.state_runtime_adapter import (
    prepare_current_world_event,
    proposal_from_normalized,
)

DESC = "一座永远在下雨的小城，人们撒谎时会漏出真心话"
PILOT_SCENE = "s-alley"
DEFAULT_OUT = (Path(__file__).resolve().parents[1]
               / "examples" / "state_runtime_pilot")


class ScriptedLLM:
    name = "state-runtime-pilot"

    def __init__(self, response: str):
        self.response = response

    def chat_json(self, system: str, user: str) -> dict:
        return json.loads(self.response)


def _collision(car_id: str = "i-pilot-car") -> dict:
    return {
        "title": "行驶中的车撞到玩家",
        "detail": "车辆撞到站在巷口的玩家，车轮在湿路面留下擦痕。",
        "location": PILOT_SCENE,
        "intensity": 0.9,
        "participants": ["item:i-pilot-car", "player",
                         f"scene:{PILOT_SCENE}"],
        "item_patches": [{
            "op": "change", "item": car_id,
            "location": PILOT_SCENE, "note": "撞停，车头凹陷",
        }],
        "actor_patches": [{
            "target": "player", "can_act": False,
            "condition": "被车撞伤",
        }],
        "scene_state_patches": [{
            "scene": f"scene:{PILOT_SCENE}", "op": "add",
            "fact": "pilot-tire-marks", "text": "路面留下轮胎擦痕",
            "duration_days": 1,
        }],
    }


def _world():
    world = generate_world(MockLLM(), "State Runtime Pilot", DESC)
    ensure_scene(MockLLM(), world, PILOT_SCENE)
    world.player["location"] = PILOT_SCENE
    errors = apply_item_patch(
        world, {"op": "add", "item": "i-pilot-car", "name": "面包车",
                "location": PILOT_SCENE, "note": "正在行驶"},
        cause="Pilot 布置")
    if errors:
        raise RuntimeError("Pilot 布置失败：" + "；".join(errors))
    world.pass_time(6, cause="Pilot 等待")
    return world


def _pulse_plan(proposal: dict) -> str:
    return json.dumps({
        "events": [], "entity_events": [proposal],
        "npc_plans": [], "daily_bits": [], "new_npcs": [],
        "new_scenes": [], "item_patches": [], "crowds": [],
    }, ensure_ascii=False)


def run(out_dir: Path = DEFAULT_OUT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, note: str = "") -> None:
        checks.append((name, passed, note))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}｜{note}")

    world = _world()
    proposal = _collision()
    normalized, errors = evolution._normalize_entity_event(world, proposal)
    check("WorldLedger 提案归一化", not errors, str(errors[:1]))
    generic = proposal_from_normalized(world, normalized)
    prepared = prepare_current_world_event(world, normalized)
    event_dict = prepared.event.to_dict()
    check("Proposal 已创建", bool(generic.changes),
          f"entities={list(event_dict['entity_ids'])}")
    check("StateRuntime.prepare 通过", prepared.event.cause == "多实体事件",
          f"clock={prepared.event.clock}")

    before = len(world.events)
    summaries = evolution.world_pulse(
        ScriptedLLM(_pulse_plan(proposal)), world,
        use_state_runtime=True)
    root = next((event for event in world.events[before:]
                 if event.kind == "world_event"
                 and event.payload.get("state_runtime")), None)
    check("真实 world_pulse 放行事件", root is not None,
          str(summaries[:1]))
    audit = root.payload["state_runtime"] if root else {}
    check("WorldLedger 只提交一次", audit.get("mode") == "validator",
          f"mode={audit.get('mode')}")
    car = next(item for scene in world.scenes.values()
               for item in scene.items if item.get("id") == "i-pilot-car")
    check("车辆状态改变", "撞停" in car.get("note", ""), car.get("note", ""))
    check("玩家状态改变", world.player.get("can_act") is False,
          world.player.get("condition", ""))
    check("地点状态改变", any(
        fact.get("id") == "pilot-tire-marks"
        for fact in world.scenes[PILOT_SCENE].state_facts))

    # Prepare a valid proposal, then change the live world before validation.
    # The fresh projection must reject it without writing either ledger.
    failure_world = _world()
    failure_normalized, failure_errors = evolution._normalize_entity_event(
        failure_world, proposal)
    failure_world.player["location"] = "s-station"
    failure_event_count = len(failure_world.events)
    try:
        prepare_current_world_event(failure_world, failure_normalized)
    except ValueError as exc:
        failure_note = str(exc)
        failure_rejected = True
    else:
        failure_note = "unexpectedly accepted"
        failure_rejected = False
    check("当前状态变化后拒绝旧 Proposal", failure_rejected, failure_note)
    check("拒绝时 WorldLedger 零写入",
          len(failure_world.events) == failure_event_count)

    ledger_path = out_dir / "state_runtime_ledger.json"
    proposal_path = out_dir / "state_runtime_proposal.json"
    audit_path = out_dir / "state_runtime_audit.md"
    Universe(worlds={world.name: world}, current=world.name).save(str(ledger_path))
    proposal_path.write_text(json.dumps(event_dict, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    with audit_path.open("w", encoding="utf-8") as stream:
        stream.write("# State Runtime Bridge Pilot\n\n")
        stream.write("- 模式：MockLLM，确定性，无 API key\n")
        stream.write("- WorldLedger 是唯一状态与事件账本\n")
        stream.write("- StateRuntime 仅执行无副作用 `prepare()` 校验\n\n")
        stream.write("| 检查 | 结果 | 备注 |\n| --- | --- | --- |\n")
        for name, passed, note in checks:
            stream.write(f"| {name} | {'PASS' if passed else 'FAIL'} "
                         f"| {note[:100]} |\n")
    passed = sum(ok for _, ok, _ in checks)
    print(f"\nState Runtime Pilot：{passed}/{len(checks)} PASS → {out_dir}")
    return {"passed": passed, "total": len(checks),
            "checks": checks, "proposal": event_dict}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    result = run()
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
