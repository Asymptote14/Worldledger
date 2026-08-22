"""Bridge WorldLedger entity events to the optional state-runtime package.

The adapter deliberately owns the domain mapping.  ``state-runtime`` only
validates generic entities, references, preconditions, and atomic patches;
WorldLedger remains responsible for NPC, item, scene, and narrative rules.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any

try:
    from state_runtime import Change, Entity, Precondition, Proposal, StateRuntime
except ImportError as exc:  # pragma: no cover - exercised by optional install
    raise ImportError(
        "state-runtime is optional; install it before using this adapter"
    ) from exc


def _find_item(world: Any, item_id: str) -> tuple[Any | None, dict | None]:
    for scene in world.scenes.values():
        for item in scene.items:
            if item.get("id") == item_id:
                return scene, item
    return None, None


def _entity_location(world: Any, entity_id: str) -> str | None:
    if entity_id == "player":
        return world.player.get("location")
    if entity_id.startswith("npc:"):
        npc = world.npcs.get(entity_id[4:])
        return npc.state.location if npc is not None else None
    if entity_id.startswith("item:"):
        scene, item = _find_item(world, entity_id[5:])
        if item is None:
            return None
        return item.get("location") or (scene.id if scene is not None else None)
    if entity_id.startswith("scene:"):
        scene_id = entity_id[6:]
        return scene_id if scene_id in world.scenes else None
    return None


def entities_from_world(world: Any) -> list[Entity]:
    """Create the generic entity projection used by the runtime bridge."""
    entities: list[Entity] = []
    entities.append(Entity("player", "player", {
        "location": world.player.get("location"),
        "can_act": bool(world.player.get("can_act", True)),
        "condition": str(world.player.get("condition", "")),
    }))
    for scene in world.scenes.values():
        entities.append(Entity(f"scene:{scene.id}", "scene", {
            "location": scene.id,
            "last_local_consequence": None,
        }))
        for item in scene.items:
            item_id = str(item.get("id", ""))
            if not item_id:
                continue
            entities.append(Entity(f"item:{item_id}", "item", {
                "location": item.get("location") or scene.id,
                "note": str(item.get("note", "")),
                "held_by": item.get("held_by"),
            }))
    for npc in world.npcs.values():
        entities.append(Entity(f"npc:{npc.id}", "npc", {
            "location": npc.state.location,
            "can_act": bool(npc.state.can_act),
            "mood": npc.state.mood,
        }))
    return entities


def runtime_from_world(world: Any) -> StateRuntime:
    """Build a runtime projection for a WorldLedger world snapshot."""
    return StateRuntime(entities_from_world(world))


def prepare_current_world_event(world: Any, normalized: dict):
    """Validate one event against a fresh projection of the current world.

    The returned prepared event is intentionally not committed.  The caller
    keeps WorldLedger as the sole state owner and event log.
    """
    runtime = runtime_from_world(world)
    proposal = proposal_from_normalized(world, normalized)
    return runtime.prepare(proposal)


def proposal_from_normalized(world: Any, normalized: dict) -> Proposal:
    """Translate a validated WorldLedger entity event into a Proposal.

    The normalized event has already passed WorldLedger-specific validation.
    This second projection adds generic location preconditions and scope so
    the runtime can independently reject stale or incomplete entity reads.
    """
    params = normalized["params"]
    refs = tuple(params.get("refs", ()))
    location = str(params.get("location", ""))
    changes: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def add_change(entity_id: str, patch: dict[str, Any]) -> None:
        if not patch:
            return
        changes.setdefault(entity_id, {}).update(deepcopy(patch))

    for patch in normalized.get("item_patches", ()):
        item_id = str(patch.get("item", ""))
        entity_id = f"item:{item_id}"
        generic_patch = {}
        for key in ("location", "note", "held_by"):
            if key in patch:
                generic_patch[key] = patch[key]
        add_change(entity_id, generic_patch)

    for patch in normalized.get("actor_patches", ()):
        target = str(patch.get("target", ""))
        entity_id = target if target == "player" else target
        generic_patch = {
            "can_act": bool(patch.get("can_act")),
            "condition": str(patch.get("condition", "")),
        }
        add_change(entity_id, generic_patch)

    for patch in normalized.get("scene_state_patches", ()):
        scene_id = str(patch.get("scene", ""))
        add_change(f"scene:{scene_id}", {
            "last_local_consequence": {
                "id": patch.get("id"),
                "text": patch.get("text", patch.get("fact", "")),
                "duration_days": patch.get("duration_days"),
            },
        })

    preconditions = tuple(
        Precondition(entity_id, "location", location)
        for entity_id in refs
        if _entity_location(world, entity_id) is not None
    )
    return Proposal(
        cause="多实体事件",
        changes=tuple(Change(entity_id, patch)
                       for entity_id, patch in changes.items()),
        preconditions=preconditions,
        visible_to=(),
        metadata={
            "worldledger_kind": "world_event",
            "title": params.get("title", ""),
            "location": location,
            "refs": list(refs),
        },
        scope=refs,
    )
