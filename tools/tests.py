"""v0 验收测试：确定性、包络线、写入即记忆、天变、门、事件日志。

运行：python -m unittest tools.tests -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from worldledger import cards, evolution, history, interpreter, physics, worldgen
from worldledger.event import (apply_item_patch, emit, event_identity_note,
                             transfer_item)
from worldledger.llm import MockLLM, SemanticSearch
from worldledger.store import (ActionState, Door, ITEM_MAX_IDLE, Law, Memory,
                    LawProfile, NPC, Scene, Universe, World, active_items,
                    experience_payload, experience_window, game_time,
                    memory_effectiveness, mood_now,
                    memory_gaps_payload,
                    retrieval_window, scene_associations, scene_changes,
                    text_similarity, touch_items, weather_now)
from worldledger.worldgen import ensure_scene, generate_world
from tools import finalstory

DESC = "一座永远在下雨的小城，人们撒谎时会漏出真心话"
DESC2 = "一座永不落日的沙漠之城，火焰只会照亮不会灼伤"


class TestDeterminism(unittest.TestCase):
    def test_same_description_same_world(self):
        llm = MockLLM()
        a = generate_world(llm, "世界1", DESC)
        b = generate_world(llm, "世界1", DESC)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_same_dialogue_same_result(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        w.player["location"] = "s-cafe"
        arin = w.npcs["n-arin"]
        r1 = interpreter.dialogue_turn(llm, w, arin, "你在等谁？")
        r2 = interpreter.dialogue_turn(llm, w, arin, "你在等谁？")
        self.assertEqual(r1.reply, r2.reply)
        self.assertEqual(r1.law_triggers, r2.law_triggers)

    def test_dialogue_ledger_orders_input_before_memory_and_reply(self):
        """一段对话的原因先入账，记忆和回应都不能排在玩家发言之前。"""
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        w.player["location"] = "s-cafe"
        arin = w.npcs["n-arin"]
        before = len(w.events)
        interpreter.dialogue_turn(llm, w, arin, "你在等谁？")
        events = w.events[before:]
        said = next(i for i, e in enumerate(events)
                    if e.kind == "player_said")
        memory = next(i for i, e in enumerate(events)
                      if e.kind == "dialogue" and "记住了：" in e.summary)
        reply = next(i for i, e in enumerate(events)
                     if e.kind == "dialogue"
                     and (e.payload or {}).get("event_params", {}).get(
                         "reply"))
        self.assertLess(said, memory)
        self.assertLess(memory, reply)

    def test_player_story_hides_private_dialogue_memory_and_uses_clock(self):
        """导出只显示听到的回应，时间读事件钟点戳而非账本序号。"""
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        arin = w.npcs["n-arin"]
        sid = w.player["location"]
        w.player["start"] = sid
        w.pass_time(6)
        w.log("dialogue", "阿凛 记住了：我不会让玩家看见这句。", "测试",
              {"npc": arin.id})
        emit(w, "player_said", {"npc": arin.id, "content": "你好"}, "测试")
        w.log("dialogue", "阿凛：你真的来了。", "测试",
              {"npc": arin.id, "reply": "你真的来了。"})
        for _ in range(30):
            w.log("daily_life", "雨落在屋檐上。", "测试",
                  {"event_params": {"location": sid,
                                      "detail": "雨落在屋檐上.",
                                      "intensity": 0.1}})
        self.assertEqual(finalstory.render_time(w.events[-1]), "第1天·白昼")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "world.json"
            out = Path(tmp) / "story.txt"
            Universe(worlds={w.name: w}, current=w.name).save(src)
            finalstory.build(str(src), str(out))
            story = out.read_text(encoding="utf-8")
        self.assertNotIn("我不会让玩家看见", story)
        self.assertIn("阿凛：你真的来了。", story)
        self.assertNotIn("此刻在", story)
        self.assertNotIn("全部事件", story)

    def test_player_story_shows_departure_and_deduplicates_background(self):
        """玩家看得见身边的人离开，开场背景不重复说同一件事。"""
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        arin = w.npcs["n-arin"]
        sid = arin.state.location
        w.player["location"] = sid
        w.player["start"] = sid
        effect = "雨会让谎话漏出真心"
        w.law_profile.laws = [Law(id="law-1", trigger="有人撒谎", effect=effect,
                                  intensity=0.6)]
        w.facts = [effect, "旧车站只在黄昏开门"]
        evolution.move_npc(w, arin, "s-station", cause="测试离场")
        departure = (f"阿凛 从「{w.scenes[sid].name}」来到"
                     f"「{w.scenes['s-station'].name}」")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "world.json"
            out = Path(tmp) / "story.txt"
            Universe(worlds={w.name: w}, current=w.name).save(src)
            finalstory.build(str(src), str(out))
            story = out.read_text(encoding="utf-8")
        self.assertIn(departure, story)
        self.assertEqual(story.count(effect), 1)

    def test_cross_scene_action_is_visible_at_origin_not_destination(self):
        """在途承诺不能让目的地玩家提前看见远处的人已经动身。"""
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        arin = w.npcs["n-arin"]
        origin = arin.state.location
        destination = "s-station"
        errors = emit(w, "npc_acted", {
            "npc": arin.id, "action": "走向旧车站",
            "location": destination, "days": 0.5,
        }, "测试")
        self.assertEqual(errors, [])
        event = w.events[-1]
        params = event.payload["event_params"]
        self.assertEqual(params["origin"], origin)
        self.assertTrue(finalstory.player_visible(w, event, origin))
        self.assertFalse(finalstory.player_visible(w, event, destination))


class TestEnvelope(unittest.TestCase):
    def test_bad_laws_rejected(self):
        cases = [
            Law(id="a", trigger="", effect="x", intensity=0.5),
            Law(id="b", trigger="x", effect="", intensity=0.5),
            Law(id="c", trigger="x", effect="y", intensity=1.5),
            Law(id="d", trigger="x", effect="y", intensity=-0.1),
        ]
        for law in cases:
            self.assertTrue(physics.validate_law(law), f"应拒绝 {law.id}")

    def test_too_many_laws_rejected(self):
        laws = [Law(id=f"l{i}", trigger="有人说话", effect="有回音",
                    intensity=0.5) for i in range(physics.MAX_LAWS + 1)]
        profile = LawProfile(expectation="测试", atmosphere="测试", laws=laws)
        self.assertTrue(physics.validate_profile(profile))

    def test_law_change_via_envelope(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        bad = LawProfile(expectation=DESC, atmosphere="雨",
                         laws=[Law(id="x", trigger="", effect="x",
                                   intensity=0.5)])
        errors = physics.apply_law_change(w, bad)
        self.assertTrue(errors)
        self.assertEqual(w.law_profile.version, 0)  # 天变未发生


class TestMemoryPersistence(unittest.TestCase):
    def test_npc_memory_survives_roundtrip(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        w.player["location"] = "s-cafe"
        arin = w.npcs["n-arin"]
        interpreter.dialogue_turn(llm, w, arin, "你在等谁？")
        self.assertTrue(arin.memories)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.json"
            u = Universe(worlds={w.name: w}, current=w.name)
            u.save(path)
            u2 = Universe.load(path)
            arin2 = u2.here.npcs["n-arin"]
            self.assertEqual([m.content for m in arin2.memories],
                             [m.content for m in arin.memories])
            self.assertEqual(u2.here.turn, w.turn)

    def test_events_append_only_with_cause(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        turns = [e.turn for e in w.events]
        self.assertEqual(turns, sorted(turns))
        self.assertEqual(len(turns), len(set(turns)))
        for e in w.events:
            self.assertTrue(e.cause.strip(), "事件必须携带原因引用")


class TestLawChange(unittest.TestCase):
    def test_version_bump_and_reinterpret(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        w.player["location"] = "s-cafe"
        errors = interpreter.change_law(llm, w, "这座城的人从不撒谎")
        self.assertEqual(errors, [])
        self.assertEqual(w.law_profile.version, 1)
        kinds = [e.kind for e in w.events]
        self.assertIn("law_changed", kinds)

        arin = w.npcs["n-arin"]
        r = interpreter.dialogue_turn(llm, w, arin, "你究竟在等什么？")
        self.assertIn("honesty", r.law_triggers)  # 新法则生效
        self.assertIn("[法则触发]", r.reply)

    def test_contradictory_law_replaced(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        self.assertIn("lie", [law.id for law in w.law_profile.laws])
        errors = interpreter.change_law(llm, w, "这座城的人从不撒谎")
        self.assertEqual(errors, [])
        ids = [law.id for law in w.law_profile.laws]
        self.assertIn("honesty", ids)
        self.assertNotIn("lie", ids)  # 旧法则废除，不许与新法则矛盾并存

class TestDoors(unittest.TestCase):
    def test_cross_and_return_memory_intact(self):
        llm = MockLLM()
        w1 = generate_world(llm, "世界1", DESC)
        w2 = generate_world(llm, "世界2", DESC2)
        w1.doors["d-1"] = Door(id="d-1", to_world="世界2",
                               to_scene="s-station")
        w2.doors["d-1"] = Door(id="d-1", to_world="世界1",
                               to_scene="s-cafe")
        u = Universe(worlds={"世界1": w1, "世界2": w2}, current="世界1")

        arin = w1.npcs["n-arin"]
        interpreter.dialogue_turn(llm, w1, arin, "你在等谁？")
        mem_before = len(arin.memories)
        rel_before = arin.relationship

        u.current = "世界2"  # 跨门
        self.assertNotEqual(u.here.law_profile.atmosphere,
                            w1.law_profile.atmosphere)
        u.current = "世界1"  # 返回
        self.assertEqual(len(arin.memories), mem_before)
        self.assertEqual(arin.relationship, rel_before)

    def test_main_cross_door_uses_caller_llm_and_respects_actionability(self):
        from worldledger import main as main_mod

        llm = MockLLM()
        w1 = generate_world(llm, "世界1", DESC)
        w2 = generate_world(llm, "世界2", DESC2)
        w1.doors["d-1"] = Door(id="d-1", to_world="世界2",
                               to_scene="s-station")
        u = Universe(worlds={"世界1": w1, "世界2": w2},
                     current="世界1")

        w1.player["can_act"] = False
        w1.player["condition"] = "昏迷"
        with redirect_stdout(StringIO()):
            main_mod.cross_door(llm, u, "d-1")
        self.assertEqual(u.current, "世界1")

        w1.player["can_act"] = True
        with redirect_stdout(StringIO()):
            main_mod.cross_door(llm, u, "d-1")
        self.assertEqual(u.current, "世界2")
        self.assertEqual(w2.player["location"], "s-station")


class TestEvolution(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        return llm, generate_world(llm, "世界1", DESC)

    def test_catchup_moves_npc_on_schedule(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        self.assertEqual(arin.state.location, "s-cafe")
        w.pass_time(18)  # 世界钟走到第 1 天深夜
        evolution.catch_up(llm, w, arin)
        self.assertEqual(arin.state.location, "s-station")
        self.assertIn("n-arin", w.scenes["s-station"].npcs)
        self.assertNotIn("n-arin", w.scenes["s-cafe"].npcs)
        self.assertTrue(any(e.kind == "npc_moved" for e in w.events))

    def test_heartbeat_updates_activity(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.pass_time(6)  # 相位 1（白昼）
        evolution.catch_up(llm, w, arin)
        self.assertEqual(arin.state.activity, "煮咖啡")

    def test_npc_interaction(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        zhou = w.npcs["n-zhou"]
        w.pass_time(6)
        evolution.catch_up_scene(llm, w, "s-cafe")
        evolution.move_npc(w, zhou, "s-cafe", "测试")
        summaries = evolution.heartbeat_scene(llm, w, "s-cafe")
        self.assertTrue(any("阿凛" in s or "老周" in s for s in summaries))
        self.assertTrue(any(e.kind == "npc_interaction" for e in w.events))

    def test_law_change_hits_mood(self):
        llm, w = self._world()
        errors = interpreter.change_law(llm, w, "这座城的人从不撒谎")
        self.assertEqual(errors, [])
        for npc in w.npcs.values():
            self.assertLess(npc.state.mood_value, 0)      # 情绪被打压
            self.assertEqual(npc.state.mood_reason, "天变")
            self.assertIn(npc.state.mood,
                          ("忧郁", "天变后的不安"))  # 个体差异

    def test_state_roundtrip(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.pass_time(18)
        evolution.catch_up(llm, w, arin)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.json"
            u = Universe(worlds={w.name: w}, current=w.name)
            u.save(path)
            u2 = Universe.load(path)
            a2 = u2.here.npcs["n-arin"]
            self.assertEqual(a2.state.location, "s-station")
            self.assertEqual(a2.state.activity, arin.state.activity)
            self.assertEqual(a2.state.mood, arin.state.mood)


class TestEvents(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        return llm, generate_world(llm, "世界1", DESC)

    def test_emit_valid_event(self):
        llm, w = self._world()
        errors = emit(w, "world_event",
                      {"title": "雨声", "detail": "雨下得更密了。",
                       "location": "s-alley", "intensity": 0.5},
                      cause="测试")
        self.assertEqual(errors, [])
        last = w.events[-1]
        self.assertEqual(last.kind, "world_event")
        self.assertEqual(last.payload["event_params"]["title"], "雨声")

    def test_emit_rejects_unknown_type(self):
        llm, w = self._world()
        errors = emit(w, "hack_the_world", {}, cause="测试")
        self.assertTrue(errors)

    def test_emit_rejects_bad_params(self):
        llm, w = self._world()
        errors = emit(w, "world_event",
                      {"title": "", "detail": "x", "intensity": 9.9},
                      cause="测试")
        self.assertTrue(errors)  # 标题空 + 强度越界

    def test_emit_rejects_bad_refs(self):
        llm, w = self._world()
        errors = emit(w, "npc_moved",
                      {"npc": "n-ghost", "from": "s-cafe", "to": "s-alley"},
                      cause="测试")
        self.assertTrue(errors)  # 引用不存在的 NPC

    def test_world_event_cannot_narrate_named_npc(self):
        """世界层事件不能偷偷替角色完成移动或行动。"""
        llm, w = self._world()
        errors = emit(w, "world_event",
                      {"title": "邮差出门", "detail": "阿凛冒雨走向钟楼。",
                       "location": "s-alley", "intensity": 0.4},
                      cause="测试")
        self.assertTrue(errors)
        self.assertFalse(any(e.kind == "world_event" for e in w.events))

    def test_world_event_cannot_use_npc_alias_or_cause(self):
        """简称和因果说明也不能成为绕过角色状态的入口。"""
        llm, w = self._world()
        w.npcs["n-arin"].name = "神秘修钟人"
        errors = emit(w, "world_event",
                      {"title": "钟楼安静", "detail": "钟楼恢复了沉寂。",
                       "location": "s-alley", "intensity": 0.3},
                      cause="修钟人离去后，钟声停止")
        self.assertTrue(errors)
        self.assertFalse(any(e.kind == "world_event" for e in w.events))

    def test_daily_life_cannot_narrate_named_npc(self):
        """日常文本也不能成为角色行动的旁路。"""
        llm, w = self._world()
        errors = emit(w, "daily_life",
                      {"detail": "阿凛冒雨走向钟楼。", "location": "s-alley",
                       "intensity": 0.2}, cause="测试")
        self.assertTrue(errors)
        self.assertFalse(any(e.kind == "daily_life" for e in w.events))

    def test_world_event_proposal_after_absence(self):
        llm, w = self._world()
        w.pass_time(25)  # 世界时钟过了 24
        evolution.world_pulse(llm, w)
        kinds = [e.kind for e in w.events]
        self.assertIn("world_event", kinds)
        # 信是车站的初始物品（场景一等状态），不是事件
        station = w.scenes["s-station"]
        self.assertTrue(any(i["name"] == "一封信" for i in station.items))
        # 再来一次不会重复提案
        w.pass_time(6)
        evolution.world_pulse(llm, w)
        discovery = [e for e in w.events if e.kind == "world_event"
                     and "发现了一封信" in e.summary]
        self.assertEqual(len(discovery), 1)


class TestCards(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        return llm, generate_world(llm, "世界1", DESC)

    def test_create_custom_npc(self):
        llm, w = self._world()
        errors = cards.create_npc(w, "信使", "披油布雨衣的送信人，从不抬头。",
                                  traits={"寡言": True, "爱说谎": True})
        self.assertEqual(errors, [])
        npc = next(n for n in w.npcs.values() if n.name == "信使")
        self.assertIn(npc.id, w.scenes[w.player["location"]].npcs)
        self.assertTrue(any(e.kind == "npc_created" for e in w.events))
        # 自定义 NPC 带爱说谎性格 → 法则照常触发
        r = interpreter.dialogue_turn(llm, w, npc, "你在等谁？")
        self.assertIn("lie", r.law_triggers)

    def test_import_export_roundtrip(self):
        llm, w = self._world()
        card = {"name": "旅人", "persona": "路过的说书人，喜欢下雨。",
                "traits": {"健谈": True},
                "memories": ["在另一座雨城听过同样的故事。"],
                "relationship": 3}
        errors = cards.import_npc(w, card, llm=llm)
        self.assertEqual(errors, [])
        npc = next(n for n in w.npcs.values() if n.name == "旅人")
        self.assertEqual([m.content for m in npc.memories],
                         ["在另一座雨城听过同样的故事。"])
        with tempfile.TemporaryDirectory() as tmp:
            path = cards.export_npc(npc, Path(tmp) / "card.json")
            back = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(back["persona"], card["persona"])
            self.assertEqual([m["content"] for m in back["memories"]],
                             card["memories"])
            self.assertIn("projections", back["memories"][0])

    def test_import_rejects_bad_card(self):
        llm, w = self._world()
        errors = cards.import_npc(w, {"name": "", "persona": "x"})
        self.assertTrue(errors)

    def test_import_from_other_world(self):
        llm = MockLLM()
        w1 = generate_world(llm, "世界1", DESC)
        w2 = generate_world(llm, "世界2", DESC2)
        errors = cards.import_from_world(w2, w1, "阿凛", llm=llm)
        self.assertEqual(errors, [])
        arin2 = next(n for n in w2.npcs.values() if n.name == "阿凛")
        arin1 = w1.npcs["n-arin"]
        self.assertEqual(arin2.persona, arin1.persona)
        self.assertTrue(any(e.kind == "npc_imported" for e in w2.events))

    def test_inject_memory(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        errors = cards.inject_memory(
            w, arin, "玩家设定：她小时候养过一只黑猫。", llm=llm)
        self.assertEqual(errors, [])
        self.assertTrue(any("黑猫" in m.content for m in arin.memories))
        self.assertTrue(any(e.cause == "玩家设定" for e in w.events))


class TestMemoryWorldProjection(unittest.TestCase):
    def _world(self):
        return generate_world(MockLLM(), "世界1", DESC)

    @staticmethod
    def _alignment(memory_id="m-1-1"):
        return json.dumps({"memories": [{
            "memory_id": memory_id,
            "age_days": 1,
            "scene": {"ref": "", "name": "北坡旧屋",
                      "then": "屋后有一株尚未开花的梅树"},
            "items": [{"ref": "", "name": "缺口小刀",
                       "then": "刀刃沾着血", "exists_now": False,
                       "current_location": "", "held_by": "",
                       "current_note": "后来遗失"}],
            "current_states": [{"text": "左手有一道尚未愈合的刀伤",
                                "review_days": 6}],
        }]}, ensure_ascii=False)

    def test_imported_past_projects_scene_item_and_current_actor_state(self):
        w = self._world()
        llm = ScriptedLLM([self._alignment()])
        errors = cards.import_npc(w, {
            "name": "迟夏", "persona": "从北坡来到城里的旅人。",
            "memories": ["昨天我在北坡旧屋用缺口小刀划伤了左手，后来刀丢了。"]
        }, llm=llm)
        self.assertEqual(errors, [])
        npc = next(n for n in w.npcs.values() if n.name == "迟夏")
        old_house = next(s for s in w.scenes.values()
                         if s.name == "北坡旧屋")
        self.assertTrue(old_house.memory_only)
        self.assertFalse(old_house.generated)
        self.assertEqual(old_house.history[0]["memory_id"], npc.memories[0].id)
        knife = next(iter(w.past_items.values()))
        self.assertEqual(knife["name"], "缺口小刀")
        self.assertFalse(knife["current_assertion"]["exists"])
        self.assertFalse(any(i.get("name") == "缺口小刀"
                             for s in w.scenes.values() for i in s.items))
        self.assertIn("尚未愈合", npc.state.facts[0]["text"])
        self.assertAlmostEqual(npc.state.facts[0]["review_clock"],
                               w.clock + 6)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projected.json"
            Universe(worlds={w.name: w}, current=w.name).save(path)
            loaded = Universe.load(path).here
        loaded_npc = next(n for n in loaded.npcs.values() if n.name == "迟夏")
        self.assertEqual(loaded_npc.state.facts, npc.state.facts)
        self.assertEqual(loaded.past_items, w.past_items)
        self.assertEqual(loaded.scenes[old_house.id].history,
                         old_house.history)

    def test_preserved_memory_item_enters_current_scene_table(self):
        w = self._world()
        response = json.dumps({"memories": [{
            "memory_id": "m-1-1", "age_days": 30, "scene": None,
            "items": [{"ref": "", "name": "母亲的铜铃",
                       "then": "母亲临行前交给我", "exists_now": True,
                       "current_location": "", "held_by": "self",
                       "current_note": "仍系在腰间"}],
            "current_states": []}]}, ensure_ascii=False)
        errors = cards.import_npc(w, {
            "name": "铃兰", "persona": "腰间总有轻响的旅人。",
            "memories": ["一个月前母亲把铜铃交给我，我一直带着。"]
        }, llm=ScriptedLLM([response]))
        self.assertEqual(errors, [])
        npc = next(n for n in w.npcs.values() if n.name == "铃兰")
        item = next(i for i in w.scenes[npc.state.location].items
                    if i.get("name") == "母亲的铜铃")
        self.assertEqual(item.get("held_by"), f"npc:{npc.id}")
        self.assertIn(item["id"], w.past_items)

    def test_body_state_projection_does_not_materialize_as_item(self):
        """身体透明度属于人物状态，不能因有名称和持有者而变成物品。"""
        w = self._world()
        response = json.dumps({"memories": [{
            "memory_id": "m-1-1", "age_days": None, "scene": None,
            "items": [{"ref": "", "name": "我的身体",
                       "then": "透明了一分", "exists_now": True,
                       "current_location": "s-cafe", "held_by": "self",
                       "current_note": "指尖依然微透明"}],
            "current_states": [{"text": "每次祈晴，身体透明度增加",
                                "review_days": None}]
        }]}, ensure_ascii=False)
        errors = cards.import_npc(w, {
            "name": "晴女", "persona": "能让局部天空放晴的少女。",
            "memories": ["第一次祈晴时，我的指尖变得透明了一点点。"]
        }, llm=ScriptedLLM([response]))

        self.assertEqual(errors, [])
        npc = next(n for n in w.npcs.values() if n.name == "晴女")
        self.assertTrue(any("身体透明度" in fact["text"]
                            for fact in npc.state.facts))
        self.assertFalse(any(item.get("name") == "我的身体"
                             for scene in w.scenes.values()
                             for item in scene.items))
        self.assertFalse(any(item.get("name") == "我的身体"
                             for item in w.past_items.values()))
        self.assertEqual(npc.memories[0].projections[0]["items"], [])

    def test_scene_generation_reads_historical_anchor_and_time_gap(self):
        w = self._world()
        sid = "s-memory-test"
        w.scenes[sid] = Scene(
            id=sid, name="旧庭院", description="", atmosphere="",
            generated=False, memory_only=True,
            history=[{"memory_id": "m-old", "npc": "n-arin",
                      "then": "院中有一口满水的井", "occurred_clock": -20.0}])
        response = json.dumps({"description": "井栏已经干裂，院墙新补过一角。",
                               "atmosphere": "荒静"}, ensure_ascii=False)
        recorder = ScriptedLLM([response])
        self.assertEqual(ensure_scene(recorder, w, sid), ["贴片生成：「旧庭院」"])
        payload = json.loads(recorder.calls[0][1])
        self.assertEqual(payload["historical_anchors"][0]["age_days"], 20.0)
        self.assertIn("满水的井", payload["historical_anchors"][0]["then"])

    def test_due_current_state_can_be_rewritten_without_medical_enum(self):
        w = self._world()
        npc = w.npcs["n-arin"]
        npc.state.facts = [{"id": "sf-cut", "text": "手背伤口仍未愈合",
                            "source_memory": "m-cut", "since_clock": -6.0,
                            "review_clock": w.clock}]
        w.pass_time(6)
        # 让其他调度不制造额外模型请求；统一脉冲只读这一份 JSON。
        for person in w.npcs.values():
            person.state.last_clock = w.clock
        pulse = {
            "events": [], "entity_events": [], "npc_plans": [],
            "state_fact_patches": [{"npc": npc.id, "op": "change",
                "fact": "sf-cut", "text": "手背留下一道浅色疤痕",
                "review_days": None, "why": "伤口已经愈合"}],
            "item_patches": [], "new_npcs": [], "crowds": [],
            "daily_bits": [], "new_scenes": [], "fact_changes": []}
        evolution.world_pulse(
            ScriptedLLM([json.dumps(pulse, ensure_ascii=False)]), w)
        self.assertEqual(npc.state.facts[0]["text"], "手背留下一道浅色疤痕")
        self.assertIsNone(npc.state.facts[0]["review_clock"])

    def test_conflicting_item_memories_preserve_uncertainty(self):
        w = self._world()
        response = {"memories": []}
        for memory_id, exists in (("m-1-1", True), ("m-1-2", False)):
            response["memories"].append({
                "memory_id": memory_id, "age_days": None, "scene": None,
                "items": [{"ref": "", "name": "旧银戒",
                           "then": "曾戴在手上", "exists_now": exists,
                           "current_location": "", "held_by": "self" if exists else "",
                           "current_note": "仍保存" if exists else "已经遗失"}],
                "current_states": []})
        errors = cards.import_npc(w, {
            "name": "南枝", "persona": "记忆相互矛盾的远行者。",
            "memories": ["我一直留着那枚旧银戒。", "那枚旧银戒早就丢了。"]
        }, llm=ScriptedLLM([json.dumps(response, ensure_ascii=False)]))
        self.assertEqual(errors, [])
        record = next(item for item in w.past_items.values()
                      if item["name"] == "旧银戒")
        self.assertTrue(record["current_assertion"]["conflict"])
        self.assertIsNone(record["current_assertion"]["exists"])
        self.assertEqual(len(record["current_assertions"]), 2)
        self.assertFalse(any(i.get("name") == "旧银戒"
                             for scene in w.scenes.values() for i in scene.items))

    def test_unstructured_birth_memory_is_rejected_without_alignment(self):
        data = MockLLM().chat_json(worldgen._WORLDGEN_SYSTEM, DESC)
        data["npcs"][0]["memories"] = ["三年前我住在一座没有入账的旧屋。"]
        with self.assertRaises(worldgen.WorldGenError):
            worldgen._build_world("错误世界", DESC, data)
        w = self._world()
        errors = cards.import_npc(w, {
            "name": "旧客", "persona": "带着未对齐过去的人。",
            "memories": ["我来自没有入账的地方。"]})
        self.assertTrue(any("必须经过模型对齐" in error for error in errors))

    def test_large_import_alignment_is_batched_before_materialization(self):
        w = self._world()
        memories = [f"过去片段 {i}" for i in range(25)]
        responses = []
        for start, end in ((1, 25), (25, 26)):
            responses.append(json.dumps({"memories": [
                {"memory_id": f"m-1-{i}", "age_days": None,
                 "scene": None, "items": [], "current_states": []}
                for i in range(start, end)]}, ensure_ascii=False))
        recorder = ScriptedLLM(responses)
        errors = cards.import_npc(w, {
            "name": "长忆", "persona": "记得许多旧事的旅行者。",
            "memories": memories}, llm=recorder)
        self.assertEqual(errors, [])
        self.assertEqual(len(recorder.calls), 2)
        npc = next(n for n in w.npcs.values() if n.name == "长忆")
        self.assertEqual(len(npc.memories), 25)

    def test_emerged_person_can_arrive_with_projected_past(self):
        w = self._world()
        errors, npc_id = cards.emerge_npc(
            w, "归舟", "多年后重新回到雨城的船工。", None,
            "s-station", "旧航线重新开通", memories=[{
                "content": "五年前我在南港留下了一只生锈的工具箱。",
                "projection": {"age_days": 1825,
                    "scene": {"ref": "", "name": "南港",
                              "then": "旧航线尽头的货运港"},
                    "items": [{"ref": "", "name": "生锈的工具箱",
                               "then": "留在仓库角落", "exists_now": None,
                               "current_location": "", "held_by": "",
                               "current_note": ""}],
                    "current_states": []}}])
        self.assertEqual(errors, [])
        self.assertIn(npc_id, w.npcs)
        self.assertTrue(any(scene.name == "南港" and scene.memory_only
                            for scene in w.scenes.values()))
        self.assertTrue(any(item.get("name") == "生锈的工具箱"
                            for item in w.past_items.values()))


class TestMemoryAttributionAndAccess(unittest.TestCase):
    def _world(self):
        return generate_world(MockLLM(), "世界1", DESC)

    def test_borrowed_body_experience_belongs_only_to_actor(self):
        w = self._world()
        actor = w.npcs["n-arin"]
        body_owner = w.npcs["n-zhou"]
        body_memories_before = len(body_owner.memories)
        w.remember(actor, "我借那具身体在站台等了一夜。", cause="测试借身",
                   kind="npc_memory", body=body_owner,
                   started_clock=2.0, ended_clock=3.0)
        memory = actor.memories[-1]
        self.assertEqual(memory.embodied_as, body_owner.id)
        self.assertEqual(len(body_owner.memories), body_memories_before)
        self.assertFalse(any("站台等了一夜" in m.content
                             for m in body_owner.memories))
        self.assertEqual(len(body_owner.memory_gaps), 1)
        subjective_gap = memory_gaps_payload(body_owner)[0]
        self.assertEqual(subjective_gap["started_clock"], 2.0)
        self.assertEqual(subjective_gap["ended_clock"], 3.0)
        self.assertNotIn("actor", subjective_gap)
        self.assertNotIn("cause", subjective_gap)

    def test_inaccessible_memory_stays_archived_but_leaves_attention(self):
        w = self._world()
        npc = w.npcs["n-arin"]
        w.remember(npc, "我记得黄昏时见过她。", cause="测试")
        memory = npc.memories[-1]
        memory.projections = [{"scene": None, "items": [],
                               "current_states": []}]
        count = len(npc.memories)
        self.assertEqual(w.set_memory_access(
            npc, [memory.id], False, "黄昏结束后的记忆消散"), [])
        self.assertEqual(len(npc.memories), count)
        self.assertFalse(memory.accessible)
        self.assertFalse(any(m.id == memory.id for m in
                             experience_window(npc, w.turn,
                                               query="黄昏见过谁")))
        self.assertEqual(memory.projections[0]["items"], [])
        self.assertEqual(w.set_memory_access(
            npc, [memory.id], True, "再次触碰旧线索"), [])
        self.assertTrue(any(m.id == memory.id for m in
                            experience_window(npc, w.turn,
                                              query="黄昏见过谁")))

    def test_inaccessible_projected_memory_still_builds_world_and_gap(self):
        w = self._world()
        actor = w.npcs["n-arin"]
        body_owner = w.npcs["n-zhou"]
        actor.memories.append(Memory(
            turn=w.turn, id="m-swap", content="那天我在旧诊所醒来。",
            accessible=False, access_cause="交换结束后无法想起",
            embodied_as=body_owner.id,
            projections=[{"age_days": 3, "duration_days": 0.5,
                          "embodied_as": body_owner.id,
                          "accessible": False,
                          "access_cause": "交换结束后无法想起",
                          "scene": {"ref": "", "name": "旧诊所",
                                    "then": "窗边放着一张窄床"},
                          "items": [], "current_states": []}]))
        self.assertEqual(history.materialize_stored_memories(w, actor), [])
        self.assertTrue(any(scene.name == "旧诊所"
                            for scene in w.scenes.values()))
        self.assertTrue(any(gap.get("source_memory") == "m-swap"
                            for gap in body_owner.memory_gaps))
        self.assertFalse(any(memory.id == "m-swap" for memory in
                             experience_window(actor, w.turn,
                                               query="旧诊所")))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "access.json"
            Universe(worlds={w.name: w}, current=w.name).save(path)
            loaded = Universe.load(path).here
        self.assertFalse(loaded.npcs[actor.id].memories[-1].accessible)
        self.assertEqual(loaded.npcs[actor.id].memories[-1].embodied_as,
                         body_owner.id)
        self.assertTrue(loaded.npcs[body_owner.id].memory_gaps)

    def test_world_pulse_forgetting_cancels_plan_from_old_memory_boundary(self):
        w = self._world()
        npc = w.npcs["n-arin"]
        body_owner = w.npcs["n-zhou"]
        w.remember(npc, "借身时我看见了锁柜里的信。", cause="测试",
                   body=body_owner, started_clock=0.0, ended_clock=0.2)
        memory = npc.memories[-1]
        w.pass_time(6)
        for person in w.npcs.values():
            person.state.last_clock = 0.0
        pulse = {
            "events": [], "entity_events": [], "state_fact_patches": [],
            "memory_access_patches": [{"npc": npc.id,
                "memories": [memory.id], "accessible": False,
                "why": "交换结束，相关经历开始消散"}],
            "npc_plans": [{"npc": npc.id,
                           "state": {"activity": "按旧记忆去找锁柜",
                                     "mood": "平静",
                                     "location": npc.state.location}}],
            "item_patches": [], "new_npcs": [], "crowds": [],
            "daily_bits": [], "new_scenes": [], "fact_changes": []}
        recorder = ScriptedLLM([json.dumps(pulse, ensure_ascii=False)])
        summaries = evolution.world_pulse(recorder, w)
        payload = json.loads(recorder.calls[0][1])
        row = next(row for row in payload["memory_access_index"]
                   if row["memory"] == memory.id)
        self.assertNotIn("content", row)
        self.assertFalse(memory.accessible)
        self.assertNotEqual(npc.state.activity, "按旧记忆去找锁柜")
        self.assertFalse(any("按旧记忆去找锁柜" in event.summary
                             for event in w.events))

    def test_borrowed_body_short_end_to_end(self):
        """借身一日：行动者记得，身体主人断档，公共后果仍在账本。"""
        w = self._world()
        actor = w.npcs["n-arin"]
        body_owner = w.npcs["n-zhou"]
        witness = w.npcs["n-man"]
        actor.links[body_owner.id] = 0.2  # 契约/关系是世界事实，不是题材枚举
        start = w.clock
        w.pass_time(24, cause="借身经历持续一日")
        w.remember(actor, "我借用老周的身体，在旧车站把信封放进了柜台抽屉。",
                   cause="交换期间的行动", kind="npc_memory", body=body_owner,
                   started_clock=start, ended_clock=w.clock)

        errors = apply_item_patch(
            w, {"op": "change", "item": "i-letter", "location": "s-station",
                "note": "信封被放进柜台抽屉"}, cause="交换期间的行动")
        self.assertEqual(errors, [])
        item_event = next(e for e in reversed(w.events)
                          if e.kind == "item_changed")
        w.remember(witness, "我看见柜台抽屉里多了一封被雨打湿的信。",
                   cause=item_event.summary)

        actor_view = experience_payload(
            experience_window(actor, w.turn, query="柜台抽屉里的信"))
        owner_view = experience_payload(
            experience_window(body_owner, w.turn, query="柜台抽屉里的信"))
        owner_gaps = memory_gaps_payload(body_owner)
        self.assertTrue(any("借用老周的身体" in row["content"]
                            for row in actor_view))
        self.assertFalse(any("柜台抽屉" in row["content"]
                             for row in owner_view))
        self.assertEqual(len(owner_gaps), 1)
        self.assertNotIn("actor", owner_gaps[0])
        self.assertNotIn("cause", owner_gaps[0])
        self.assertTrue(any("信封被放进柜台抽屉" in e.summary
                            for e in w.events))
        self.assertTrue(any("看见柜台抽屉" in m.content
                            for m in witness.memories))
        memory_event = next(e for e in reversed(w.events)
                            if e.kind == "npc_memory"
                            and "借用老周的身体" in e.summary)
        params = memory_event.payload
        self.assertEqual(params["actor"], actor.id)
        self.assertEqual(params["body"], body_owner.id)
        self.assertEqual(params["ended_clock"] - params["started_clock"], 1.0)


class TestPatchGen(unittest.TestCase):
    def test_dna_only_start_generated(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        generated = [s for s in w.scenes.values() if s.generated]
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].id, w.player["location"])

    def test_ensure_scene_generates_patch(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        cafe = w.scenes["s-cafe"]
        self.assertFalse(cafe.generated)
        summaries = ensure_scene(llm, w, "s-cafe")
        self.assertEqual(len(summaries), 1)
        self.assertTrue(cafe.generated)
        self.assertTrue(cafe.description)
        self.assertTrue(any(e.kind == "scene_generated" for e in w.events))
        # 幂等：再生成一次无副作用
        self.assertEqual(ensure_scene(llm, w, "s-cafe"), [])
        self.assertEqual(sum(1 for e in w.events
                             if e.kind == "scene_generated"), 1)

    def test_patch_deterministic(self):
        llm = MockLLM()
        a = generate_world(llm, "世界1", DESC)
        b = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, a, "s-alley")
        ensure_scene(llm, b, "s-alley")
        self.assertEqual(a.scenes["s-alley"].description,
                         b.scenes["s-alley"].description)

    def test_patch_survives_roundtrip(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.json"
            u = Universe(worlds={w.name: w}, current=w.name)
            u.save(path)
            u2 = Universe.load(path)
            self.assertTrue(u2.here.scenes["s-cafe"].generated)
            self.assertFalse(u2.here.scenes["s-alley"].generated)

    def test_fog_npc_place_materializes_before_dialogue(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        man = w.npcs["n-man"]  # 小满驻在积水小巷（雾中）
        w.player["location"] = man.state.location
        self.assertFalse(w.scenes[man.state.location].generated)
        ensure_scene(llm, w, man.state.location)  # 聊之前先兑现驻地
        self.assertTrue(w.scenes[man.state.location].generated)
        self.assertIn(man.id, w.scenes[man.state.location].npcs)
        r = interpreter.dialogue_turn(llm, w, man, "你在等什么？")
        self.assertIn("信", r.reply)


class TestDistantPulse(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-cafe"
        return llm, w

    def test_player_action_is_judged_with_relevant_lived_experience(self):
        """接受或拒绝玩家动作必须看这个人活过什么，而非只看静态人设。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.memories.clear()
        w.remember(arin, "我曾被陌生人突然抱住，之后很怕别人靠太近",
                   cause="测试", importance=0.9,
                   keywords=["拥抱", "陌生人"])
        for i in range(99):
            w.remember(arin, f"普通经历 {i}", cause="测试")
        arin.state.last_turn = w.turn
        result = json.dumps({
            "accepted": False, "reply": "请离我远一点。",
            "relationship_delta": -2, "mood_delta": -0.3,
            "memory_importance": 0.7, "memory": "他突然想抱住我。",
            "law_ids": [], "days": 0}, ensure_ascii=False)
        recorder = ScriptedLLM([result])
        interpreter.player_action(recorder, w, arin, "突然抱住她")
        payload = json.loads(recorder.calls[0][1])
        lived = payload["npc"]["lived_experiences"]
        self.assertEqual(payload["npc"]["experience_count"], 100)
        self.assertLessEqual(len(lived), 12)
        self.assertTrue(any("怕别人靠太近" in row["content"] for row in lived))

    def test_pulse_advances_detached_npcs(self):
        llm, w = self._world()
        zhou = w.npcs["n-zhou"]  # 车站（已生成、非玩家所在）
        arin = w.npcs["n-arin"]  # 咖啡店（玩家所在）
        w.pass_time(6)           # 相位 1（白昼）
        pulse_turn = w.turn      # 裁决发生在这一回合
        evolution.world_pulse(llm, w)
        # 日常小事在脉冲末尾写入，会让回合 +1——裁决时点看 pulse_turn
        self.assertEqual(zhou.state.last_turn, pulse_turn)
        self.assertEqual(zhou.state.activity, "打盹")
        self.assertEqual(arin.state.last_turn, pulse_turn)  # 同一脉冲统一调度

    def test_pulse_skips_fog_scenes(self):
        llm, w = self._world()
        man = w.npcs["n-man"]  # 雾中巷子：冻结
        last = man.state.last_turn
        w.pass_time(6)
        evolution.world_pulse(llm, w)
        self.assertEqual(man.state.last_turn, last)

    def test_pulse_interval_gating(self):
        llm, w = self._world()
        w.pass_time(6)
        evolution.world_pulse(llm, w)
        w.pass_time(1)  # 间隔不足 6 回合
        self.assertEqual(evolution.world_pulse(llm, w), [])

    def test_pulse_exact_boundary_tolerates_float_roundoff(self):
        """六个心跳不能因 5.999999999999999 晚一轮触发。"""
        llm, w = self._world()
        w.heartbeat = 0.0417
        w.clock = 0.5004
        w.pulse_last_clock = w.clock
        w.pass_time(6)
        self.assertLess(evolution.elapsed_steps(w, 0.5004), 6.0)
        evolution.world_pulse(llm, w)
        self.assertAlmostEqual(w.pulse_last_clock, w.clock)

    def test_ledger_writes_do_not_advance_npc_time(self):
        """账本再热闹，也不能把零时长事件误算成角色过了日子。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.pass_time(6)
        evolution.world_pulse(llm, w)
        before_clock = arin.state.last_clock
        before_pulse = w.pulse_last_clock

        for i in range(50):
            w.log("audit_note", f"审计记录 {i}", "测试")

        self.assertEqual(evolution.world_pulse(llm, w), [])
        self.assertEqual(arin.state.last_clock, before_clock)
        self.assertEqual(w.pulse_last_clock, before_pulse)

        w.pass_time(6)
        evolution.world_pulse(llm, w)
        self.assertGreater(arin.state.last_clock, before_clock)

    def test_pulse_writes_events_with_cause(self):
        llm, w = self._world()
        w.player["location"] = "s-station"  # 玩家在车站
        w.pass_time(16)                     # 深夜：阿凛作息指向车站
        evolution.world_pulse(llm, w)
        moved = [e for e in w.events if e.kind == "npc_moved"
                 and e.cause == "间隔心跳"]
        self.assertFalse(moved)  # 先开始赶路，不能由状态快照瞬移
        self.assertEqual(w.npcs["n-arin"].state.location, "s-cafe")
        self.assertEqual(w.npcs["n-arin"].state.action.location, "s-station")

    def test_pulse_fires_world_events_by_calendar(self):
        llm, w = self._world()
        w.player["location"] = "s-cafe"
        w.pass_time(50)  # turn 2 → 52：日历上信与钟声都已到期
        evolution.world_pulse(llm, w)
        kinds = [e.kind for e in w.events]
        self.assertIn("world_event", kinds)
        # 信是车站初始物品（状态）；发现信的事件（微分）在世界演化里
        self.assertTrue(any("发现了一封信" in e.summary for e in w.events))
        self.assertTrue(any(i["name"] == "一封信"
                            for i in w.scenes["s-station"].items))
        # 事件因「世界演化」写入——与玩家观察无关
        self.assertTrue(any(e.cause == "世界演化" for e in w.events))

    def test_world_grows_new_scene(self):
        llm, w = self._world()
        w.pass_time(80)  # turn 82 ≥ 72：世界生长到期
        before = len(w.scenes)
        evolution.world_pulse(llm, w)
        self.assertEqual(len(w.scenes), before + 1)
        kinds = [e.kind for e in w.events]
        self.assertIn("scene_extended", kinds)
        # 新贴片是雾中的，双向链接（从已生成的车站长出来）
        new_scene = next(s for s in w.scenes.values()
                         if not s.generated and s.name == "雾中的旧码头")
        self.assertIn(new_scene.id, w.scenes["s-station"].exits)
        self.assertIn("s-station", new_scene.exits)

    def test_pulse_frequency_scales_with_distance(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        ensure_scene(llm, w, "s-alley")
        w.player["location"] = "s-station"  # 玩家在车站
        arin = w.npcs["n-arin"]  # 咖啡店（距离 1 → 间隔 6）
        man = w.npcs["n-man"]    # 小巷（距离 2 → 间隔 12）
        w.pass_time(6)           # turn 3 → 9
        evolution.world_pulse(llm, w)
        self.assertEqual(arin.state.last_turn, 9)  # 距离近，跳了
        self.assertEqual(man.state.last_turn, 1)   # 距离远，没跳
        w.pass_time(6)           # 上一脉冲的日常小事把回合 +1
        t2 = w.turn
        evolution.world_pulse(llm, w)
        self.assertEqual(man.state.last_turn, t2)  # 欠账到期，补跳

    def test_game_time_stamps(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        self.assertEqual(arin.state.last_time, game_time(1))
        w.pass_time(18)  # 世界钟到第 1 天深夜
        evolution.catch_up(llm, w, arin)
        self.assertEqual(arin.state.last_time, "第1天·深夜")


class TestSemanticRetrieval(unittest.TestCase):
    """语义检索：问得准的能被捞出——长对话后关键事实不丢。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        return llm, w

    def test_text_similarity_ranks_related_higher(self):
        q = "我的名字叫什么"
        hit = text_similarity(q, "玩家的真名是黑猫")
        miss = text_similarity(q, "今天雨好大")
        self.assertGreater(hit, miss)

    def test_window_surfaces_semantic_match(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.memories.clear()
        w.remember(arin, "玩家的真名是「黑猫」，只有我知道。",
                   cause="测试", importance=0.7)
        w.pass_time(60)  # 衰减到很旧，纯重要度排不进窗口
        for i in range(8):
            w.remember(arin, f"闲聊 {i}", cause="测试", importance=0.3)
        # 无语义检索：尘封（回归钉住）
        plain = retrieval_window(arin, w.turn)
        self.assertFalse(any("黑猫" in m.content for m in plain))
        # 语义检索：捞出来
        win = retrieval_window(arin, w.turn, query="我叫什么",
                               embedder=SemanticSearch({}))
        self.assertTrue(any("黑猫" in m.content for m in win))
        self.assertLessEqual(len(win), 6)  # 体积仍恒定

    def test_dialogue_payload_retrieves_from_unbounded_experience(self):
        """人物过去可增长，但一次对话只读与当前话题相关的有限切片。"""
        llm, w = self._world()
        w.player["location"] = "s-cafe"
        arin = w.npcs["n-arin"]
        arin.memories.clear()
        w.remember(arin, "我在旧桥下捡到过一枚铜铃", cause="测试",
                   keywords=["旧桥", "铜铃"])
        for i in range(79):
            w.remember(arin, f"普通日子 {i}", cause="测试")
        reply = json.dumps({
            "reply": "哦。", "choices": [], "law_ids": [],
            "relationship_delta": 0, "mood_delta": 0.0,
            "memory_importance": 0.5, "memory": "聊了几句。"},
            ensure_ascii=False)
        goal = json.dumps({"events": [], "goal_updates": {},
                           "new_goals": []}, ensure_ascii=False)
        recorder = ScriptedLLM([reply, reply, reply, goal])
        arin.state.last_turn = w.turn  # 跳过读取补算的 LLM 调用
        interpreter.dialogue_turn(recorder, w, arin, "旧桥下的铜铃呢？")
        payload = json.loads(recorder.calls[0][1])
        mems = payload["npc"]["memories"]
        self.assertLessEqual(len(mems), 12)
        self.assertEqual(payload["npc"]["experience_count"], 80)
        self.assertTrue(any("铜铃" in memory for memory in mems))

    def test_dialogue_memory_focus_carries_neighboring_past_into_followup(self):
        """模型引用一段往事后，含糊追问仍能沿该经历的前后继续深入。"""
        llm, w = self._world()
        w.player["location"] = "s-cafe"
        arin = w.npcs["n-arin"]
        arin.memories.clear()
        arin.goals.clear()
        for text in ("那天出门前母亲递给我一把伞",
                     "我在旧桥上弄丢了弟弟",
                     "从桥上回来后我再也不肯撑伞",
                     "后来我搬到了车站附近"):
            w.remember(arin, text, cause="测试", importance=0.8)
        experience_window(arin, w.turn, limit=12)
        lost = next(m for m in arin.memories if "弄丢了弟弟" in m.content)
        first = json.dumps({
            "reply": "我在那座桥上失去过一个人。", "reaction": "",
            "action": None, "choices": [], "law_ids": [],
            "relationship_delta": 0, "mood_delta": -0.1,
            "memory_importance": 0.0, "memory": "",
            "memory_refs": [lost.id], "item_patches": []},
            ensure_ascii=False)
        arin.state.last_turn = w.turn
        interpreter.dialogue_turn(ScriptedLLM([first]), w, arin,
                                  "你为什么害怕旧桥？")
        self.assertEqual(arin.state.memory_focus, [lost.id])

        second = json.dumps({
            "reply": "后来我一个人回了家。", "reaction": "",
            "action": None, "choices": [], "law_ids": [],
            "relationship_delta": 0, "mood_delta": -0.1,
            "memory_importance": 0.0, "memory": "",
            "memory_refs": [], "item_patches": []}, ensure_ascii=False)
        recorder = ScriptedLLM([second])
        arin.state.last_turn = w.turn
        interpreter.dialogue_turn(recorder, w, arin, "后来呢？")
        payload = json.loads(recorder.calls[0][1])
        contents = [row["content"]
                    for row in payload["npc"]["lived_experiences"]]
        self.assertIn("我在旧桥上弄丢了弟弟", contents)
        self.assertIn("从桥上回来后我再也不肯撑伞", contents)


class TestMemoryUpgrade(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        return llm, generate_world(llm, "世界1", DESC)

    def test_low_importance_experiences_are_not_deleted(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.memories.clear()
        arin.beliefs.clear()
        # 注意力可以忽略闲聊，但个人档案不能替这个人删除活过的日子。
        w.remember(arin, "重要的承诺", cause="测试", importance=1.0)
        for i in range(60):
            w.remember(arin, f"闲聊 {i}", cause="测试", importance=0.1)
        self.assertEqual(len(arin.memories), 61)
        self.assertIn("重要的承诺", [m.content for m in arin.memories])
        self.assertIn("闲聊 0", [m.content for m in arin.memories])

    def test_thousand_experiences_survive_save_load_with_bounded_attention(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.memories.clear()
        for i in range(1000):
            w.remember(arin, f"人生经历 {i}", cause="测试",
                       importance=0.2 + (i % 5) * 0.1)
        window = experience_window(
            arin, w.turn, query="人生经历 17", limit=12)
        self.assertEqual(len(arin.memories), 1000)
        self.assertLessEqual(len(window), 12)
        self.assertTrue(any("人生经历 17" in m.content for m in window))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep-person.json"
            Universe(worlds={w.name: w}, current=w.name).save(path)
            loaded = Universe.load(path).here.npcs[arin.id]
        self.assertEqual(len(loaded.memories), 1000)
        self.assertEqual(len({m.id for m in loaded.memories}), 1000)

    def test_effectiveness_decays_with_time(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.memories.clear()
        w.remember(arin, "旧事", cause="测试", importance=0.8)
        old = arin.memories[0]
        w.pass_time(200)
        self.assertLess(memory_effectiveness(old, w.turn),
                        old.importance)  # 衰减发生

    def test_dialogue_payload_uses_effective_retrieval(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.memories.clear()
        w.remember(arin, "重要旧事", cause="测试", importance=1.0)
        w.pass_time(30)  # 衰减一半多一点：1.0×e^(-30/48)≈0.54 > 0.3
        for i in range(5):
            w.remember(arin, f"新日常 {i}", cause="测试", importance=0.3)
        # 有效检索应把高重要度的旧事排进 top
        ranked = sorted(arin.memories,
                        key=lambda m: memory_effectiveness(m, w.turn),
                        reverse=True)
        self.assertIn("重要旧事", [m.content for m in ranked[:5]])

    def test_retrieval_window_keeps_recent_and_important(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.memories.clear()
        w.remember(arin, "旧而重要", cause="测试", importance=1.0)
        w.pass_time(30)
        for i in range(4):
            w.remember(arin, f"刚聊的 {i}", cause="测试", importance=0.2)
        win = retrieval_window(arin, w.turn)
        contents = [m.content for m in win]
        # 拟人化窗口：最近 3 条全在（话头接得上）
        for i in (3, 2, 1):
            self.assertIn(f"刚聊的 {i}", contents)
        # 旧而重要的补进来（伏笔不断线）
        self.assertIn("旧而重要", contents)
        # 体积恒定（recent 3 + fill 3 = 6，可重叠）
        self.assertLessEqual(len(win), 6)


class TestMoodDynamics(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        return llm, generate_world(llm, "世界1", DESC)

    def test_dialogue_pushes_mood_and_decays(self):
        llm, w = self._world()
        w.player["location"] = "s-cafe"
        arin = w.npcs["n-arin"]
        r = interpreter.dialogue_turn(llm, w, arin, "你在等谁？")
        self.assertGreater(arin.state.mood_value, 0)
        self.assertEqual(arin.state.mood_reason, "与玩家的对话：你在等谁？")
        self.assertEqual(arin.state.mood, evolution.mood_label(arin, "平静"))
        before = arin.state.mood_value
        w.pass_time(200)
        evolution.decay_mood(arin, 200)
        self.assertLess(arin.state.mood_value, before)

    def test_static_trait_no_longer_scales_mood(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.traits = {"沉静": True}
        v1 = evolution.push_mood(arin, 0.5, "测试")
        arin2 = w.npcs["n-zhou"]
        arin2.traits = {"健谈": True}
        v2 = evolution.push_mood(arin2, 0.5, "测试")
        self.assertEqual(v1, v2)  # 幅度由读过经历的裁决结果决定

    def test_mood_bonus_on_relationship(self):
        llm, w = self._world()
        w.player["location"] = "s-cafe"
        arin = w.npcs["n-arin"]
        evolution.push_mood(arin, 0.5, "测试")
        r = interpreter.dialogue_turn(llm, w, arin, "你在等谁？")
        # 心情好（>0.2）时关系涨得更多：2 + 1 = 3
        self.assertEqual(r.relationship, 3)


class TestRumor(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        return llm, generate_world(llm, "世界1", DESC)

    def test_rumor_reaches_nearby_npcs_with_distortion(self):
        llm, w = self._world()
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-cafe"
        arin = w.npcs["n-arin"]  # 咖啡店，与车站相邻
        w.pass_time(50)
        evolution.world_pulse(llm, w)
        rumors = [m for m in arin.memories if m.content.startswith(("听说", "都说"))]
        self.assertTrue(rumors)
        # 真相在场景状态与事件日志里：传闻与真相不同（带盐）
        self.assertTrue(any(i["name"] == "一封信"
                            for i in w.scenes["s-station"].items))
        event = next(e for e in w.events if "发现了一封信" in e.summary)
        self.assertNotIn(event.summary, [m.content for m in arin.memories])
        # 影响落账：记忆事件带原因（不再是硬编码的「流言」标签）
        self.assertTrue(any("事件：" in e.cause for e in w.events
                            if e.kind == "npc_memory"))


class TestDrive(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-station"
        return llm, w

    def test_goal_not_triggered_without_activation(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.pass_time(6)
        evolution.heartbeat(llm, w, arin)
        # 没听说信 → 条件不成熟，不主动
        self.assertFalse(any(e.kind == "npc_acted" for e in w.events))
        self.assertEqual(arin.goals[0]["progress"], 0.0)

    def test_goal_activated_by_rumor_triggers_action(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.remember(arin, "听说「一封信」出现了，可没人说得清是给谁的。",
                   cause="流言", importance=0.6)
        arin.relationship = 5
        w.pass_time(6)
        evolution.heartbeat(llm, w, arin)
        acted = [e for e in w.events if e.kind == "npc_acted"]
        self.assertTrue(acted)
        self.assertEqual(acted[-1].cause, "角色驱动力")
        self.assertEqual(arin.goals[0]["progress"], 0.5)
        self.assertEqual(arin.state.location, "s-cafe")  # 还在路上
        self.assertEqual(arin.state.action.location, "s-station")

    def test_existing_scene_misplaced_in_place_is_normalized(self):
        """已有场景 id 写进 place 仍要变成真实目的地，不能退化为原地动作。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        plan = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": arin.id, "action": "去旧车站找线索", "place": "s-station"}}],
            "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        evolution.propose_proactive(ScriptedLLM([plan]), w, arin)
        self.assertEqual(arin.state.action.location, "s-station")
        acted = next(e for e in reversed(w.events) if e.kind == "npc_acted")
        params = (acted.payload or {}).get("event_params", {})
        self.assertEqual(params.get("location"), "s-station")
        self.assertNotIn("place", params)

    def test_local_action_does_not_create_in_transit_state(self):
        """原地记事、犹豫或准备已经发生，不应成为下一轮必被中止的长动作。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        plan = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": arin.id,
                "action": "阿凛在咖啡店的账本上记下明天去旧车站的念头。",
                "location": "s-cafe"}}],
            "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        evolution.propose_proactive(ScriptedLLM([plan]), w, arin)
        self.assertEqual(arin.state.action.text, "")

    def test_travel_is_not_overwritten_by_new_goal_action(self):
        """未抵达前的新提案只能暂缓，不能把角色送到叙事里的目的地。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(text="去旧车站找线索",
                                        location="s-station", progress=0.4)
        before = len(w.events)
        plan = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": arin.id, "action": "推开旧车站的门，向老周说明线索",
                "location": "s-station"}}],
            "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        summaries = evolution.propose_proactive(ScriptedLLM([plan]), w, arin)
        self.assertEqual(arin.state.location, "s-cafe")
        self.assertEqual(arin.state.action.text, "去旧车站找线索")
        self.assertFalse(any(e.kind in ("npc_acted", "action_aborted")
                             for e in w.events[before:]))
        self.assertFalse(summaries)  # 已有承诺时不再询问模型，也不制造 UI 噪声

    def test_note_left_at_deep_relationship(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.remember(arin, "听说「一封信」出现了。", cause="流言",
                   importance=0.6)
        arin.goals[0]["progress"] = 0.5
        arin.relationship = 10
        w.pass_time(6)
        evolution.heartbeat(llm, w, arin)
        notes = [e for e in w.events if e.kind == "note_left"]
        self.assertTrue(notes)
        self.assertEqual(notes[-1].cause, "角色驱动力")
        self.assertEqual(arin.goals[0]["progress"], 1.0)

    def test_goals_survive_roundtrip_and_card(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.goals[0]["progress"] = 0.5
        card = cards.npc_card(arin)
        self.assertIn("goals", card)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.json"
            u = Universe(worlds={w.name: w}, current=w.name)
            u.save(path)
            u2 = Universe.load(path)
            self.assertEqual(u2.here.npcs["n-arin"].goals[0]["progress"],
                             0.5)


class TestGoalLifecycle(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-station"
        return llm, w

    def _prepare_completion(self, llm, w):
        arin = w.npcs["n-arin"]
        w.remember(arin, "听说「一封信」出现了。", cause="流言",
                   importance=0.6)
        arin.goals[0]["progress"] = 0.5
        arin.relationship = 10
        w.pass_time(6)
        evolution.heartbeat(llm, w, arin)
        return arin

    def test_completion_transforms_goal(self):
        llm, w = self._world()
        arin = self._prepare_completion(llm, w)
        kinds = [e.kind for e in w.events]
        self.assertIn("goal_completed", kinds)
        self.assertIn("goal_emerged", kinds)  # 新目标的出生也入账
        self.assertIn("note_left", kinds)
        # 信念蒸馏
        self.assertTrue(any("曾完成：" in b for b in arin.beliefs))
        # 新目标涌现（目标链生长）
        ids = [g["id"] for g in arin.goals]
        self.assertIn("find-sender", ids)
        # 旧目标留在库里（传记），标记完成
        done = [g for g in arin.goals if g["id"] == "find-letter"]
        self.assertEqual(done[0]["progress"], 1.0)
        self.assertTrue(done[0].get("done"))

    def test_completed_goal_not_reevaluated(self):
        llm, w = self._world()
        arin = self._prepare_completion(llm, w)
        note_count = sum(1 for e in w.events if e.kind == "note_left")
        w.pass_time(6)
        evolution.heartbeat(llm, w, arin)
        # 已完成目标不再触发重复事件
        self.assertEqual(sum(1 for e in w.events if e.kind == "note_left"),
                         note_count)

    def test_multi_goal_persist_roundtrip(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        self.assertGreaterEqual(len(arin.goals), 2)  # 多目标
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.json"
            u = Universe(worlds={w.name: w}, current=w.name)
            u.save(path)
            u2 = Universe.load(path)
            goals = u2.here.npcs["n-arin"].goals
            self.assertEqual(len(goals), len(arin.goals))
            self.assertIn("cafe-alive", [g["id"] for g in goals])


class TestOpener(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-cafe"  # 玩家在咖啡店（阿凛所在）
        return llm, w

    def test_opener_after_law_change(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        interpreter.change_law(llm, w, "这座城的人从不撒谎")
        w.pass_time(6)
        summaries = evolution.heartbeat(llm, w, arin)
        self.assertTrue(arin.state.pending_opener)
        self.assertTrue(any(e.kind == "npc_interaction"
                            and (e.payload or {}).get("event_params", {}).get(
                                "target") == "player"
                            for e in w.events))
        self.assertTrue(any("叫住了你" in s for s in summaries))

    def test_dialogue_clears_opener(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.pending_opener = "我有话想说。"
        interpreter.dialogue_turn(llm, w, arin, "什么事？")
        self.assertEqual(arin.state.pending_opener, "")

    def test_opener_cooldown(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        interpreter.change_law(llm, w, "这座城的人从不撒谎")
        w.pass_time(6)
        evolution.heartbeat(llm, w, arin)
        count = sum(1 for e in w.events if e.kind == "npc_interaction"
                    and (e.payload or {}).get("event_params", {}).get(
                        "target") == "player")
        w.pass_time(6)  # 还在冷却期
        evolution.heartbeat(llm, w, arin)
        self.assertEqual(
            sum(1 for e in w.events if e.kind == "npc_interaction"
                and (e.payload or {}).get("event_params", {}).get(
                    "target") == "player"), count)

    def test_opener_only_in_player_scene(self):
        llm, w = self._world()
        zhou = w.npcs["n-zhou"]  # 在车站，玩家在咖啡店
        interpreter.change_law(llm, w, "这座城的人从不撒谎")
        w.pass_time(6)
        evolution.heartbeat(llm, w, zhou)
        self.assertEqual(zhou.state.pending_opener, "")
        self.assertFalse(any(e.kind == "npc_interaction"
                             and (e.payload or {}).get("event_params", {}).get(
                                 "target") == "player"
                             for e in w.events))

    def test_opener_persists_roundtrip(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.pending_opener = "别走。"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.json"
            u = Universe(worlds={w.name: w}, current=w.name)
            u.save(path)
            u2 = Universe.load(path)
            self.assertEqual(u2.here.npcs["n-arin"].state.pending_opener,
                             "别走。")

    def test_world_pulse_ticks_the_player_scene_after_time_passes(self):
        """玩家在场不再是特例：真实时间过去后由统一脉冲裁决。"""
        llm, w = self._world()
        w.player["location"] = "s-cafe"
        arin = w.npcs["n-arin"]
        before = arin.state.last_clock
        w.pass_time(6, cause="测试等待")
        evolution.world_pulse(llm, w)
        self.assertGreater(arin.state.last_clock, before)

    def test_world_pulse_can_address_player_as_present_actor(self):
        """同场 NPC 可在统一脉冲中向玩家发起真实互动，不走 UI 特例。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        w.pass_time(6)
        plan = json.dumps({
            "events": [],
            "npc_plans": [{
                "npc": arin.id,
                "state": {"activity": "擦着吧台", "mood": "平静",
                          "location": "s-cafe"},
                "action": None,
                "interaction": {"with": "player",
                                "line": "你刚才一直看着那封信？"},
                "goal_updates": {}, "new_goals": [],
            }],
            "influences": [], "daily_bits": [], "new_npcs": [],
            "new_scenes": [], "item_patches": [], "crowds": [],
        }, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        event = next(e for e in w.events if e.kind == "npc_interaction"
                     and (e.payload or {}).get("event_params", {}).get(
                         "target") == "player")
        params = (event.payload or {}).get("event_params", {})
        self.assertEqual(params["location"], "s-cafe")
        self.assertIn("一直看着", params["line"])
        self.assertEqual(arin.state.pending_opener, params["line"])
        self.assertTrue(any("我对你说" in m.content for m in arin.memories))
        self.assertTrue(finalstory.player_visible(w, event, "s-cafe"))
        self.assertIn("对 你", finalstory.render(w, event))

    def test_player_interaction_requires_same_scene(self):
        """玩家是行动者，不是全知收件箱：异地搭话不成立。"""
        _, w = self._world()
        zhou = w.npcs["n-zhou"]
        self.assertEqual(evolution.start_player_interaction(
            w, zhou, "你在做什么？", "测试"), [])
        self.assertFalse(any(e.kind == "npc_interaction"
                             and (e.payload or {}).get("event_params", {}).get(
                                 "target") == "player"
                             for e in w.events))

    def test_player_departure_is_local_trace_not_global_coordinate(self):
        """离开被起点/终点看见，其他地方的人既看不见也拿不到实时坐标。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]      # 起点：咖啡店
        man = w.npcs["n-man"]        # 无关地点：小巷
        w.player["location"] = "s-station"
        w.log("scene_entered", "你离开咖啡店，去了旧车站", "测试",
              {"actor": "player", "from": "s-cafe", "to": "s-station",
               "scene": "s-station"})
        trace = evolution._player_traces(w, arin)
        self.assertEqual(trace[-1]["from"], "s-cafe")
        self.assertEqual(trace[-1]["to"], "s-station")
        self.assertEqual(evolution._player_traces(w, man), [])
        payload = evolution.build_pulse_payload(
            w, [arin, man], evolution.scene_distances(w, "s-station"),
            "s-station")
        self.assertNotIn("player_location", payload["world"])
        self.assertEqual(payload["due_npcs"][0]["player"]
                         ["recent_visible_actions"][-1]["to"], "s-station")


class TestIntentState(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-station"
        return llm, w

    def test_intent_is_a_ledgered_state_with_real_targets(self):
        """短期打算落进角色库和账本，但不会被当作已发生的行动。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        summaries = evolution.set_intent(
            w, arin,
            {"text": "黄昏去旧车站核对那封信", "targets": ["一封信", "旧车站"]},
            cause="测试")
        self.assertTrue(summaries)
        self.assertEqual(arin.state.intent.text, "黄昏去旧车站核对那封信")
        self.assertEqual(set(arin.state.intent.targets),
                         {"item:i-letter", "scene:s-station"})
        self.assertEqual(arin.state.location, "s-cafe")
        event = w.events[-1]
        self.assertEqual(event.kind, "npc_intent")
        self.assertEqual(event.payload["event_params"]["targets"],
                         arin.state.intent.targets)

    def test_intent_can_be_revised_then_released(self):
        """意图没有类型枚举，只有同一个状态的形成、替换和放下。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        evolution.set_intent(w, arin, {"text": "傍晚去旧车站", "targets": ["旧车站"]},
                             cause="测试")
        evolution.set_intent(w, arin, {"text": "先在咖啡店查账本", "targets": ["scene:s-cafe"]},
                             cause="新线索")
        self.assertEqual(arin.state.intent.text, "先在咖啡店查账本")
        self.assertIn("改了打算", w.events[-1].summary)
        evolution.set_intent(w, arin, None, cause="店门提前打烊")
        self.assertEqual(arin.state.intent.text, "")
        self.assertIn("放下了打算", w.events[-1].summary)

    def test_intent_rejects_unknown_targets(self):
        _, w = self._world()
        arin = w.npcs["n-arin"]
        before = len(w.events)
        summaries = evolution.set_intent(
            w, arin, {"text": "去不存在的塔", "targets": ["scene:s-nowhere"]}, cause="测试")
        self.assertTrue(any("真实实体" in s for s in summaries))
        self.assertEqual(arin.state.intent.text, "")
        self.assertEqual(len(w.events), before)

    def test_local_intent_does_not_start_travel(self):
        """决定以后去某处只是角色状态，不能由引擎替她移动。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        plan = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": arin.id, "action": "在咖啡店的账本上记下明天去旧车站的打算。"}}],
            "intent": {"text": "明天去旧车站核对信上的日期",
                       "targets": ["scene:s-station", "item:i-letter"]},
            "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        evolution.propose_proactive(ScriptedLLM([plan]), w, arin)
        self.assertEqual(arin.state.location, "s-cafe")
        self.assertEqual(arin.state.action.text, "")
        self.assertEqual(arin.state.intent.text, "明天去旧车站核对信上的日期")

    def test_rejected_action_cannot_clear_existing_intent(self):
        _, w = self._world()
        arin = w.npcs["n-arin"]
        evolution.set_intent(w, arin, {"text": "等雨小些再去旧车站", "targets": ["旧车站"]},
                             cause="测试")
        plan = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": arin.id, "action": "走向旧车站的站台", "location": ""}}],
            "intent": None, "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        evolution.propose_proactive(ScriptedLLM([plan]), w, arin)
        self.assertEqual(arin.state.intent.text, "等雨小些再去旧车站")

    def test_intent_cannot_bypass_its_state_writer(self):
        """注册事件只是账本形状，模型必须走计划的 intent 字段同步状态。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        plan = json.dumps({
            "events": [{"type": "npc_intent", "params": {
                "npc": arin.id, "intent": "去旧车站", "targets": ["scene:s-station"]}}],
            "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        summaries = evolution.propose_proactive(ScriptedLLM([plan]), w, arin)
        self.assertTrue(any("intent 字段" in s for s in summaries))
        self.assertEqual(arin.state.intent.text, "")
        self.assertFalse(any(e.kind == "npc_intent" for e in w.events))

    def test_intent_persists_roundtrip(self):
        _, w = self._world()
        arin = w.npcs["n-arin"]
        evolution.set_intent(w, arin, {"text": "晚些时候查看车站时钟", "targets": ["旧车站"]},
                             cause="测试")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.json"
            Universe(worlds={w.name: w}, current=w.name).save(path)
            restored = Universe.load(path).here.npcs["n-arin"]
        self.assertEqual(restored.state.intent.text, "晚些时候查看车站时钟")
        self.assertEqual(restored.state.intent.targets, ["scene:s-station"])


class TestActionState(unittest.TestCase):
    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-station"
        return llm, w

    def _trigger_act(self, llm, w):
        arin = w.npcs["n-arin"]
        w.remember(arin, "听说「一封信」出现了。", cause="流言",
                   importance=0.6)
        arin.relationship = 5
        w.pass_time(6)
        evolution.heartbeat(llm, w, arin)
        return arin

    def test_act_sets_action_state(self):
        llm, w = self._world()
        arin = self._trigger_act(llm, w)
        self.assertTrue(arin.state.action.text)
        self.assertEqual(arin.state.action.location, "s-station")

    def test_action_advances_and_resolves(self):
        llm, w = self._world()
        arin = self._trigger_act(llm, w)
        w.pass_time(24)  # 一天：动作应该完成
        evolution.heartbeat(llm, w, arin)
        kinds = [e.kind for e in w.events]
        self.assertIn("action_done", kinds)
        self.assertEqual(arin.state.action.text, "")  # 清空
        self.assertTrue(any("我做完了这件事" in m.content
                           for m in arin.memories))

    def test_travel_arrives_only_when_action_completes(self):
        """去找人先在路上；抵达后才改变位置，才有可能同场互动。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        self.assertEqual(arin.state.location, "s-cafe")
        arin.state.action = ActionState(
            text="去旧车站找刚离开的玩家", location="s-station",
            progress=0.5)
        w.pass_time(6)
        evolution.advance_action(llm, w, arin, 6)
        self.assertEqual(arin.state.location, "s-cafe")
        w.pass_time(6)
        done = json.dumps({"outcome": "赶到站台，却只看见还在滴水的长椅。",
                           "patch": None}, ensure_ascii=False)
        evolution.advance_action(ScriptedLLM([done]), w, arin, 6)
        self.assertEqual(arin.state.location, "s-station")
        self.assertTrue(any(e.kind == "npc_moved" and e.cause == "角色抵达"
                            for e in w.events))

    def test_action_persists_roundtrip(self):
        llm, w = self._world()
        arin = self._trigger_act(llm, w)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.json"
            u = Universe(worlds={w.name: w}, current=w.name)
            u.save(path)
            u2 = Universe.load(path)
            a2 = u2.here.npcs["n-arin"]
            self.assertEqual(a2.state.action.text, arin.state.action.text)
            self.assertEqual(a2.state.action.started_clock,
                             arin.state.action.started_clock)
            self.assertEqual(a2.state.action.due_clock,
                             arin.state.action.due_clock)

    def test_action_resolves_only_at_its_exact_due_clock(self):
        """动作自己的时长决定完成点；到点前不调用结局模型。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        plan = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": arin.id, "action": "在咖啡店清点旧账",
                "location": "s-cafe", "days": 0.5,
                "targets": ["scene:s-cafe"]}}],
            "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        evolution.propose_proactive(ScriptedLLM([plan]), w, arin)
        self.assertAlmostEqual(arin.state.action.due_clock
                               - arin.state.action.started_clock, 0.5)
        self.assertTrue(arin.state.action.text)  # 原地持续动作不再当场消失

        class FailIfCalled:
            def chat_json(self, system, user):
                raise AssertionError("到期前不应调用动作结局模型")

        w.pass_time(11)
        evolution.advance_action(FailIfCalled(), w, arin, 11)
        self.assertTrue(arin.state.action.text)
        w.pass_time(1)
        done = json.dumps({"outcome": "旧账清点完毕。", "patch": None},
                          ensure_ascii=False)
        evolution.advance_action(ScriptedLLM([done]), w, arin, 1)
        self.assertFalse(arin.state.action.text)
        self.assertIn("action_done", [e.kind for e in w.events])

    def test_action_window_defers_until_its_world_clock(self):
        """未来时刻的动作先落为打算，到钟后才成为进行中的事实。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        first = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": arin.id, "action": "黄昏去咖啡店关窗",
                "location": "s-cafe", "days": 0.25,
                "earliest_clock": 0.5, "latest_clock": 0.75,
                "targets": ["scene:s-cafe"]}}],
            "intent": {"text": "黄昏去咖啡店关窗",
                       "targets": ["scene:s-cafe"],
                       "earliest_clock": 0.5, "latest_clock": 0.75},
            "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        summaries = evolution.propose_proactive(ScriptedLLM([first]), w, arin)
        self.assertFalse(arin.state.action.text)
        self.assertEqual(arin.state.intent.earliest_clock, 0.5)
        self.assertTrue(any("尚未到开始时刻" in s for s in summaries))
        self.assertFalse(any(e.kind == "npc_acted" for e in w.events))

        w.pass_time(12)  # 世界钟到 0.5，进入允许窗口
        second = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": arin.id, "action": "黄昏去咖啡店关窗",
                "location": "s-cafe", "days": 0.25,
                "earliest_clock": 0.5, "latest_clock": 0.75,
                "targets": ["scene:s-cafe"]}}],
            "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        evolution.propose_proactive(ScriptedLLM([second]), w, arin)
        self.assertTrue(arin.state.action.text)
        self.assertEqual(arin.state.action.started_clock, 0.5)

    def test_action_window_rejects_after_latest_start(self):
        """错过窗口的动作不能在深夜被追认成按时完成。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        w.pass_time(24)  # 当前世界钟 1.0，窗口已经结束
        plan = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": arin.id, "action": "黄昏去咖啡店关窗",
                "location": "s-cafe", "days": 0.25,
                "earliest_clock": 0.5, "latest_clock": 0.75,
                "targets": ["scene:s-cafe"]}}],
            "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        summaries = evolution.propose_proactive(ScriptedLLM([plan]), w, arin)
        self.assertFalse(arin.state.action.text)
        self.assertTrue(any("错过开始时刻" in s for s in summaries))
        self.assertFalse(any(e.kind == "npc_acted" for e in w.events))

    def test_world_pulse_uses_the_same_action_window(self):
        """统一脉冲不能绕过单人主动裁决的时间闸。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        w.pass_time(6)  # 当前 0.25，动作窗口从 0.5 开始
        arin.state.last_clock = 0.0
        pulse = json.dumps({
            "events": [], "entity_events": [],
            "npc_plans": [{"npc": arin.id,
                "state": {"activity": "等待黄昏", "mood": "平静",
                          "location": "s-cafe"},
                "action": {"type": "npc_acted", "params": {
                    "npc": arin.id, "action": "黄昏关窗",
                    "location": "s-cafe", "days": 0.25,
                    "earliest_clock": 0.5, "latest_clock": 0.75,
                    "targets": ["scene:s-cafe"]}},
                "intent": {"text": "黄昏关窗",
                           "targets": ["scene:s-cafe"],
                           "earliest_clock": 0.5, "latest_clock": 0.75},
                "goal_updates": {}, "new_goals": []}],
            "daily_bits": [], "new_npcs": [], "new_scenes": [],
            "item_patches": [], "crowds": [],
        }, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([pulse]), w)
        self.assertFalse(arin.state.action.text)
        self.assertEqual(arin.state.intent.earliest_clock, 0.5)
        self.assertTrue(any("尚未到开始时刻" in s for s in summaries))
        self.assertFalse(any(e.kind == "npc_acted" for e in w.events))

    def test_missing_required_target_aborts_before_resolution(self):
        """到期前目标消失时，中止引用真实变更，且不调用结局模型。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        plan = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": arin.id, "action": "去车站取那封信",
                "location": "s-station", "days": 0.25,
                "targets": ["item:i-letter"],
                "requires": ["item:i-letter"]}}],
            "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        evolution.propose_proactive(ScriptedLLM([plan]), w, arin)
        errors = apply_item_patch(w, {
            "op": "remove", "item": "i-letter", "location": "s-station",
            "note": "被别人带走了",
        }, cause="测试移走目标")
        self.assertFalse(errors)

        class FailIfCalled:
            def chat_json(self, system, user):
                raise AssertionError("前置条件失效后不应调用结局模型")

        w.pass_time(6)
        summaries = evolution.advance_action(FailIfCalled(), w, arin, 6)
        self.assertFalse(arin.state.action.text)
        self.assertNotIn("action_done", [e.kind for e in w.events])
        aborted = next(e for e in reversed(w.events)
                       if e.kind == "action_aborted")
        self.assertIn("被别人带走", aborted.cause)
        self.assertTrue(any("已不存在" in s for s in summaries))

    def test_active_action_defers_new_proposal(self):
        """进行中的承诺不能被同一角色下一次提案无因覆盖。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(text="在咖啡店整理账本",
                                        location="s-cafe", progress=0.0)
        # 新动作顶上来（ScriptedLLM 灌一个 npc_acted）
        goal = json.dumps({
            "events": [{"type": "npc_acted",
                        "params": {"npc": "n-arin",
                                   "action": "去咖啡店擦桌子",
                                   "location": "s-cafe"}}],
            "goal_updates": {}, "new_goals": []}, ensure_ascii=False)
        summaries = evolution.propose_proactive(ScriptedLLM([goal]), w, arin)
        self.assertNotIn("action_aborted", [e.kind for e in w.events])
        self.assertEqual(arin.state.action.text, "在咖啡店整理账本")
        self.assertFalse(summaries)

    def test_action_outcome_cannot_smuggle_npc_encounter(self):
        """动作结局在移动前审计，不能用 outcome 伪造与另一人的相见。"""
        llm, w = self._world()
        arin, zhou = w.npcs["n-arin"], w.npcs["n-zhou"]
        evolution.move_npc(w, zhou, "s-station", cause="测试同场")
        arin.state.action = ActionState(text="去旧车站找线索",
                                        location="s-station", progress=0.99)
        w.pass_time(6)
        result = json.dumps({
            "outcome": "阿凛来到旧车站，与老周短暂相见并交谈。",
            "patch": None}, ensure_ascii=False)
        summaries = evolution.advance_action(ScriptedLLM([result]), w, arin, 6)
        self.assertTrue(any("驳回动作结局" in s for s in summaries))
        self.assertEqual(arin.state.location, "s-cafe")
        self.assertTrue(arin.state.action.text)
        self.assertFalse(any(e.kind == "action_done" for e in w.events))

    def test_pulse_advances_detached_actions(self):
        """远处角色的进行中动作：世界脉冲里随回合推进直至完成。"""
        llm, w = self._world()
        arin = self._trigger_act(llm, w)  # 她去了车站打听信
        arin.state.action.started_clock = w.clock
        arin.state.action.due_clock = w.clock + 0.25  # 再过四分之一天到期
        w.player["location"] = "s-alley"  # 玩家走开，她进入到期名单
        w.pass_time(12)                    # 世界时间走
        evolution.world_pulse(llm, w)
        kinds = [e.kind for e in w.events]
        self.assertIn("action_done", kinds)  # 没人看她，她也做完了
        self.assertEqual(arin.state.action.text, "")
        self.assertTrue(any("我做完了这件事" in m.content
                            for m in arin.memories))

    def test_pulse_resolves_due_action_without_a_returned_npc_plan(self):
        """世界钟到点就结算，不能等模型再次想起这个人。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(
            text="在咖啡店清点旧账", location="s-cafe",
            started_clock=w.clock, due_clock=w.clock + 0.25,
            source_turn=w.turn,
        )
        w.pass_time(6)
        pulse = json.dumps({
            "events": [], "entity_events": [], "npc_plans": [],
            "daily_bits": [], "new_npcs": [], "new_scenes": [],
            "item_patches": [], "crowds": [],
        }, ensure_ascii=False)
        done = json.dumps({"outcome": "旧账已经清点完毕。", "patch": None},
                          ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([pulse, done]), w)
        self.assertFalse(arin.state.action.text)
        self.assertIn("action_done", [e.kind for e in w.events])

    def test_pulse_defers_plan_action_built_from_pre_completion_state(self):
        """脉冲计划看到旧动作时，不能在同轮结算后立刻覆盖新动作。"""
        _, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(
            text="在咖啡店清点旧账", location="s-cafe",
            started_clock=w.clock, due_clock=w.clock + 0.25,
            source_turn=w.turn,
        )
        w.pass_time(6)
        pulse = json.dumps({
            "events": [], "entity_events": [],
            "npc_plans": [{
                "npc": arin.id,
                "state": {"activity": "整理柜台", "mood": "平静",
                          "location": "s-cafe"},
                "action": {"type": "npc_acted", "params": {
                    "npc": arin.id, "action": "再清点一遍旧账",
                    "location": "s-cafe", "days": 0.5}},
                "goal_updates": {}, "new_goals": [],
            }],
            "daily_bits": [], "new_npcs": [], "new_scenes": [],
            "item_patches": [], "crowds": [],
        }, ensure_ascii=False)
        done = json.dumps({"outcome": "旧账已经清点完毕。", "patch": None},
                          ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([pulse, done]), w)
        self.assertFalse(arin.state.action.text)
        self.assertFalse(any(e.kind == "npc_acted" for e in w.events))
        self.assertTrue(any("新动作暂缓" in s for s in summaries))

    def test_action_completion_wakes_arrival_scene(self):
        """抵达不是只改坐标：完成动作后，抵达者和现场者都应获下一次裁决机会。"""
        _, w = self._world()
        arin, zhou = w.npcs["n-arin"], w.npcs["n-zhou"]
        evolution.move_npc(w, zhou, "s-station", cause="测试布置")
        w.wakeups.clear()
        arin.state.action = ActionState(
            text="去旧车站查看信箱", location="s-station",
            started_clock=w.clock, due_clock=w.clock + 0.25,
            source_turn=w.turn,
        )
        w.pass_time(6)
        done = json.dumps({"outcome": "信箱已经检查完毕。", "patch": None},
                          ensure_ascii=False)
        evolution.advance_action(ScriptedLLM([done]), w, arin, 6)

        finished = next(e for e in reversed(w.events) if e.kind == "action_done")
        self.assertEqual(finished.payload["event_params"]["location"], "s-station")
        self.assertIn(arin.id, w.wakeups)
        self.assertIn(zhou.id, w.wakeups)


class TestEventWakeups(unittest.TestCase):
    """事件相关者优先进入下一次裁决窗口，但没有任何强制行为配额。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        return w

    def test_item_change_wakes_local_and_interested_npcs(self):
        w = self._world()
        arin, zhou = w.npcs["n-arin"], w.npcs["n-zhou"]
        evolution.move_npc(w, arin, "s-station", cause="测试布置")
        evolution.move_npc(w, zhou, "s-cafe", cause="测试布置")
        zhou.goals = [{"id": "g-watch", "text": "等那封信的变化",
                       "progress": 0.0, "targets": ["item:i-letter"]}]
        w.wakeups.clear()

        errors = apply_item_patch(w, {
            "op": "change", "item": "i-letter", "location": "s-station",
            "note": "信封边缘被雨水浸湿",
        }, cause="一阵急雨")

        self.assertFalse(errors)
        self.assertIn(arin.id, w.wakeups)  # 同场看见
        self.assertIn(zhou.id, w.wakeups)  # 有目标引用

    def test_woken_npc_enters_pulse_before_regular_interval(self):
        """玩家等待一小段时间时，远处被事件碰到的人不用等完整生活周期。"""
        w = self._world()
        arin = w.npcs["n-arin"]
        w.player["location"] = "s-alley"
        # 断开地图可达性，使车站角色的普通轮询间隔为 24 回合而非 6 回合。
        w.scenes["s-alley"].exits = []
        arin.state.last_clock = w.clock
        arin.state.last_turn = w.turn
        w.wakeups = {arin.id: w.turn}
        w.pass_time(6)
        plan = json.dumps({
            "events": [], "entity_events": [],
            "npc_plans": [{
                "npc": arin.id,
                "state": {"activity": "重新读了一遍那封信", "mood": "平静",
                          "location": arin.state.location},
                "action": None, "interaction": None,
                "goal_updates": {}, "new_goals": [],
            }],
            "daily_bits": [], "new_npcs": [], "new_scenes": [],
            "item_patches": [], "crowds": [],
        }, ensure_ascii=False)

        evolution.world_pulse(ScriptedLLM([plan]), w)

        self.assertEqual(arin.state.activity, "重新读了一遍那封信")
        self.assertNotIn(arin.id, w.wakeups)

    def test_wakeup_queue_survives_roundtrip(self):
        """未处理的事件机会不能因存档读档而被静默遗失。"""
        w = self._world()
        arin = w.npcs["n-arin"]
        w.wakeups = {arin.id: w.turn + 3}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.json"
            Universe(worlds={w.name: w}, current=w.name).save(path)
            restored = Universe.load(path).here
        self.assertEqual(restored.wakeups, {arin.id: w.turn + 3})


class TestEntityEventTransaction(unittest.TestCase):
    """一个事实同时改变物品与行动者，任何后果失败都不留半截世界。"""

    def _world_with_car(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-cafe"
        errors = apply_item_patch(w, {
            "op": "add", "item": "i-car", "name": "旧汽车",
            "location": "s-cafe", "note": "发动机运转，正在向前行驶",
        }, cause="测试布置")
        self.assertFalse(errors)
        return w

    @staticmethod
    def _collision(item="i-car"):
        return {
            "title": "车辆碰撞",
            "detail": "行驶中的旧汽车撞上了站在行进路径上的玩家",
            "location": "s-cafe",
            "intensity": 0.9,
            "participants": [f"item:{item}", "player", "scene:s-cafe"],
            "item_patches": [{
                "op": "change", "item": item, "location": "s-cafe",
                "note": "急停后车头凹陷，发动机熄火",
            }],
            "actor_patches": [{
                "target": "player", "can_act": False,
                "condition": "碰撞后昏迷",
            }],
        }

    def test_moving_car_and_player_commit_as_one_world_fact(self):
        w = self._world_with_car()
        w.pass_time(6)
        plan = json.dumps({
            "events": [], "entity_events": [self._collision()],
            "npc_plans": [], "daily_bits": [], "new_npcs": [],
            "new_scenes": [], "item_patches": [], "crowds": [],
        }, ensure_ascii=False)
        before = len(w.events)
        evolution.world_pulse(ScriptedLLM([plan]), w)

        root = w.events[before]
        self.assertEqual(root.kind, "world_event")
        self.assertEqual(root.payload["event_params"]["refs"],
                         ["item:i-car", "player", "scene:s-cafe"])
        self.assertIn("actor_patches", root.payload["consequences"])
        car = next(i for i in w.scenes["s-cafe"].items
                   if i.get("id") == "i-car")
        self.assertIn("车头凹陷", car["note"])
        self.assertFalse(w.player["can_act"])
        self.assertEqual(w.player["condition"], "碰撞后昏迷")
        self.assertEqual(w.player["condition_cause_turn"], root.turn)
        arin = w.npcs["n-arin"]
        self.assertTrue(any("我看见：车辆碰撞" in memory.content
                            for memory in arin.memories))
        changed = next(e for e in w.events[before:]
                       if e.kind == "item_changed")
        self.assertTrue(changed.cause.startswith(f"事件#{root.turn}："))
        result = interpreter.player_action(
            ScriptedLLM([]), w, w.npcs["n-arin"], "站起来")
        self.assertFalse(result.accepted)
        self.assertIn("昏迷", result.reply)
        old_location = w.player["location"]
        errors = evolution.move_player(w, "s-station")
        self.assertTrue(any("无法移动" in error for error in errors))
        self.assertEqual(w.player["location"], old_location)

    def test_world_pulse_can_validate_entity_event_with_state_runtime(self):
        """真实心跳路径可选择启用 StateRuntime 的无副作用校验。"""
        try:
            import state_runtime  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))
        w = self._world_with_car()
        w.pass_time(6)
        plan = json.dumps({
            "events": [], "entity_events": [self._collision()],
            "npc_plans": [], "daily_bits": [], "new_npcs": [],
            "new_scenes": [], "item_patches": [], "crowds": [],
        }, ensure_ascii=False)
        before = len(w.events)
        summaries = evolution.world_pulse(
            ScriptedLLM([plan]), w, use_state_runtime=True)
        self.assertFalse(any("StateRuntime" in s for s in summaries), summaries)
        root = w.events[before]
        audit = root.payload["state_runtime"]
        self.assertEqual(audit["status"], "validated")
        self.assertEqual(audit["mode"], "validator")
        self.assertFalse(any(
            e.payload.get("state_runtime") for e in w.events[before + 1:]
        ))

    def test_world_pulse_runtime_validation_rejects_without_partial_write(self):
        """心跳入口的悬空引用/错场景均零写入。"""
        try:
            import state_runtime  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))
        cases = [("missing", "i-ghost-car"), ("wrong_location", "i-car")]
        for kind, car_id in cases:
            with self.subTest(kind=kind):
                w = self._world_with_car()
                if kind == "wrong_location":
                    w.player["location"] = "s-station"
                w.pass_time(6)
                plan = json.dumps({
                    "events": [],
                    "entity_events": [self._collision(item=car_id)],
                    "npc_plans": [], "daily_bits": [], "new_npcs": [],
                    "new_scenes": [], "item_patches": [], "crowds": [],
                }, ensure_ascii=False)
                before_events = len(w.events)
                before_note = next(
                    i["note"] for i in w.scenes["s-cafe"].items
                    if i.get("id") == "i-car")
                summaries = evolution.world_pulse(
                    ScriptedLLM([plan]), w, use_state_runtime=True)
                self.assertTrue(any("驳回" in s for s in summaries), summaries)
                self.assertEqual(len(w.events), before_events)
                self.assertEqual(next(
                    i["note"] for i in w.scenes["s-cafe"].items
                    if i.get("id") == "i-car"), before_note)
                self.assertNotIn("can_act", w.player)
                self.assertFalse(w.scenes["s-cafe"].state_facts)

    def test_collision_adds_expiring_local_scene_consequence_atomically(self):
        """同一事件可留下局部短时环境后果，并由世界钟自动结束。"""
        w = self._world_with_car()
        proposal = self._collision()
        proposal["scene_state_patches"] = [{
            "scene": "scene:s-cafe", "op": "add",
            "fact": "collision-water-mark",
            "text": "碰撞点留下短暂的水花和轮胎水痕",
            "duration_days": 0.5,
        }]

        before = len(w.events)
        summaries = evolution.commit_entity_event(w, proposal)

        self.assertFalse(any("驳回多实体事件" in text for text in summaries))
        scene = w.scenes["s-cafe"]
        mark = next(fact for fact in scene.state_facts
                    if fact["id"] == "collision-water-mark")
        self.assertAlmostEqual(mark["expires_clock"] - w.clock, 0.5)
        root = next(e for e in w.events[before:]
                    if e.kind == "world_event")
        self.assertEqual(
            root.payload["consequences"]["scene_state_patches"][0]["fact"],
            "collision-water-mark")
        self.assertTrue(any(
            e.kind == "scene_state_changed"
            and e.payload.get("event_params", {}).get("op") == "add"
            for e in w.events[before:]))
        restored = World.from_dict(w.to_dict())
        restored_mark = next(
            fact for fact in restored.scenes["s-cafe"].state_facts
            if fact["id"] == "collision-water-mark")
        self.assertAlmostEqual(restored_mark["expires_clock"],
                               mark["expires_clock"])

        # 半天后由普通世界心跳清除局部水痕；车辆与玩家的持续后果不被清除。
        w.pass_time(12)
        plan = json.dumps({
            "events": [], "entity_events": [],
            "npc_plans": [], "daily_bits": [], "new_npcs": [],
            "new_scenes": [], "item_patches": [], "crowds": [],
        }, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)

        self.assertFalse(any(fact["id"] == "collision-water-mark"
                             for fact in scene.state_facts))
        self.assertTrue(any(
            e.kind == "scene_state_changed"
            and e.payload.get("event_params", {}).get("op") == "remove"
            and e.payload.get("event_params", {}).get("fact")
            == "collision-water-mark"
            for e in w.events[before:]))
        car = next(i for i in scene.items if i.get("id") == "i-car")
        self.assertIn("车头凹陷", car["note"])
        self.assertFalse(w.player["can_act"])

    def test_world_pulse_accepts_prefixed_scene_state_reference(self):
        """普通心跳的局部状态提案与原子事件共享 scene:id 引用语义。"""
        w = self._world_with_car()
        w.pass_time(6)
        plan = json.dumps({
            "events": [], "entity_events": [],
            "scene_state_patches": [{
                "scene": "scene:s-cafe", "op": "add",
                "fact": "brief-fog", "text": "门口停着一小片雾",
                "duration_days": 0.25, "why": "冷空气遇到热车"
            }],
            "npc_plans": [], "daily_bits": [], "new_npcs": [],
            "new_scenes": [], "item_patches": [], "crowds": [],
        }, ensure_ascii=False)

        summaries = evolution.world_pulse(ScriptedLLM([plan]), w)

        self.assertFalse(any("驳回局部场景状态" in text
                             for text in summaries))
        self.assertTrue(any(fact["id"] == "brief-fog"
                            for fact in w.scenes["s-cafe"].state_facts))

    def test_collision_rejects_an_actionable_npc_condition_patch(self):
        """持续伤口不能伪装成行动资格变化，必须进入状态事实库。"""
        w = self._world_with_car()
        proposal = self._collision()
        proposal["participants"] = ["item:i-car", "npc:n-arin", "scene:s-cafe"]
        proposal["actor_patches"] = [{
            "target": "npc:n-arin", "can_act": True,
            "condition": "手臂擦伤",
        }]

        summaries = evolution.commit_entity_event(w, proposal)

        arin = w.npcs["n-arin"]
        self.assertTrue(any("行动资格未变化" in text for text in summaries))
        self.assertTrue(arin.state.can_act)
        self.assertEqual(arin.state.condition, "")
        self.assertTrue(evolution.is_actionable(arin))

    def test_collision_can_atomically_add_an_npc_state_fact(self):
        """伤口等持续后果进入角色自己的状态库，且与物态同批提交。"""
        w = self._world_with_car()
        proposal = self._collision()
        proposal["participants"] = ["item:i-car", "npc:n-arin", "scene:s-cafe"]
        proposal["actor_patches"] = []
        proposal["state_fact_patches"] = [{
            "npc": "npc:n-arin", "op": "add", "fact": "injured-arm",
            "text": "左臂有一道新擦伤", "review_days": 2,
        }]

        summaries = evolution.commit_entity_event(w, proposal)

        self.assertFalse(any("驳回多实体事件" in text for text in summaries))
        arin = w.npcs["n-arin"]
        injury = next(fact for fact in arin.state.facts
                       if fact["id"] == "injured-arm")
        self.assertEqual(injury["text"], "左臂有一道新擦伤")
        self.assertAlmostEqual(injury["review_clock"] - w.clock, 2.0)
        car = next(i for i in w.scenes["s-cafe"].items
                   if i.get("id") == "i-car")
        self.assertIn("发动机熄火", car["note"])
        root = next(e for e in w.events if e.kind == "world_event"
                    and e.summary.startswith("车辆碰撞"))
        self.assertEqual(root.payload["consequences"]["state_fact_patches"][0]
                         ["fact"], "injured-arm")

    def test_item_action_uses_clock_and_completes_through_transaction(self):
        w = self._world_with_car()
        start = {
            "title": "汽车开始前进",
            "detail": "旧汽车发动后沿街道持续向前行驶",
            "location": "s-cafe", "intensity": 0.5,
            "participants": ["item:i-car", "scene:s-cafe"],
            "item_patches": [{
                "op": "change", "item": "i-car", "location": "s-cafe",
                "note": "发动机轰鸣，正在沿街道前进",
            }],
            "actor_patches": [],
            "item_actions": [{
                "item": "i-car", "text": "沿街道继续前进",
                "days": 0.25, "targets": ["player"], "requires": [],
            }],
            "completes": [],
        }
        evolution.commit_entity_event(w, start)
        car = next(i for i in w.scenes["s-cafe"].items
                   if i.get("id") == "i-car")
        self.assertAlmostEqual(car["action"]["due_clock"]
                               - car["action"]["started_clock"], 0.25)
        w.pass_time(5)
        self.assertFalse(evolution.due_item_actions(w))
        w.pass_time(1)
        self.assertEqual(evolution.due_item_actions(w)[0]["ref"], "item:i-car")

        finish = self._collision()
        finish["completes"] = ["item:i-car"]
        plan = json.dumps({
            "events": [], "entity_events": [finish],
            "npc_plans": [], "daily_bits": [], "new_npcs": [],
            "new_scenes": [], "item_patches": [], "crowds": [],
        }, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertFalse(car.get("action"))
        self.assertIn("车头凹陷", car["note"])
        self.assertFalse(w.player["can_act"])

    def test_item_action_cannot_complete_without_a_consequence(self):
        w = self._world_with_car()
        car = next(i for i in w.scenes["s-cafe"].items
                   if i.get("id") == "i-car")
        car["action"] = {
            "text": "沿街道继续前进", "started_clock": w.clock - 0.25,
            "due_clock": w.clock, "targets": [], "requires": [],
            "source_turn": w.turn,
        }
        before_events = len(w.events)
        summaries = evolution.commit_entity_event(w, {
            "title": "汽车抵达", "detail": "旧汽车抵达了道路尽头",
            "location": "s-cafe", "intensity": 0.3,
            "participants": ["item:i-car", "scene:s-cafe"],
            "item_patches": [], "actor_patches": [], "item_actions": [],
            "completes": ["item:i-car"],
        })
        self.assertTrue(any("必须同时落库实际后果" in s for s in summaries))
        self.assertEqual(len(w.events), before_events)
        self.assertTrue(car["action"])

    def test_failed_consequence_leaves_no_partial_world(self):
        w = self._world_with_car()
        proposal = self._collision()
        # 所有实体引用都合法，但物品 change 没有任何实际内容。预演时玩家
        # 状态会先在副本改变，随后物品后果失败；真实世界必须仍然零写入。
        proposal["item_patches"] = [{
            "op": "change", "item": "i-car", "location": "s-cafe",
        }]
        before_events = len(w.events)
        before_player = dict(w.player)
        summaries = evolution.commit_entity_event(w, proposal)
        self.assertTrue(any("驳回多实体事件" in s for s in summaries))
        self.assertEqual(len(w.events), before_events)
        self.assertEqual(w.player, before_player)
        car = next(i for i in w.scenes["s-cafe"].items
                   if i.get("id") == "i-car")
        self.assertIn("正在向前行驶", car["note"])


class TestRainFixtureSample(unittest.TestCase):
    """雨城 = 黄金样本：已知输入的确定性夹具（不是主线，是校验手段）。

    这条测试把样本的「长相」钉死——谁动雨城套件，这里立刻叫。
    引擎不认识雨城；雨城只是 demo 传进去的一个描述参数。
    """

    def test_rain_sample_looks_exactly_as_pinned(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        self.assertEqual(w.law_profile.atmosphere, "雨·永续")
        self.assertEqual([l.id for l in w.law_profile.laws], ["lie"])
        self.assertEqual({s.name for s in w.scenes.values()},
                         {"旧车站", "街角咖啡店", "积水小巷"})
        self.assertEqual({n.name for n in w.npcs.values()},
                         {"阿凛", "老周", "小满"})
        # 样本的角色弧线：查信链 + 等信
        arin = w.npcs["n-arin"]
        self.assertEqual([g["id"] for g in arin.goals],
                         ["find-letter", "cafe-alive"])

    def test_rain_sample_full_run_healthy(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-alley"  # 玩家在雾中小巷：所有 NPC 都进裁决窗口
        for _ in range(16):  # 96 回合：日历、互动、流言、生长全走一遍
            w.pass_time(6)
            evolution.world_pulse(llm, w)
        kinds = {e.kind for e in w.events}
        self.assertIn("world_event", kinds)      # 发现信（样本日历）
        self.assertIn("scene_extended", kinds)   # 旧码头（样本生长）
        self.assertIn("npc_interaction", kinds)  # 老周与阿凛的雨
        # 信是车站的初始物品（样本：场景一等状态）
        self.assertTrue(any(i["name"] == "一封信"
                            for i in w.scenes["s-station"].items))
        self.assertFalse(any(e.kind == "weather_shift" for e in w.events))


class TestEngineIsWorldAgnostic(unittest.TestCase):
    """引擎身份自检：描述决定世界，引擎不认识任何具体世界。"""

    def test_desert_desc_gives_desert_world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界2", DESC2)
        names = [s.name for s in w.scenes.values()]
        self.assertNotIn("旧车站", names)
        self.assertIn("绿洲市集", names)
        npc_names = [n.name for n in w.npcs.values()]
        self.assertNotIn("阿凛", npc_names)
        # 沙漠世界同样完整运转：法则照常触发
        self.assertTrue(w.law_profile.laws)

    def test_neutral_desc_gives_neutral_world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界3", "一座普通的小镇，日子平淡")
        names = [s.name for s in w.scenes.values()]
        self.assertNotIn("旧车站", names)
        self.assertNotIn("绿洲市集", names)

    def test_rain_suite_is_stable_fixture(self):
        # 雨城 = 测试夹具：同一描述永远得到同一世界（回归护栏）
        llm = MockLLM()
        a = generate_world(llm, "世界1", DESC)
        b = generate_world(llm, "世界1", DESC)
        self.assertEqual(a.to_dict(), b.to_dict())


class ScriptedLLM:
    """按脚本回放的假 LLM：测试否决层与软校验。"""

    name = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def chat(self, system, user):
        if not self.responses:
            raise AssertionError("ScriptedLLM 响应耗尽")
        self.calls.append((system, user))
        return self.responses.pop(0)

    def chat_json(self, system, user):
        return json.loads(self.chat(system, user))


class TestRuntimeAgencyMapping(unittest.TestCase):
    def _world(self):
        return generate_world(MockLLM(), "世界1", DESC)

    def test_agency_batch_is_atomic_and_survives_save(self):
        w = self._world()
        a, b = w.npcs["n-arin"], w.npcs["n-zhou"]
        a.state.action = ActionState(text="继续整理咖啡店", due_clock=2.0)
        rejected, changed = w.apply_agency_patches([
            {"body": a.id, "actor": b.id, "until_clock": 1.0,
             "why": "世界法则生效"},
            {"body": "npc:missing", "actor": a.id, "until_clock": 1.0,
             "why": "无效引用"},
        ])
        self.assertTrue(rejected)
        self.assertEqual(changed, set())
        self.assertEqual(w.agency, {})
        self.assertTrue(a.state.action.text)

        errors, changed = w.apply_agency_patches([
            {"body": a.id, "actor": b.id, "until_clock": 1.0,
             "why": "世界法则生效"},
            {"body": b.id, "actor": a.id, "until_clock": 1.0,
             "why": "世界法则生效"},
        ])
        self.assertFalse(any(text.startswith("驳回") for text in errors))
        self.assertEqual(changed, {a.id, b.id})
        self.assertIs(w.actor_for_body(a), b)
        self.assertIs(w.actor_for_body(b), a)
        self.assertFalse(a.state.action.text)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agency.json"
            Universe(worlds={w.name: w}, current=w.name).save(path)
            loaded = Universe.load(path).here
        self.assertEqual(loaded.actor_for_body(a.id).id, b.id)
        self.assertEqual(loaded.actor_for_body(b.id).id, a.id)

    def test_agency_change_gives_actor_private_memory_not_body_owner(self):
        """行动者记得归属变化，身体主人只得到无内容断档。"""
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        actor_before = len(actor.memories)
        body_before = len(body.memories)

        w.apply_agency_patches([{
            "body": body.id, "actor": actor.id, "until_clock": 2.0,
            "why": "测试中的行动主体变化",
        }])
        self.assertTrue(any("开始通过" in memory.content
                            for memory in actor.memories[actor_before:]))
        self.assertEqual(len(body.memories), body_before)
        self.assertEqual(len(body.memory_gaps), 0)

        w.apply_agency_patches([{
            "body": body.id, "actor": body.id, "until_clock": 0.0,
            "why": "测试中的行动主体恢复",
        }])
        self.assertTrue(any("回到自己的身体" in memory.content
                            for memory in actor.memories[actor_before:]))
        self.assertEqual(len(body.memories), body_before)
        self.assertEqual(len(body.memory_gaps), 1)

    def test_pulse_payload_uses_actor_cognition_and_body_identity(self):
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        actor.persona = "只会慢慢说话的守夜人"
        body.persona = "不应进入本次裁决的店长人格"
        actor.goals = [{"id": "actor-goal", "text": "检查末班车",
                        "progress": 0.0}]
        body.goals = [{"id": "body-goal", "text": "冲一杯咖啡",
                       "progress": 0.0}]
        w.remember(actor, "我记得旧站台最后一班车。", cause="测试")
        actor.state.memory_focus = [actor.memories[-1].id]
        w.apply_agency_patches([{
            "body": body.id, "actor": actor.id, "until_clock": 2.0,
            "why": "世界法则生效",
        }])
        payload = evolution.build_pulse_payload(
            w, [body], {body.state.location: 0}, body.state.location)
        row = payload["due_npcs"][0]
        self.assertEqual(row["id"], body.id)
        self.assertEqual(row["name"], body.name)
        self.assertEqual(row["actor"]["id"], actor.id)
        self.assertEqual(row["persona"], actor.persona)
        self.assertEqual([g["id"] for g in row["goals"]], ["actor-goal"])
        self.assertTrue(any("最后一班车" in text for text in row["memories"]))
        self.assertNotIn(body.persona, json.dumps(row, ensure_ascii=False))

    def test_world_pulse_establishes_mapping_before_using_old_plan(self):
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        ensure_scene(MockLLM(), w, body.state.location)
        w.pass_time(6)
        for npc in w.npcs.values():
            npc.state.last_clock = w.clock
        body.state.last_clock = 0.0
        old_activity = body.state.activity
        pulse = json.dumps({
            "events": [], "entity_events": [], "state_fact_patches": [],
            "memory_access_patches": [],
            "agency_patches": [{
                "body": body.id, "actor": actor.id,
                "until_clock": w.clock + 1.0, "why": "世界法则生效",
            }],
            "npc_plans": [{
                "npc": body.id,
                "state": {"activity": "沿用身体主人的旧计划",
                          "mood": "平静", "location": body.state.location},
                "action": None, "interaction": None,
                "goal_updates": {}, "new_goals": [],
            }],
            "item_patches": [], "new_npcs": [], "crowds": [],
            "daily_bits": [], "new_scenes": [], "fact_changes": [],
        }, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([pulse]), w)
        self.assertIs(w.actor_for_body(body), actor)
        self.assertEqual(body.state.activity, old_activity)
        self.assertTrue(any("重新适应" in text for text in summaries))

    def test_repeating_moment_commits_agency_on_world_clock(self):
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        ensure_scene(MockLLM(), w, body.state.location)
        w.moments = [{
            "due_day": 1,
            "repeat_days": 2.0,
            "what": "两人的行动归属在醒来时发生变化",
            "done": False,
            "agency_patches": [{
                "body": body.id, "actor": actor.id,
                "duration_days": 1.0, "why": "周期性世界承诺",
            }, {
                "body": actor.id, "actor": body.id,
                "duration_days": 1.0, "why": "周期性世界承诺",
            }],
        }]
        w.pass_time(6)
        pulse = json.dumps({
            "events": [], "entity_events": [], "state_fact_patches": [],
            "memory_access_patches": [], "agency_patches": [],
            "npc_plans": [], "item_patches": [], "new_npcs": [],
            "crowds": [], "daily_bits": [], "new_scenes": [],
            "fact_changes": [],
        }, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([pulse]), w)
        self.assertEqual(w.actor_for_body(body), actor)
        self.assertEqual(w.actor_for_body(actor), body)
        self.assertFalse(w.moments[0]["done"])
        self.assertTrue(any("agency" in event.kind
                            for event in w.events))
        self.assertTrue(any(event.cause == "周期性世界承诺"
                            for event in w.events))

    def test_crossed_scheduled_moments_commit_in_clock_order(self):
        """一次等待跨过两个时刻时，开始与恢复仍分别原子落账。"""
        w = self._world()
        w.heartbeat = 0.25
        a, b = w.npcs["n-arin"], w.npcs["n-zhou"]
        w.moments = [{
            "due_day": 1, "what": "两人的行动归属发生变化",
            "agency_patches": [{
                "body": a.id, "actor": b.id, "duration_days": 1.0,
                "why": "第一次行动归属变化",
            }, {
                "body": b.id, "actor": a.id, "duration_days": 1.0,
                "why": "第一次行动归属变化",
            }],
        }, {
            "due_day": 2, "what": "两人的行动归属恢复",
            "agency_patches": [{
                "body": a.id, "actor": a.id, "duration_days": 1.0,
                "why": "行动归属恢复",
            }, {
                "body": b.id, "actor": b.id, "duration_days": 1.0,
                "why": "行动归属恢复",
            }],
        }]
        w.pass_time(6)  # clock = 1.5，跨过第 1 天和第 2 天的时刻
        pulse = json.dumps({
            "events": [], "entity_events": [], "state_fact_patches": [],
            "memory_access_patches": [], "agency_patches": [],
            "npc_plans": [], "item_patches": [], "new_npcs": [],
            "crowds": [], "daily_bits": [], "new_scenes": [],
            "fact_changes": [],
        }, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([pulse]), w)
        self.assertEqual(w.agency, {})
        self.assertTrue(all(moment["done"] for moment in w.moments))
        self.assertGreaterEqual(
            len([e for e in w.events if e.kind == "agency_changed"]), 4)
        self.assertGreaterEqual(
            len([e for e in w.events if e.kind == "memory_gap_recorded"]), 2)
        self.assertFalse(any(text.startswith("驳回行动主体映射：")
                             for text in summaries))
        restore_event = next(e for e in w.events
                             if e.kind == "agency_changed"
                             and "恢复为身体主人" in e.summary)
        restore_world_event = next(e for e in w.events
                                   if e.kind == "world_event"
                                   and "行动归属恢复" in e.summary)
        self.assertEqual(restore_event.day, restore_world_event.day)
        self.assertGreater(restore_world_event.turn, restore_event.turn)

    def test_due_moment_commits_before_normal_pulse_interval(self):
        """确定性归位不应等待普通 NPC 脉冲。"""
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        w.heartbeat = 0.25
        w.clock = 0.25
        w.pulse_last_clock = w.clock
        w.apply_agency_patches([{
            "body": body.id, "actor": actor.id,
            "until_clock": 1.25, "why": "测试借身",
        }])
        w.moments = [{
            "due_day": 2, "what": "归位时刻到达",
            "agency_patches": [{
                "body": body.id, "actor": body.id,
                "duration_days": 0, "why": "归位时刻到达",
            }],
        }]
        w.pass_time(4)
        summaries = evolution.world_pulse(ScriptedLLM([]), w)
        self.assertEqual(w.agency, {})
        event = next(e for e in w.events
                      if e.kind == "world_event" and "归位时刻到达" in e.summary)
        self.assertAlmostEqual(event.day, 1.25, places=6)
        self.assertTrue(any("归位时刻到达" in text for text in summaries))

    def test_named_scheduled_moment_is_structured_and_marks_done(self):
        """既定时刻的具名主体通过 refs 合法落账，并同步完成标记。"""
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        w.moments = [{
            "due_day": 1,
            "what": f"{actor.name}通过{body.name}的身体行动",
            "agency_patches": [{
                "body": body.id, "actor": actor.id, "duration_days": 1.0,
                "why": f"{actor.name}通过{body.name}的身体行动",
            }],
        }]
        w.pass_time(6)
        pulse = json.dumps({
            "events": [], "entity_events": [], "state_fact_patches": [],
            "memory_access_patches": [], "agency_patches": [],
            "npc_plans": [], "item_patches": [], "new_npcs": [],
            "crowds": [], "daily_bits": [], "new_scenes": [],
            "fact_changes": [],
        }, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([pulse]), w)
        self.assertTrue(w.moments[0]["done"])
        self.assertIs(w.actor_for_body(body), actor)
        self.assertTrue(any(e.kind == "world_event"
                            and actor.name in e.summary
                            for e in w.events))
        self.assertFalse(any(text.startswith("驳回既定时刻：")
                             for text in summaries))

    def test_named_moment_without_agency_commits_declared_refs(self):
        """具名既定时刻不依赖行动归属映射，也能以声明引用落账。"""
        w = self._world()
        a, b = w.npcs["n-arin"], w.npcs["n-zhou"]
        w.moments = [{
            "due_day": 1,
            "what": f"黄昏时{a.name}与{b.name}在旧站台相见",
            "location": a.state.location,
            "refs": [f"npc:{a.id}", f"npc:{b.id}",
                     f"scene:{a.state.location}"],
        }]
        w.pass_time(6)
        pulse = json.dumps({
            "events": [], "entity_events": [], "state_fact_patches": [],
            "memory_access_patches": [], "agency_patches": [],
            "npc_plans": [], "item_patches": [], "new_npcs": [],
            "crowds": [], "daily_bits": [], "new_scenes": [],
            "fact_changes": [],
        }, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([pulse]), w)
        event = next(e for e in w.events if e.kind == "world_event")
        params = event.payload["event_params"]
        self.assertTrue(w.moments[0]["done"])
        self.assertEqual(params["refs"], sorted(w.moments[0]["refs"]))
        self.assertFalse(any(text.startswith("驳回既定时刻：")
                             for text in summaries))

    def test_wait_across_periods_commits_every_missed_occurrence(self):
        """跨过多个周期后，每个世界钟边界都留下独立承诺事件。"""
        w = self._world()
        w.heartbeat = 0.25
        w.moments = [{
            "due_day": 1,
            "repeat_days": 1.0,
            "what": "钟楼在清晨报时",
            "refs": [f"scene:{w.player['location']}"],
        }]
        w.pass_time(12)  # clock = 3.0，跨过 0、1、2、3 四个周期边界
        pulse = json.dumps({
            "events": [], "entity_events": [], "state_fact_patches": [],
            "memory_access_patches": [], "agency_patches": [],
            "npc_plans": [], "item_patches": [], "new_npcs": [],
            "crowds": [], "daily_bits": [], "new_scenes": [],
            "fact_changes": [],
        }, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([pulse]), w)
        events = [e for e in w.events
                  if e.kind == "world_event" and "钟楼在清晨报时" in e.summary]
        self.assertEqual([event.day for event in events], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(w.moments[0]["last_occurrence"], 3)
        self.assertAlmostEqual(w.clock, 3.0)

    def test_agency_does_not_move_body_from_actor_location(self):
        """行动主体变化只换控制权，状态快照不能让身体瞬移。"""
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        original_body_location = body.state.location
        self.assertNotEqual(original_body_location, actor.state.location)
        w.apply_agency_patches([{
            "body": body.id, "actor": actor.id,
            "until_clock": w.clock + 2.0, "why": "测试中的行动主体变化",
        }])
        ensure_scene(MockLLM(), w, original_body_location)
        w.player["location"] = original_body_location
        w.pass_time(6)
        body.state.last_clock = w.clock - 2.0
        plan = json.dumps({
            "events": [], "entity_events": [], "state_fact_patches": [],
            "memory_access_patches": [], "agency_patches": [],
            "npc_plans": [{
                "npc": body.id,
                "state": {"activity": "想起行动者的原住处",
                          "mood": "平静", "location": actor.state.location},
                "action": None, "interaction": None,
                "goal_updates": {}, "new_goals": [],
            }],
            "item_patches": [], "new_npcs": [], "crowds": [],
            "daily_bits": [], "new_scenes": [], "fact_changes": [],
        }, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertEqual(body.state.location, original_body_location)
        self.assertIn(body.id, w.scenes[original_body_location].npcs)
        self.assertNotIn(body.id, w.scenes[actor.state.location].npcs)
        self.assertEqual(body.state.activity, "想起行动者的原住处")
        self.assertTrue(any("直接改变身体位置" in text for text in summaries))

    def test_scheduled_agency_precedes_event_memory_projection(self):
        """周期映射先成立，随后产生的事件不能按身体主人写记忆。"""
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        w.moments = [{
            "due_day": 1, "repeat_days": 2.0,
            "what": "两人的行动归属在醒来时发生变化",
            "agency_patches": [{
                "body": body.id, "actor": actor.id,
                "duration_days": 1.0, "why": "周期性世界承诺",
            }],
        }]
        w.pass_time(6)
        pulse = json.dumps({
            "events": [{"type": "world_event", "params": {
                "title": "身体里的陌生脚步", "detail": "身体里的陌生脚步",
                "location": body.state.location, "intensity": 0.1,
            }}],
            "entity_events": [], "state_fact_patches": [],
            "memory_access_patches": [], "agency_patches": [],
            "npc_plans": [], "item_patches": [], "new_npcs": [],
            "crowds": [], "daily_bits": [], "new_scenes": [],
            "fact_changes": [],
        }, ensure_ascii=False)
        scripted = ScriptedLLM([pulse])
        evolution.world_pulse(scripted, w)
        payload = json.loads(scripted.calls[0][1])
        self.assertEqual(payload["world"]["agency"][0]["actor"], actor.id)
        event_memories = [e for e in w.events
                          if e.kind == "npc_memory"
                          and "身体里的陌生脚步" in e.summary]
        self.assertTrue(event_memories)
        self.assertTrue(all(e.payload.get("actor") == actor.id
                            for e in event_memories
                            if e.payload.get("body") == body.id))

    def test_action_started_in_borrowed_body_keeps_actor_after_expiry(self):
        """归位后物理动作可完成，但经历仍归动作启动者。"""
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        w.apply_agency_patches([{
            "body": body.id, "actor": actor.id,
            "until_clock": 0.2, "why": "测试借身",
        }])
        evolution._begin_action(
            w, body, {"action": "在原地整理借来的账本",
                      "location": body.state.location, "days": 0.25},
            w.turn)
        self.assertEqual(body.state.action.actor_id, actor.id)
        w.pass_time(6)  # 到 0.25 天：映射先到期，动作随后结算
        self.assertNotIn(body.id, w.agency)
        resolve = json.dumps({"outcome": "账本被整理完毕。", "patch": None},
                             ensure_ascii=False)
        evolution.advance_action(
            ScriptedLLM([resolve]), w, body, turns=6)
        done = next(e for e in reversed(w.events) if e.kind == "action_done")
        done_params = done.payload["event_params"]
        self.assertEqual(done_params["body"], body.id)
        self.assertEqual(done_params["actor"], actor.id)
        self.assertIn("行动者：", event_identity_note(w, done.kind,
                                                   done.payload))
        self.assertTrue(any("账本被整理完毕" in memory.content
                            for memory in actor.memories))
        self.assertFalse(any("账本被整理完毕" in memory.content
                             for memory in body.memories))
        self.assertTrue(body.memory_gaps)

    def test_agency_start_records_interrupted_old_action(self):
        """新行动主体覆盖旧计划时，旧动作必须留下中止痕迹。"""
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        evolution._begin_action(
            w, body, {"action": "去杂货铺买东西",
                      "location": body.state.location, "days": 1.0},
            w.turn)
        summaries, changed = w.apply_agency_patches([{
            "body": body.id, "actor": actor.id,
            "until_clock": w.clock + 1.0, "why": "第一次借身开始",
        }])
        self.assertFalse(any(text.startswith("驳回行动主体映射：")
                             for text in summaries))
        self.assertIn(body.id, changed)
        self.assertFalse(body.state.action.text)
        aborted = next(e for e in reversed(w.events)
                        if e.kind == "action_aborted")
        self.assertEqual(aborted.payload["event_params"]["actor"], body.id)
        self.assertTrue(any("搁下了这件事" in memory.content
                            for memory in body.memories))

    def test_borrowed_body_remote_action_requires_explicit_travel(self):
        """借身时不能把行动者原地点误当成身体当前位置。"""
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        w.apply_agency_patches([{
            "body": body.id, "actor": actor.id,
            "until_clock": w.clock + 2.0, "why": "测试借身",
        }])
        actor.goals.append({"id": "g-remote", "text": "查看远处的物品",
                            "progress": 0.0, "targets": []})
        body.state.last_clock = w.clock - 1.0
        plan = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": body.id, "action": "查看行动者原住处的物品",
                "location": actor.state.location, "days": 0.1,
            }}],
            "intent": None, "goal_updates": {}, "new_goals": [],
        }, ensure_ascii=False)
        summaries = evolution.propose_proactive(
            ScriptedLLM([plan]), w, body)
        self.assertFalse(any(e.kind == "npc_acted" for e in w.events))
        self.assertTrue(any("travel=true" in text for text in summaries))

    def test_player_story_labels_borrowed_action_as_body(self):
        """玩家只看到身体的行为，不被内部 actor 身份强行全知。"""
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        w.player["location"] = body.state.location
        w.player["start"] = body.state.location
        w.apply_agency_patches([{
            "body": body.id, "actor": actor.id, "until_clock": 2.0,
            "why": "测试借身行动",
        }])
        errors = emit(w, "npc_acted", {
            "npc": body.id, "action": "在门口停下",
            "location": body.state.location, "days": 0.0,
        }, "测试借身行动")
        self.assertEqual(errors, [])
        event = w.events[-1]
        params = event.payload["event_params"]
        self.assertEqual(params["body"], body.id)
        self.assertEqual(params["actor"], actor.id)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "world.json"
            out = Path(tmp) / "story.txt"
            Universe(worlds={w.name: w}, current=w.name).save(src)
            finalstory.build(str(src), str(out))
            story = out.read_text(encoding="utf-8")
        self.assertIn(f"{body.name}的身体 主动：在门口停下", story)
        self.assertNotIn(actor.name + "通过", story)

    def test_dialogue_shows_body_but_writes_experience_to_actor(self):
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        w.player["location"] = body.state.location
        w.apply_agency_patches([{
            "body": body.id, "actor": actor.id, "until_clock": 2.0,
            "why": "世界法则生效",
        }])
        actor_count = len(actor.memories)
        body_count = len(body.memories)
        response = json.dumps({
            "reply": "这地方我完全不熟。", "reaction": "慌张地看向吧台。",
            "action": None, "relationship_delta": 0, "mood_delta": -0.1,
            "memory": "我通过阿凛的身体醒在陌生咖啡店。",
            "memory_importance": 0.8, "memory_keywords": ["咖啡店"],
            "memory_refs": [], "law_ids": [], "choices": [],
            "item_patches": [],
        }, ensure_ascii=False)
        llm = ScriptedLLM([response])
        result = interpreter.dialogue_turn(llm, w, body, "你今天怎么怪怪的？")
        payload = json.loads(llm.calls[0][1])["npc"]
        self.assertEqual(payload["id"], body.id)
        self.assertEqual(payload["actor"]["id"], actor.id)
        self.assertEqual(payload["persona"], actor.persona)
        self.assertEqual(result.reply, "这地方我完全不熟。")
        self.assertEqual(len(actor.memories), actor_count + 1)
        self.assertEqual(actor.memories[-1].embodied_as, body.id)
        self.assertEqual(len(body.memories), body_count)
        dialogue = next(e for e in reversed(w.events)
                        if e.kind == "dialogue" and "reply" in
                        (e.payload or {}).get("event_params", {}))
        self.assertIn(body.name, dialogue.summary)
        self.assertNotIn(actor.name, dialogue.summary)

    def test_expiry_creates_one_complete_gap(self):
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        w.apply_agency_patches([{
            "body": body.id, "actor": actor.id, "until_clock": 1.0,
            "why": "世界法则生效",
        }])
        w.remember_as(body, "我先检查了陌生房间。", cause="映射期间")
        w.pass_time(12, cause="半天过去")
        w.remember_as(body, "我又在街上遇见了玩家。", cause="映射期间")
        w.pass_time(12, cause="又半天过去")
        self.assertNotIn(body.id, w.agency)
        self.assertEqual(len(body.memory_gaps), 1)
        gap = body.memory_gaps[0]
        self.assertEqual(gap["started_clock"], 0.0)
        self.assertEqual(gap["ended_clock"], 1.0)
        self.assertTrue(any("陌生房间" in m.content for m in actor.memories))
        self.assertTrue(any("遇见了玩家" in m.content for m in actor.memories))
        self.assertFalse(any("陌生房间" in m.content for m in body.memories))

    def test_observer_only_records_what_the_body_did(self):
        w = self._world()
        actor, body = w.npcs["n-zhou"], w.npcs["n-arin"]
        witness = w.npcs["n-man"]
        evolution.move_npc(w, witness, body.state.location, cause="测试同场")
        w.player["location"] = "s-station"
        w.apply_agency_patches([{
            "body": body.id, "actor": actor.id, "until_clock": 2.0,
            "why": "世界法则生效",
        }])
        plan = json.dumps({
            "activity": "认错了咖啡机", "mood": "慌张",
            "location": body.state.location,
            "interaction": {"with": witness.id,
                            "line": "这个东西到底怎么用？",
                            "link_delta": 0.0},
        }, ensure_ascii=False)
        evolution.heartbeat(ScriptedLLM([plan]), w, body)
        event = next(e for e in reversed(w.events)
                     if e.kind == "npc_interaction")
        params = event.payload["event_params"]
        self.assertEqual(params["npc"], body.id)
        self.assertNotIn("actor", params)
        self.assertTrue(any(body.name in m.content for m in witness.memories))
        self.assertFalse(any(actor.name in m.content for m in witness.memories))
        self.assertTrue(any("这个东西到底怎么用" in m.content
                            for m in actor.memories))


class TestDialogueReactions(unittest.TestCase):
    def _world(self):
        w = generate_world(MockLLM(), "世界1", DESC)
        ensure_scene(MockLLM(), w, "s-cafe")
        w.player["location"] = "s-cafe"
        return w, w.npcs["n-arin"]

    @staticmethod
    def _response(**overrides):
        data = {
            "reply": "", "reaction": "", "action": None,
            "choices": [], "law_ids": [], "relationship_delta": 0,
            "mood_delta": 0.0, "memory_importance": 0.4,
            "memory": "", "memory_keywords": [], "item_patches": [],
        }
        data.update(overrides)
        return json.dumps(data, ensure_ascii=False)

    def test_npc_can_react_silently_without_a_fake_reply(self):
        w, arin = self._world()
        # 本用例只测对话裁决，不额外启动目标裁决。
        w.social[f"{arin.id}->proactive"] = w.turn
        result = interpreter.dialogue_turn(ScriptedLLM([
            self._response(reaction="她沉默了一会儿，把目光移向窗外。")
        ]), w, arin, "你知道那封信吗？")

        self.assertEqual(result.reply, "")
        self.assertIn("沉默", result.reaction)
        reacted = next(e for e in w.events if e.kind == "npc_reacted")
        self.assertIn("窗外", reacted.summary)
        self.assertFalse(any(
            e.kind == "dialogue"
            and (e.payload or {}).get("event_params", {}).get("reply")
            for e in w.events))

    def test_npc_can_answer_only_with_a_real_sustained_action(self):
        w, arin = self._world()
        result = interpreter.dialogue_turn(ScriptedLLM([
            self._response(action={
                "text": "披上外套，去旧车站查看那封信",
                "location": "s-station", "days": 0.5,
                "targets": ["item:i-letter"],
                "requires": ["item:i-letter"],
            })
        ]), w, arin, "你不去看看那封信吗？")

        self.assertEqual(result.reply, "")
        self.assertEqual(result.reaction, "")
        self.assertIn("去旧车站", result.action)
        self.assertEqual(arin.state.action.location, "s-station")
        self.assertGreater(arin.state.action.due_clock, w.clock)
        self.assertTrue(any(e.kind == "npc_acted" for e in w.events))

    def test_missing_response_becomes_observable_nonresponse_not_speech(self):
        w, arin = self._world()
        w.social[f"{arin.id}->proactive"] = w.turn
        result = interpreter.dialogue_turn(
            ScriptedLLM([self._response()]), w, arin, "你在听吗？")
        self.assertEqual(result.reply, "")
        self.assertIn("没有作出", result.reaction)
        self.assertEqual(w.events[-1].kind, "npc_reacted")


class TestGrounding(unittest.TestCase):
    """一致性加固：结构化引用硬否决 + 自由文本软校验。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-cafe"
        return llm, w

    def test_world_pulse_vetoes_unknown_npc_plan(self):
        llm, w = self._world()
        w.pass_time(6)
        bad = json.dumps({"events": [],
                          "npc_plans": [{"npc": "n-ghost",
                                         "state": None}],
                          "influences": [], "new_scenes": []},
                         ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([bad]), w)
        # 不存在的 NPC 被否决：世界无变化，无崩溃
        self.assertFalse(any(e.kind == "npc_acted" for e in w.events))

    def test_npc_action_cannot_smuggle_dialogue_with_existing_npc(self):
        """单人行动不能绕开 interaction，把一段对话只写进说者记忆。"""
        llm, w = self._world()
        arin, zhou = w.npcs["n-arin"], w.npcs["n-zhou"]
        evolution.move_npc(w, zhou, arin.state.location, cause="测试同场")
        errors = emit(w, "npc_acted", {
            "npc": arin.id,
            "action": "阿凛看见老周后说：『我找到那封信了。』",
            "location": arin.state.location,
        }, cause="测试")
        self.assertTrue(any("npc_interaction" in error for error in errors))
        self.assertFalse(any(e.kind == "npc_acted" for e in w.events))
        goal = arin.goals[0]
        old_progress = float(goal["progress"])
        plan = json.dumps({
            "events": [{"type": "npc_acted", "params": {
                "npc": arin.id,
                "action": "阿凛对老周说：『我找到那封信了。』",
                "location": arin.state.location}}],
            "goal_updates": {goal["id"]: {
                "progress": old_progress + 0.2, "because": "刚和老周说过"}},
            "new_goals": []}, ensure_ascii=False)
        evolution.propose_proactive(ScriptedLLM([plan]), w, arin)
        self.assertEqual(float(goal["progress"]), old_progress)

    def test_cross_scene_action_requires_structured_destination(self):
        """已经走向另一处场景却不填 location，不能只在叙事里完成移动。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        errors = emit(w, "npc_acted", {
            "npc": arin.id,
            "action": "阿凛快步走向旧车站，想查清那封信的来历。",
        }, cause="测试")
        self.assertTrue(any("必须给 location" in error for error in errors))
        self.assertFalse(any(e.kind == "npc_acted" for e in w.events))

    def test_world_pulse_vetoes_oversized_extension(self):
        llm, w = self._world()
        w.pass_time(80)
        bad = json.dumps({
            "events": [], "npc_plans": [], "influences": [],
            "new_scenes": [{"from": "s-cafe",
                            "name": "x" * 30,
                            "hint": "y" * 100}]}, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([bad]), w)
        self.assertTrue(any("驳回脑补" in s for s in summaries))
        self.assertFalse(any(e.kind == "scene_extended" for e in w.events))

    def test_dialogue_grounding_warning(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        # 阿凛在咖啡店，切片里没有老周——回复提及老周 → 软提示
        catchup = json.dumps({"location": "s-cafe", "activity": "煮咖啡",
                              "mood": "平静", "moved": False, "memory": ""},
                             ensure_ascii=False)
        fake = json.dumps({
            "reply": "老周告诉我，车站出事了。",
            "choices": [], "law_ids": [],
            "relationship_delta": 0, "mood_delta": 0.0,
            "memory_importance": 0.5,
            "memory": "老周说过车站出事了。"}, ensure_ascii=False)
        goal_empty = json.dumps({"events": [], "goal_updates": {},
                                 "new_goals": []}, ensure_ascii=False)
        # 第一次调用是读取补算，第二次才是对话裁决，第三次是目标裁决
        w.pass_time(6)
        r = interpreter.dialogue_turn(
            ScriptedLLM([catchup, fake, goal_empty]), w, arin, "怎么了？")
        self.assertIn("老周", r.reply)  # 不拦截（叙事自由）
        self.assertTrue(any("老周" in x for x in r.grounding_warnings))


class TestEventDerivedKnowledge(unittest.TestCase):
    """知识只能从已记录的事件派生，不能由世界脉冲直接注入。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-station"
        return llm, w

    def _pulse_with(self, w, **plan_data):
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [],
                           "new_scenes": [],
                           "daily_bits": [], "item_patches": [],
                           "new_npcs": [], "crowds": []},
                          ensure_ascii=False)
        data = json.loads(plan)
        data.update(plan_data)
        plan = json.dumps(data, ensure_ascii=False)
        return evolution.world_pulse(ScriptedLLM([plan]), w)

    def test_unsupported_influence_cannot_write_memory(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        before = len(arin.memories)
        self._pulse_with(w, influences=[{
            "target": "npc:n-arin", "change": "她听说钟楼敲了十三下",
            "why": "午夜钟声"}])
        self.assertEqual(len(arin.memories), before)

    def test_world_event_creates_event_derived_memories(self):
        llm, w = self._world()
        self._pulse_with(w, events=[{
            "type": "world_event",
            "params": {"title": "钟楼异响", "detail": "午夜多响了一下",
                       "location": "s-cafe", "intensity": 0.3}}])
        memories = [e for e in w.events if e.kind == "npc_memory"]
        self.assertTrue(memories)
        self.assertTrue(all(e.cause.startswith("事件：") for e in memories))
        self.assertTrue(any("钟楼异响" in e.summary for e in memories))


class TestSceneMemory(unittest.TestCase):
    """场景级记忆：变化记忆（有界窗口）+ 关联记忆（强度索引）。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-station"
        return llm, w

    def test_scene_recent_records_and_caps(self):
        llm, w = self._world()
        for i in range(12):
            emit(w, "world_event",
                 {"title": f"事{i}", "detail": f"第{i}件事",
                  "location": "s-station", "intensity": 0.5},
                 cause="测试")
        recent = w.scenes["s-station"].recent
        self.assertEqual(len(recent), 8)  # 有界窗口
        self.assertEqual(recent[-1]["summary"], "事11：第11件事")
        self.assertEqual(recent[0]["summary"], "事4：第4件事")

    def test_associations_built_and_capped(self):
        llm, w = self._world()
        for _ in range(12):
            evolution.move_npc(w, w.npcs["n-arin"], "s-station", cause="测试")
            evolution.move_npc(w, w.npcs["n-arin"], "s-cafe", cause="测试")
        key = "|".join(sorted(("s-station", "s-cafe")))
        self.assertEqual(w.associations[key], 9.0)  # 强度封顶
        pairs = scene_associations(w, "s-station")
        self.assertEqual(pairs[0][0].id, "s-cafe")

    def test_scene_memory_survives_save_load(self):
        llm, w = self._world()
        evolution.move_npc(w, w.npcs["n-arin"], "s-cafe", cause="测试")
        import tempfile
        from pathlib import Path
        u = Universe(worlds={w.name: w}, current=w.name)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "u.json"
            u.save(p)
            u2 = Universe.load(p)
        w2 = u2.here
        self.assertEqual(w2.scenes["s-cafe"].recent,
                         w.scenes["s-cafe"].recent)
        self.assertEqual(w2.associations, w.associations)

    def test_dialogue_payload_carries_scene_memory(self):
        llm, w = self._world()
        evolution.move_npc(w, w.npcs["n-arin"], "s-station", cause="测试")
        emit(w, "world_event",
             {"title": "怪事", "detail": "钟声十三下",
              "location": "s-station", "intensity": 0.5},
             cause="测试")
        reply = json.dumps({
            "reply": "哦。", "choices": [], "law_ids": [],
            "relationship_delta": 0, "mood_delta": 0.0,
            "memory_importance": 0.5, "memory": "他问了怪事。"},
            ensure_ascii=False)
        goal = json.dumps({"events": [], "goal_updates": {},
                           "new_goals": []}, ensure_ascii=False)
        recorder = ScriptedLLM([reply, reply, reply, goal])
        w.npcs["n-arin"].state.last_turn = w.turn  # 跳过读取补算的 LLM 调用
        interpreter.dialogue_turn(recorder, w, w.npcs["n-arin"], "怪事？")
        payload = json.loads(recorder.calls[0][1])
        self.assertIn("scene_recent", payload["world"])
        self.assertIn("associations", payload["world"])
        self.assertTrue(any("怪事" in s
                            for s in payload["world"]["scene_recent"]))
        self.assertTrue(any(a["strength"] >= 1
                            for a in payload["world"]["associations"]))


class TestSceneItems(unittest.TestCase):
    """场景物品层：物品 = 场景一等状态（覆写 + 跃变日志，三律）。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-station"
        return llm, w

    def test_initial_items_are_scene_state(self):
        llm, w = self._world()
        station = w.scenes["s-station"]
        self.assertTrue(any(i["name"] == "一封信" for i in station.items))

    def test_add_patch_logs_appearance(self):
        llm, w = self._world()
        errors = apply_item_patch(
            w, {"op": "add", "item": "i-cup", "name": "打翻的茶杯",
                "location": "s-cafe", "note": "茶还冒着热气"},
            cause="测试")
        self.assertEqual(errors, [])
        cafe = w.scenes["s-cafe"]
        self.assertTrue(any(i["id"] == "i-cup" for i in cafe.items))
        kinds = [e.kind for e in w.events]
        self.assertIn("item_added", kinds)

    def test_item_ids_are_unique_across_the_world(self):
        """物品引用是世界级语法，不能只在单个场景里唯一。"""
        llm, w = self._world()
        errors = apply_item_patch(
            w, {"op": "add", "item": "i-letter", "name": "另一封信",
                "location": "s-cafe", "note": "不应落库"},
            cause="测试")
        self.assertTrue(errors)
        self.assertIn("全局唯一", errors[0])
        self.assertFalse(any(i.get("name") == "另一封信"
                             for i in w.scenes["s-cafe"].items))

    def test_fold_uses_item_identity_for_legacy_duplicate_ids(self):
        """旧档若已有重复 id，折叠收尾仍必须落在物品真实所在场景。"""
        llm, w = self._world()
        cafe = w.scenes["s-cafe"]
        duplicate = {"id": "i-letter", "name": "咖啡店账本",
                     "note": "纸页卷边", "last_turn": w.turn,
                     "fold": {"start": w.clock, "count": 2,
                              "last": "纸页卷边"}}
        cafe.items.append(duplicate)
        transfer_item(w, duplicate, "", "测试旧档")
        folded = [e for e in w.events if e.kind == "daily_life"
                  and "咖啡店账本" in e.summary]
        self.assertTrue(folded)
        self.assertEqual(folded[-1].payload["event_params"]["location"],
                         "s-cafe")

    def test_remove_patch_requires_existing_item(self):
        llm, w = self._world()
        errors = apply_item_patch(
            w, {"op": "remove", "item": "i-ghost", "location": "s-cafe"},
            cause="测试")
        self.assertTrue(errors)  # 不凭空删改
        self.assertFalse(any(e.kind == "item_removed" for e in w.events))

    def test_change_patch_overwrites_and_logs_transition(self):
        llm, w = self._world()
        apply_item_patch(
            w, {"op": "change", "item": "i-letter", "location": "s-station",
                "name": "被拆开过的信", "note": "信封口有折痕"},
            cause="测试")
        station = w.scenes["s-station"]
        letter = next(i for i in station.items if i["id"] == "i-letter")
        self.assertEqual(letter["name"], "被拆开过的信")
        kinds = [e.kind for e in w.events]
        self.assertIn("item_changed", kinds)
        # 跃变必须有内容：空覆写不许（防摇摆）
        errors = apply_item_patch(
            w, {"op": "change", "item": "i-letter",
                "location": "s-station"}, cause="测试")
        self.assertTrue(errors)

    def test_remove_logs_what_was_lost(self):
        llm, w = self._world()
        apply_item_patch(
            w, {"op": "remove", "item": "i-letter", "location": "s-station",
                "note": "被风吹进了雨里"},
            cause="测试")
        self.assertFalse(any(i["id"] == "i-letter"
                             for i in w.scenes["s-station"].items))
        ev = next(e for e in w.events if e.kind == "item_removed")
        self.assertIn("一封信", ev.summary)  # 日志保留「有过被吹走的信」

    def test_items_survive_save_load(self):
        llm, w = self._world()
        apply_item_patch(
            w, {"op": "change", "item": "i-letter", "location": "s-station",
                "note": "被拆开过"}, cause="测试")
        u = Universe(worlds={w.name: w}, current=w.name)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "u.json"
            u.save(p)
            u2 = Universe.load(p)
        station2 = u2.here.scenes["s-station"]
        self.assertEqual([i["note"] for i in station2.items
                          if i["id"] == "i-letter"], ["被拆开过"])

    def test_transform_spawns_byproducts(self):
        """转化：烟 → 烟蒂 + 烟灰（同一跃变，副产品入账）。"""
        llm, w = self._world()
        apply_item_patch(
            w, {"op": "add", "item": "i-cig", "name": "一支烟",
                "location": "s-station", "note": "刚点上"},
            cause="老陈")
        errors = apply_item_patch(
            w, {"op": "change", "item": "i-cig", "location": "s-station",
                "name": "烟蒂", "note": "被抽完了",
                "spawn": [{"id": "i-ash", "name": "烟灰",
                           "note": "落在站台边"}]},
            cause="老陈抽烟")
        self.assertEqual(errors, [])
        station = w.scenes["s-station"]
        by_id = {i["id"]: i for i in station.items}
        self.assertEqual(by_id["i-cig"]["name"], "烟蒂")  # 烟 → 烟蒂
        self.assertIn("i-ash", by_id)                      # 烟灰入账
        kinds = [e.kind for e in w.events]
        self.assertIn("item_changed", kinds)
        self.assertIn("item_added", kinds)

    def test_transform_spawn_three_laws(self):
        llm, w = self._world()
        # 产物 id 重复 → 拒；缺名称 → 拒
        errors = apply_item_patch(
            w, {"op": "change", "item": "i-letter", "location": "s-station",
                "spawn": [{"id": "i-letter", "name": "重复"}]},
            cause="测试")
        self.assertTrue(errors)
        self.assertEqual(len(w.scenes["s-station"].items), 1)

    def test_items_swept_after_long_idle(self):
        """自动消逝：无人提及 96 回合后物品被时间扫走，且日志可回放。"""
        llm, w = self._world()
        station = w.scenes["s-station"]
        apply_item_patch(
            w, {"op": "add", "item": "i-ash", "name": "烟灰",
                "location": "s-station", "note": "没人管的烟灰"},
            cause="老陈")
        ash = next(i for i in station.items if i["id"] == "i-ash")
        ash["last_turn"] = 1            # 伪造：从第 1 回合起没人提起
        w.turn = 1 + ITEM_MAX_IDLE + 1  # 超过 96 回合
        swept = w.sweep_items()
        self.assertTrue(any("烟灰" in s for s in swept))
        self.assertFalse(any(i["id"] == "i-ash" for i in station.items))
        ev = [e for e in w.events
              if e.kind == "item_removed" and "烟灰" in e.summary]
        self.assertTrue(ev)
        self.assertEqual(ev[-1].cause, "时间流逝")

    def test_recent_items_survive_sweep(self):
        llm, w = self._world()
        station = w.scenes["s-station"]
        w.turn = 200
        w.scenes["s-station"].items[0]["last_turn"] = 199  # 刚被提及
        swept = w.sweep_items()
        self.assertFalse(swept)
        self.assertTrue(any(i["id"] == "i-letter" for i in station.items))

    def test_touch_items_refreshes_freshness(self):
        llm, w = self._world()
        station = w.scenes["s-station"]
        letter = station.items[0]
        letter["last_turn"] = 1
        touch_items(station, 100)
        self.assertEqual(letter["last_turn"], 100)
        w.turn = 101
        self.assertFalse(w.sweep_items())  # 刚被保鲜，不会被扫走

    def test_last_turn_survives_save_load(self):
        llm, w = self._world()
        touch_items(w.scenes["s-station"], 42)
        u = Universe(worlds={w.name: w}, current=w.name)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "u.json"
            u.save(p)
            u2 = Universe.load(p)
        self.assertEqual(u2.here.scenes["s-station"].items[0]["last_turn"], 42)

    def test_player_takes_item_in_dialogue(self):
        llm, w = self._world()
        evolution.move_npc(w, w.npcs["n-arin"], "s-station", cause="测试")
        w.player["location"] = "s-station"
        arin = w.npcs["n-arin"]
        reply = json.dumps({
            "reply": "拿去吧。", "choices": [], "law_ids": [],
            "relationship_delta": 0, "mood_delta": 0.0,
            "memory_importance": 0.5, "memory": "他把信拿走了。",
            "item_patches": [{"op": "remove", "item": "i-letter",
                              "location": "s-station"}]},
            ensure_ascii=False)
        goal = json.dumps({"events": [], "goal_updates": {},
                           "new_goals": []}, ensure_ascii=False)
        recorder = ScriptedLLM([reply, reply, reply, goal])
        w.npcs["n-arin"].state.last_turn = w.turn
        interpreter.dialogue_turn(recorder, w, w.npcs["n-arin"], "把信给我。")
        # 信离开车站物品表，进入玩家携带物
        self.assertFalse(any(i["id"] == "i-letter"
                             for i in w.scenes["s-station"].items))
        self.assertTrue(any(i["name"] == "一封信"
                            for i in w.player.get("items", [])))
        self.assertTrue(any(e.kind == "item_removed" for e in w.events))


class TestPacing(unittest.TestCase):
    """节奏：大事喘息开关 + 日常小事通道。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-alley"
        return llm, w

    def _big_event(self):
        return {"type": "world_event",
                "params": {"title": "大事", "detail": "天空裂开",
                           "location": "s-station", "intensity": 0.9}}

    def test_big_event_cooldown_gives_world_room_to_breathe(self):
        llm, w = self._world()
        w.pass_time(6)
        plan = json.dumps({"events": [self._big_event()],
                           "npc_plans": [], "influences": [],
                           "daily_bits": [], "new_scenes": [],
                           "item_patches": []}, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        big = [e for e in w.events if e.kind == "world_event"
               and (e.payload or {}).get("event_params", {})
               .get("intensity") == 0.9]
        self.assertEqual(len(big), 1)  # 第一件大事放行
        # 紧接着再来一件大事 → 喘息开关否决
        w.pass_time(6)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertEqual(len([e for e in w.events if e.kind == "world_event"
                              and (e.payload or {}).get("event_params", {})
                              .get("intensity") == 0.9]), 1)

    def test_big_event_allowed_after_cooldown(self):
        llm, w = self._world()
        w.pass_time(6)
        plan = json.dumps({"events": [self._big_event()],
                           "npc_plans": [], "influences": [],
                           "daily_bits": [], "new_scenes": [],
                           "item_patches": []}, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        w.pass_time(30)  # 超过喘息期
        evolution.world_pulse(ScriptedLLM([plan]), w)
        big = [e for e in w.events if e.kind == "world_event"
               and (e.payload or {}).get("event_params", {})
               .get("intensity") == 0.9]
        self.assertEqual(len(big), 2)

    def test_daily_bits_are_logged(self):
        llm, w = self._world()
        w.pass_time(6)
        plan = json.dumps({
            "events": [], "npc_plans": [], "influences": [],
            "daily_bits": [{"detail": "檐角的猫躲进纸箱里",
                            "location": "s-station", "intensity": 0.2}],
            "new_scenes": [], "item_patches": []}, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        dailies = [e for e in w.events if e.kind == "daily_life"]
        self.assertTrue(dailies)
        self.assertIn("猫", dailies[-1].summary)
        # 日常小事也进入场景变化记忆
        self.assertTrue(any("猫" in r["summary"]
                            for r in w.scenes["s-station"].recent))

    def test_daily_bit_bad_location_rejected(self):
        llm, w = self._world()
        w.pass_time(6)
        plan = json.dumps({
            "events": [], "npc_plans": [], "influences": [],
            "daily_bits": [{"detail": "某处的小事", "location": "s-ghost",
                            "intensity": 0.2}],
            "new_scenes": [], "item_patches": []}, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertFalse(any(e.kind == "daily_life" for e in w.events))


class TestPlayerActions(unittest.TestCase):
    """玩家动作层：类型 + 关系阈值门 + 接受/拒绝 + 后果写回。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-cafe"
        return llm, w

    def test_low_relationship_kiss_is_refused(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]  # 关系 0：夹具启发式判拒
        r = interpreter.player_action(llm, w, arin, "我亲了她。")
        self.assertFalse(r.accepted)
        self.assertLessEqual(arin.relationship, 0)  # 拒绝没有关系收益
        kinds = [e.kind for e in w.events]
        self.assertIn("player_acted", kinds)
        self.assertIn("action_refused", kinds)

    def test_handshake_accepted_at_neutral(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        r = interpreter.player_action(llm, w, arin, "我伸出手，想和她握手。")
        self.assertTrue(r.accepted)
        event = next(e for e in w.events if e.kind == "player_acted")
        self.assertEqual((event.payload or {}).get("event_params", {}).get(
            "location"), w.player["location"])
        self.assertFalse(any(e.kind == "action_refused" for e in w.events))

    def test_player_action_binds_visible_targets_and_rejects_remote_ones(self):
        _, w = self._world()
        arin = w.npcs["n-arin"]
        response = json.dumps({
            "accepted": True, "reply": "她看了一眼你的手。",
            "relationship_delta": 0, "mood_delta": 0.0,
            "memory_importance": 0.4, "memory": "我看见他指向了车站的信。",
            "law_ids": [], "days": 0.0,
            "targets": ["item:i-letter"]}, ensure_ascii=False)
        result = interpreter.player_action(
            ScriptedLLM([response]), w, arin, "我指向车站的那封信。")
        self.assertFalse(result.accepted)
        self.assertIn("当前可见范围", result.reply)
        event = next(e for e in w.events if e.kind == "player_acted")
        self.assertEqual(event.payload["event_params"]["targets"],
                         ["item:i-letter"])

        valid = json.dumps({
            "accepted": True, "reply": "她点了点头。",
            "relationship_delta": 0, "mood_delta": 0.0,
            "memory_importance": 0.4, "memory": "我看见他指了指窗边。",
            "law_ids": [], "days": 0.0,
            "targets": ["npc:n-arin", "scene:s-cafe"]}, ensure_ascii=False)
        result = interpreter.player_action(
            ScriptedLLM([valid]), w, arin, "我指了指窗边。")
        self.assertTrue(result.accepted)
        self.assertEqual(result.targets, ["npc:n-arin", "scene:s-cafe"])

    def test_high_relationship_kiss_accepted(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.relationship = 60  # 长关系
        r = interpreter.player_action(llm, w, arin, "我亲了她。")
        self.assertTrue(r.accepted)

    def test_action_is_remembered(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        interpreter.player_action(llm, w, arin, "我伸出手，想和她握手。")
        self.assertTrue(any("握" in m.content for m in arin.memories))
        ev = [e for e in w.events if e.kind == "npc_memory"]
        self.assertTrue(any("动作" in e.cause for e in ev))

    def test_action_reply_enters_player_visible_ledger(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        result = interpreter.player_action(llm, w, arin,
                                           "我伸出手，想和她握手。")
        replies = [e for e in w.events
                   if e.kind == "dialogue"
                   and e.cause == "玩家动作"]
        self.assertTrue(replies)
        self.assertIn(result.reply,
                      (replies[-1].payload or {}).get("event_params", {})
                      .get("reply", ""))

    def test_action_does_not_apply_after_catchup_moves_npc_away(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.player["location"] = "s-station"
        w.pass_time(6)
        catchup = json.dumps({
            "location": "s-cafe", "activity": "收拾柜台",
            "mood": "平静", "moved": True, "memory": ""},
            ensure_ascii=False)
        result = interpreter.player_action(
            ScriptedLLM([catchup]), w, arin, "我伸手碰了碰她的肩膀")
        self.assertFalse(result.accepted)
        self.assertIn("不在这里", result.reply)
        self.assertFalse(any(e.kind == "player_acted" for e in w.events))

    def test_dialogue_does_not_reach_npc_after_catchup_moves_away(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.player["location"] = "s-station"
        w.pass_time(6)
        catchup = json.dumps({
            "location": "s-cafe", "activity": "收拾柜台",
            "mood": "平静", "moved": True, "memory": ""},
            ensure_ascii=False)
        result = interpreter.dialogue_turn(
            ScriptedLLM([catchup]), w, arin, "你去哪了？")
        self.assertIn("不在这里", result.reply)
        self.assertFalse(any(e.kind == "player_said" for e in w.events))

    def test_refusal_hurts_relationship(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        r = interpreter.player_action(llm, w, arin, "我把她推开了。")
        self.assertTrue(r.accepted)  # 推开无阈值
        self.assertLess(arin.relationship, 0)  # 推开伤害关系
        self.assertLess(arin.state.mood_value, 0)  # 情绪受损


class TestPopulationEcology(unittest.TestCase):
    """人口生态：新角色涌现（事件/生活驱动，有因才生、冷却有界）。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-cafe"
        return llm, w

    def _proposal(self, name="新来的学妹"):
        return {"name": name,
                "persona": "开学季转学来的少女，安静，总在雨里看天空。",
                "goal": {"id": "g-n1", "text": "在这座城交到第一个朋友",
                         "progress": 0.0},
                "location": "s-cafe",
                "reason": "开学季，新生入学"}

    def test_new_npc_emerges_with_reason_and_goal(self):
        llm, w = self._world()
        before = len(w.npcs)
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [], "influences": [],
                           "daily_bits": [], "new_npcs": [self._proposal()],
                           "new_scenes": [], "item_patches": []},
                          ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertEqual(len(w.npcs), before + 1)
        girl = next(n for n in w.npcs.values() if n.name == "新来的学妹")
        self.assertEqual(girl.goals[0]["text"], "在这座城交到第一个朋友")
        self.assertIn("npc_emerged", [e.kind for e in w.events])
        self.assertTrue(any(e.cause == "世界演化" for e in w.events))

    def test_new_npc_without_reason_rejected(self):
        llm, w = self._world()
        proposal = self._proposal()
        proposal["reason"] = ""
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [], "influences": [],
                           "daily_bits": [], "new_npcs": [proposal],
                           "new_scenes": [], "item_patches": []},
                          ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertTrue(any("缘由必填" in s for s in summaries))
        self.assertFalse(any(e.kind == "npc_emerged" for e in w.events))

    def test_population_influx_has_cooldown(self):
        llm, w = self._world()
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [], "influences": [],
                           "daily_bits": [], "new_npcs": [self._proposal()],
                           "new_scenes": [], "item_patches": []},
                          ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        # 紧接着再来一个 → 冷却否决
        w.pass_time(6)
        plan2 = plan.replace("新来的学妹", "另一个新人")
        summaries = evolution.world_pulse(ScriptedLLM([plan2]), w)
        self.assertTrue(any("人口在休息" in s for s in summaries))
        self.assertEqual(len(w.npcs), 4)  # 还是 3+1

    def test_mock_life_newcomer_arrives(self):
        llm, w = self._world()
        w.pass_time(50)  # 越过第 48 回合：生活流入
        evolution.world_pulse(llm, w)
        self.assertTrue(any(n.name == "学徒·小晴"
                            for n in w.npcs.values()))
        ev = [e for e in w.events if e.kind == "npc_emerged"]
        self.assertTrue(ev)
        self.assertIn("开学季", ev[-1].summary)

    def test_new_npc_can_arrive_with_local_activity(self):
        """新人物到场时可带当前状态，但不凭空开始一个已执行的动作。"""
        llm, w = self._world()
        proposal = self._proposal("修表匠·禾")
        proposal["activity"] = "蹲在钟楼下检查一块停摆的怀表"
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [], "influences": [],
                           "daily_bits": [], "new_npcs": [proposal],
                           "new_scenes": [], "item_patches": []},
                          ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        he = next(n for n in w.npcs.values() if n.name == "修表匠·禾")
        self.assertEqual(he.state.activity, proposal["activity"])
        self.assertEqual(he.state.action.text, "")

    def test_emergent_npc_is_immediately_an_ordinary_npc(self):
        """调度频率是预算问题，新人物不带本体论上的角色等级。"""
        llm, w = self._world()
        errors, hid = cards.emerge_npc(
            w, "货郎", "挑担路过的货郎，卖针线。",
            {"id": "g-sell", "text": "卖完这担针线", "progress": 0.0},
            "s-cafe", "集市日，货郎经过")
        self.assertEqual(errors, [])
        huo = w.npcs[hid]
        self.assertEqual(huo.goals[0]["text"], "卖完这担针线")
        self.assertFalse(hasattr(huo, "is_passersby"))
        w.pass_time(6)
        payload = evolution.build_pulse_payload(w, [huo], {}, "s-cafe")
        self.assertIn(huo.id, [n["id"] for n in payload["due_npcs"]])

    def test_crowd_text_is_scene_texture(self):
        llm, w = self._world()
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [], "influences": [],
                           "daily_bits": [],
                           "crowds": [{"location": "s-station",
                                       "text": "开学季，站台多了几张新面孔"}],
                           "new_npcs": [], "new_scenes": [],
                           "item_patches": []}, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertEqual(w.scenes["s-station"].crowd,
                         "开学季，站台多了几张新面孔")
        self.assertTrue(any("新面孔" in e.summary for e in w.events))

    def test_crowd_cannot_narrate_named_npc_action(self):
        """人流快照在写入前也遵守角色状态边界。"""
        llm, w = self._world()
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [], "influences": [],
                           "daily_bits": [],
                           "crowds": [{"location": "s-station",
                                       "text": "阿凛冒雨走过站台"}],
                           "new_npcs": [], "new_scenes": [],
                           "item_patches": []}, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertEqual(w.scenes["s-station"].crowd, "")
        self.assertTrue(any("驳回人流" in s for s in summaries))

    def test_payload_bounded_after_fade_refill_cycles(self):
        """P1：反复「补满→全退雾→再补」后，脉冲载荷仍有界。

        人口信号只代表活跃人口；雾中人口只走 fog_count + 最近 20 位。
        账本可无限厚，载荷永远薄。
        """
        llm, w = self._world()
        for cycle in range(20):
            while cards._active_count(w) < physics.MAX_NPCS:
                errors, _ = cards.emerge_npc(
                    w, f"路人{cycle}-{cards._active_count(w)}", "普通人。",
                    None, "s-cafe", "集市日，人来人往")
                self.assertEqual(errors, [])
            for n in w.npcs.values():
                n.in_fog = True
                n.fog_note = "退去雾中。"
                for s in w.scenes.values():
                    if n.id in s.npcs:
                        s.npcs.remove(n.id)
        self.assertGreater(len(w.npcs), 100)  # 账本确实很厚（雾中人口）
        payload = evolution.build_pulse_payload(
            w, [], {}, w.player["location"])
        self.assertLessEqual(payload["population"]["count"],
                             physics.MAX_NPCS)
        self.assertLessEqual(len(payload["population"]["members"]),
                             physics.MAX_NPCS)
        self.assertLessEqual(len(payload["fog_npcs"]), 20)
        size = len(json.dumps(payload, ensure_ascii=False))
        self.assertLess(size, 12000)  # 载荷有界：不随雾中人口增长

    def test_emergent_npc_survives_roundtrip(self):
        llm, w = self._world()
        cards.emerge_npc(w, "货郎", "挑担路过的货郎。", None, "s-cafe",
                         "集市日")
        u = Universe(worlds={w.name: w}, current=w.name)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "u.json"
            u.save(p)
            u2 = Universe.load(p)
            huo2 = next(n for n in u2.here.npcs.values()
                        if n.name == "货郎")
            self.assertEqual(huo2.state.location, "s-cafe")

    def test_fog_npc_with_goals_still_lives(self):
        """雾中有目标的角色：生活照旧——雾只是我们看不见，不是时间冻结。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]  # 有目标
        w.player["location"] = "s-cafe"  # 玩家去咖啡店
        evolution.move_npc(w, arin, "s-alley", cause="测试")  # 她去了雾中小巷
        stamp = arin.state.last_turn
        w.pass_time(50)  # 超过 LIFE_INTERVAL
        evolution.world_pulse(llm, w)
        self.assertGreater(arin.state.last_turn, stamp)  # 雾中也在过日子

    def test_fog_npc_without_goals_stays_quiet(self):
        """雾中无目标的角色：保持安静（没有故事要推，不占裁决）。"""
        llm, w = self._world()
        man = cards.create_npc(w, "闲人", "无所事事的闲人。")
        if man:
            man = next(n for n in w.npcs.values() if n.name == "闲人")
        else:
            man = next(n for n in w.npcs.values() if n.name == "闲人")
        evolution.move_npc(w, man, "s-alley", cause="测试")
        stamp = man.state.last_turn
        w.pass_time(50)
        evolution.world_pulse(llm, w)
        self.assertEqual(man.state.last_turn, stamp)


class TestNpcActionability(unittest.TestCase):
    """角色能否继续行动：世界写词汇，引擎守后果。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-cafe"
        return w, w.npcs["n-arin"]

    @staticmethod
    def _source(w, title="重大事件"):
        emit(w, "world_event", {
            "title": title, "detail": f"{title}留下了无法忽略的痕迹",
            "location": "s-cafe", "intensity": 0.6}, cause="测试")
        return w.events[-1].summary

    @staticmethod
    def _plan(events):
        return json.dumps({
            "events": events, "npc_plans": [], "daily_bits": [],
            "new_npcs": [], "new_scenes": [], "item_patches": [],
            "crowds": []}, ensure_ascii=False)

    def test_world_plan_can_stop_an_actor_with_a_cause(self):
        w, arin = self._world()
        source = self._source(w)
        w.pass_time(6)
        plan = self._plan([{
            "type": "npc_state_changed",
            "params": {"npc": arin.id, "can_act": False,
                       "condition": "重大事件后暂时停摆",
                       "cause_event": source}}])
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertFalse(arin.state.can_act)
        self.assertFalse(arin.in_fog)
        self.assertEqual(arin.state.condition, "重大事件后暂时停摆")
        self.assertEqual(arin.state.activity, "重大事件后暂时停摆")
        self.assertFalse(any(arin.id in scene.npcs
                             for scene in w.scenes.values()))
        event = next(e for e in reversed(w.events)
                     if e.kind == "npc_state_changed")
        self.assertEqual(event.cause, f"事件：{source}")

    def test_non_actionable_npc_cannot_be_advanced_or_addressed(self):
        w, arin = self._world()
        source = self._source(w)
        evolution.set_actionability(
            w, arin, False, "重大事件后暂时停摆", source)
        stamp = arin.state.last_clock
        w.pass_time(12)
        evolution.world_pulse(ScriptedLLM([self._plan([])]), w)
        self.assertEqual(arin.state.last_clock, stamp)
        result = interpreter.dialogue_turn(ScriptedLLM([]), w, arin, "你还好吗？")
        self.assertIn("无法回应", result.reply)
        self.assertFalse(any(e.kind == "player_said" for e in w.events))

    def test_reactivation_is_a_new_caused_event(self):
        w, arin = self._world()
        fallen = self._source(w)
        evolution.set_actionability(
            w, arin, False, "重大事件后暂时停摆", fallen)
        rewritten = self._source(w, "过去被改写")
        summaries = evolution.set_actionability(
            w, arin, True, "历史改写后仍然活着", rewritten)
        self.assertTrue(arin.state.can_act)
        self.assertIn(arin.id, w.scenes[arin.state.location].npcs)
        self.assertTrue(any("恢复行动" in summary for summary in summaries))
        self.assertEqual(
            [e.kind for e in w.events].count("npc_state_changed"), 2)

    def test_state_change_rejects_unrecorded_cause(self):
        w, arin = self._world()
        summaries = evolution.set_actionability(
            w, arin, False, "重大事件后暂时停摆", "并不存在的事件")
        self.assertTrue(any("不在近期账本" in summary for summary in summaries))
        self.assertTrue(arin.state.can_act)

    def test_state_change_rejects_location_in_condition(self):
        w, arin = self._world()
        source = self._source(w)
        condition = f"昏迷在{w.scenes['s-cafe'].name}"
        summaries = evolution.set_actionability(
            w, arin, False, condition, source)
        self.assertTrue(any("不得声明地点" in summary for summary in summaries))
        self.assertTrue(arin.state.can_act)

    def test_state_change_rejects_a_narrated_event_chain(self):
        w, arin = self._world()
        source = self._source(w)
        summaries = evolution.set_actionability(
            w, arin, False,
            "深夜独自前往北坡，不慎滑倒后暂时失去行动能力", source)
        self.assertTrue(any("超长" in summary for summary in summaries))
        self.assertTrue(arin.state.can_act)

    def test_state_change_requires_a_recent_recorded_cause(self):
        w, arin = self._world()
        source = self._source(w)
        for index in range(8):
            self._source(w, f"后续事件{index}")
        summaries = evolution.set_actionability(
            w, arin, False, "已死亡", source)
        self.assertTrue(any("不在近期账本" in summary for summary in summaries))
        self.assertTrue(arin.state.can_act)

    def test_non_actionable_npc_does_not_receive_rumors(self):
        w, arin = self._world()
        source = self._source(w)
        evolution.set_actionability(
            w, arin, False, "重大事件后暂时停摆", source)
        before = len(arin.memories)
        evolution.spread_rumor(None, w, "world_event", {
            "location": "s-cafe", "title": "新的异象"})
        self.assertEqual(len(arin.memories), before)

    def test_fact_change_cannot_narrate_inactive_actor(self):
        w, arin = self._world()
        source = self._source(w)
        evolution.set_actionability(w, arin, False,
                                   "重大事件后暂时停摆", source)
        w.facts = ["旧法则"]
        w.pass_time(6)
        plan = self._plan([])
        data = json.loads(plan)
        data["fact_changes"] = [{
            "op": "change", "old": "旧法则", "fact": "新法则",
            "why": "阿凛在废墟中苏醒，看见了新的征兆"}]
        summaries = evolution.world_pulse(
            ScriptedLLM([json.dumps(data, ensure_ascii=False)]), w)
        self.assertEqual(w.facts, ["旧法则"])
        self.assertTrue(any("不得代替具名角色行动" in summary
                            for summary in summaries))


class _CaptureLLM:
    """捕获裁决载荷的测试 LLM：返回空计划，但记下最近一次 user 载荷。"""

    last_user = None

    def chat(self, system: str, user: str) -> str:
        self.__class__.last_user = user
        return "{}"

    def chat_json(self, system: str, user: str, attempts: int = 2) -> dict:
        return json.loads(self.chat(system, user))


class TestCausalWorld(unittest.TestCase):
    """因果世界：日常累积、目标碰撞、同世界分叉。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-alley"
        return llm, w

    def _plan(self, **extra):
        base = {"events": [], "npc_plans": [], "influences": [],
                "daily_bits": [], "new_scenes": [], "item_patches": [],
                "new_npcs": [], "crowds": []}
        base.update(extra)
        return json.dumps(base, ensure_ascii=False)

    def test_daily_bit_trace_accumulates_on_item(self):
        llm, w = self._world()
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("name") == "一封信")
        old_note = letter.get("note", "")
        w.pass_time(6)
        plan = self._plan(daily_bits=[{
            "detail": "雨滴沿着檐角滴在信封上", "location": "s-station",
            "intensity": 0.2,
            "trace": {"item": "i-letter", "change": "信纸又湿了一点"}}])
        summaries = evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertIn("信纸又湿了一点", letter["note"])
        self.assertNotEqual(letter["note"], old_note)
        self.assertTrue(any("渐渐变了" in s for s in summaries))  # 日志里也有
        ev = next(e for e in w.events if e.kind == "daily_life"
                  and e.payload.get("event_params", {}).get("item"))
        self.assertEqual(ev.payload["event_params"]["item"], "i-letter")

    def test_similar_ambient_lines_in_one_scene_are_suppressed(self):
        """换几个字重说同一幕不是两件世界事实。"""
        llm, w = self._world()
        w.pass_time(6)
        plan = self._plan(daily_bits=[
            {"detail": "摊主们开始收摊，挑着担子往巷口走",
             "location": "s-station", "intensity": 0.1},
            {"detail": "收摊的摊主挑起担子，陆续朝巷口走去",
             "location": "s-station", "intensity": 0.1},
            {"detail": "屋檐下飘出晚饭的炊烟",
             "location": "s-station", "intensity": 0.1},
        ])
        evolution.world_pulse(ScriptedLLM([plan]), w)
        lines = [e for e in w.events if e.kind == "daily_life"
                 and e.payload.get("event_params", {}).get("location")
                 == "s-station"]
        self.assertEqual(len(lines), 2)

    def test_item_fold_is_not_broken_by_unrelated_ledger_events(self):
        """微变按世界时间折叠，不因期间日志很多而伪装成多次跃变。"""
        llm, w = self._world()
        first = self._plan(daily_bits=[{
            "detail": "雨滴落在信封上", "location": "s-station",
            "intensity": 0.2,
            "trace": {"item": "i-letter", "change": "信纸湿了一角"}}])
        w.pass_time(6)
        evolution.world_pulse(ScriptedLLM([first]), w)
        for index in range(12):
            w.log("npc_memory", f"无关记录 {index}", "测试")
        second = self._plan(daily_bits=[{
            "detail": "雨仍落在信封上", "location": "s-station",
            "intensity": 0.2,
            "trace": {"item": "i-letter", "change": "字迹开始晕开"}}])
        w.pass_time(6)
        evolution.world_pulse(ScriptedLLM([second]), w)
        traces = [e for e in w.events if e.kind == "daily_life"
                  and e.payload.get("event_params", {}).get("item")
                  == "i-letter"]
        self.assertEqual(len(traces), 1)
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("id") == "i-letter")
        self.assertEqual(letter["note"], "字迹开始晕开")

    def test_trace_note_is_bounded_snapshot(self):
        """note 是有界的当前快照；历史在事件账本里。"""
        llm, w = self._world()
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("name") == "一封信")
        changes = ["信纸又湿了一点", "信封皱起一角", "字迹开始晕开"]
        for chg in changes:
            w.pass_time(6)
            plan = self._plan(daily_bits=[{
                "detail": "雨滴落在信封上", "location": "s-station",
                "intensity": 0.2,
                "trace": {"item": "i-letter", "change": chg}}])
            evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertEqual(letter["note"], "字迹开始晕开")  # 快照，不是历史
        self.assertLess(len(letter["note"]), 100)
        evs = [e for e in w.events if e.kind == "daily_life"]
        self.assertTrue(any("信纸又湿了一点" in e.summary for e in evs))
        # 中间微变折叠在快照里，不在事件账本（有界留痕）
        self.assertFalse(any("字迹开始晕开" in e.summary for e in evs))

    def test_trace_consecutive_changes_fold_until_leap(self):
        """连续渐变折叠：中间微变只覆写快照，跃变时一条总结入账。"""
        llm, w = self._world()
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("name") == "一封信")
        changes = ["信纸又湿了一点", "信封皱起一角", "字迹开始晕开"]
        for chg in changes:
            w.pass_time(6)
            plan = self._plan(daily_bits=[{
                "detail": "雨滴落在信封上", "location": "s-station",
                "intensity": 0.2,
                "trace": {"item": "i-letter", "change": chg}}])
            evolution.world_pulse(ScriptedLLM([plan]), w)
        # 折叠中：只有首条入账，中间微变不在事件账本（note 是最新快照）
        evs = [e for e in w.events if e.kind == "daily_life"
               and "信" in e.summary]
        self.assertEqual(len(evs), 1)
        self.assertEqual(letter["note"], "字迹开始晕开")
        # 跃变（转手）收尾折叠：一条总结入账，保留因果根与末态
        from worldledger.event import transfer_item
        w.pass_time(6)
        transfer_item(w, letter, "player", "测试")
        evs = [e for e in w.events if e.kind == "daily_life"
               and "渐变" in e.summary]
        self.assertTrue(evs)
        self.assertIn("字迹开始晕开", evs[-1].summary)  # 末态保留
        self.assertIn("2 次", evs[-1].summary)          # 折叠数

    def test_item_change_records_cause_turn(self):
        """cause_turn：物品最后变更可机械追查到账本序号。"""
        llm, w = self._world()
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("name") == "一封信")
        w.pass_time(6)
        errors = apply_item_patch(w, {"op": "change", "item": "i-letter",
                                      "location": "s-station",
                                      "note": "被拆开了"},
                                  "有人拆信")
        self.assertFalse(errors)
        self.assertEqual(letter["cause_turn"], w.turn)
        self.assertEqual(letter["cause"], "有人拆信")
        self.assertIn("被拆开了", letter["note"])

    def test_daily_bit_trace_bad_ref_rejected(self):
        llm, w = self._world()
        w.pass_time(6)
        plan = self._plan(daily_bits=[{
            "detail": "什么也没发生", "location": "s-station",
            "intensity": 0.2,
            "trace": {"item": "i-ghost", "change": "不该存在的变化"}}])
        summaries = evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertTrue(any("驳回累积" in s for s in summaries))

    def test_goal_collision_floats_and_logs(self):
        llm, w = self._world()
        arin, man = w.npcs["n-arin"], w.npcs["n-man"]
        evolution.move_npc(w, man, "s-cafe", cause="测试")  # 两个目标者同到期
        arin.goals[0]["targets"] = ["一封信"]
        man.goals[0]["targets"] = ["一封信"]
        w.pass_time(12)
        evolution.world_pulse(ScriptedLLM([self._plan()]), w)
        kinds = [e.kind for e in w.events]
        self.assertIn("collision", kinds)
        ev = next(e for e in w.events if e.kind == "collision")
        self.assertIn("一封信", ev.summary)
        self.assertEqual(ev.cause, "目标碰撞")  # 入账留痕

    def test_invalid_targets_dropped_at_canonicalization(self):
        """无效引用在写入时就被丢弃（不落库、不参与碰撞）。"""
        llm, w = self._world()
        from worldledger.store import canonical_targets
        out = canonical_targets(w, ["一封信", "item:i-ghost", "不存在的地名"])
        self.assertEqual(out, ["item:i-letter"])  # 只留解析成功的

    def test_goal_adjudicator_sees_recent_events(self):
        """裁决者看得见自己场景里刚发生的事（知识边界内）。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        loc = arin.state.location
        emit(w, "item_changed",
             {"item": "i-letter", "location": loc,
              "note": "信封被撕开过"}, cause="测试")
        cap = _CaptureLLM()
        evolution.propose_proactive(cap, w, arin)
        user = json.loads(cap.last_user)
        self.assertTrue(any("被撕开过" in e
                            for e in user.get("recent_events", [])))

    def test_invalid_target_refs_leave_no_garbage(self):
        """无效引用不参与碰撞、不在 social 留垃圾。"""
        llm, w = self._world()
        arin, man = w.npcs["n-arin"], w.npcs["n-man"]
        evolution.move_npc(w, man, "s-cafe", cause="测试")
        arin.goals[0]["targets"] = ["item:i-ghost"]
        man.goals[0]["targets"] = ["item:i-ghost"]
        w.pass_time(12)
        evolution.world_pulse(ScriptedLLM([self._plan()]), w)
        self.assertNotIn("collision", [e.kind for e in w.events])
        self.assertFalse(any(k.startswith("coll|") for k in w.social))

    def test_collision_state_stays_out_of_model_payload(self):
        """碰撞边沿检测是引擎内部状态，不进模型载荷。"""
        llm, w = self._world()
        arin, man = w.npcs["n-arin"], w.npcs["n-man"]
        evolution.move_npc(w, man, "s-cafe", cause="测试")
        arin.goals[0]["targets"] = ["一封信"]
        man.goals[0]["targets"] = ["一封信"]
        w.pass_time(12)
        evolution.world_pulse(ScriptedLLM([self._plan()]), w)
        self.assertTrue(any(k.startswith("coll|") for k in w.social))
        payload = evolution.build_pulse_payload(
            w, [], {}, w.player["location"])
        self.assertFalse(any(k.startswith("coll|")
                             for k in payload["social"]))

    def test_scene_changes_exports_ledger_facts(self):
        """重返场景：从账本机械导出事实变化（零 LLM、附事件 id）。"""
        llm, w = self._world()
        stamp = w.turn
        emit(w, "item_changed",
             {"item": "i-letter", "location": "s-station",
              "note": "信封被撕开过"}, cause="测试")
        emit(w, "daily_life",
             {"detail": "台阶的沟又深了", "location": "s-station",
              "intensity": 0.2, "item": "i-letter"}, cause="测试")
        emit(w, "npc_interaction",
             {"npc": "n-arin", "target": "n-man",
              "line": "雨还没有停。", "location": "s-station"}, cause="测试")
        evolution.move_npc(w, w.npcs["n-man"], "s-station", cause="测试")
        changes = scene_changes(w, "s-station", stamp, limit=4)
        self.assertTrue(changes)
        self.assertTrue(all(c["event_id"] for c in changes))
        self.assertTrue(all(c["cause"] for c in changes))
        kinds = {c["kind"] for c in changes}
        self.assertIn("item_changed", kinds)
        self.assertIn("npc_interaction", kinds)
        self.assertIn("npc_moved", kinds)  # 谁来过/离开
        self.assertFalse(any("_event_index" in change for change in changes))
        self.assertLessEqual(len(changes), 4)  # 有界

    def test_goal_targets_canonicalized_at_write(self):
        """写入时规范化：裸名落库即 id 形式（同名物品也无歧义）。"""
        llm, w = self._world()
        errors, hid = cards.emerge_npc(
            w, "货郎", "挑担路过的货郎。",
            {"id": "g-1", "text": "找到那封信", "progress": 0.0,
             "targets": ["一封信"]},
            "s-cafe", "集市日")
        self.assertEqual(errors, [])
        huo = w.npcs[hid]
        self.assertEqual(huo.goals[0]["targets"], ["item:i-letter"])

    def test_scene_changes_returns_latest_window(self):
        """长时离开：返回的是最新 limit 条，不是最早的命中。"""
        llm, w = self._world()
        stamp = w.turn
        for i in range(8):
            emit(w, "item_changed",
                 {"item": "i-letter", "location": "s-station",
                  "note": f"变化{i}"}, cause="测试")
        changes = scene_changes(w, "s-station", stamp, limit=4)
        notes = [c["fact"] for c in changes]
        self.assertEqual(len(changes), 4)
        self.assertIn("变化7", notes[-1])   # 最新
        self.assertIn("变化4", notes[0])    # 窗口起点
        self.assertNotIn("变化0", " ".join(notes))  # 最早的被正确跳过

    def test_scene_seen_cursor_reads_limited_changes_without_skipping(self):
        """回读分页：6 条待读、每次 4 条，分两次完整消费。"""
        llm, w = self._world()
        stamp = w.turn
        for i in range(6):
            emit(w, "npc_interaction", {
                "npc": "n-arin", "target": "n-man",
                "line": f"对话{i}", "location": "s-station",
            }, cause="测试")
        from worldledger import main as main_mod
        with redirect_stdout(StringIO()):
            main_mod._print_scene_changes(w, "s-station")
        first_cursor = w.player["seen_event_indices"]["s-station"]
        self.assertEqual(
            w.player["seen"]["s-station"],
            w.events[first_cursor].turn,
        )
        self.assertLess(first_cursor, len(w.events) - 1)
        with redirect_stdout(StringIO()):
            main_mod._print_scene_changes(w, "s-station")
        second_cursor = w.player["seen_event_indices"]["s-station"]
        self.assertEqual(second_cursor, len(w.events) - 1)
        self.assertEqual(second_cursor - first_cursor, 2)
        with redirect_stdout(StringIO()) as output:
            main_mod._print_scene_changes(w, "s-station")
        self.assertEqual(output.getvalue(), "")

    def test_scene_seen_progress_lives_in_player_not_social(self):
        """阅读进度是玩家状态，不进 social / 模型载荷。"""
        llm, w = self._world()
        emit(w, "item_changed",
             {"item": "i-letter", "location": "s-station",
              "note": "信封变了"}, cause="测试")
        from worldledger import main as main_mod
        main_mod._print_scene_changes(w, "s-station")
        self.assertIn("seen", w.player)
        self.assertEqual(w.player["seen"]["s-station"], w.turn)
        self.assertFalse(any(k.startswith("seen|") for k in w.social))
        payload = evolution.build_pulse_payload(
            w, [], {}, w.player["location"])
        self.assertFalse(any(k.startswith("seen|")
                             for k in payload["social"]))

    def test_fact_changes_are_causal_and_traced(self):
        """世界设定是活的：增/删/改有因才变，留痕可回放。"""
        llm, w = self._world()
        w.facts = ["夜间短暂交换行动权限", "黄昏时在旧堤相见"]
        w.pass_time(6)
        plan = json.dumps({
            "events": [], "npc_plans": [], "influences": [],
            "daily_bits": [], "new_scenes": [], "item_patches": [],
            "new_npcs": [], "crowds": [],
            "fact_changes": [
                {"op": "change", "old": "夜间短暂交换行动权限",
                 "fact": "潮汐窗口过后，交换无法再发生",
                 "why": "潮汐塔停机，连接两人的信号中断了"},
            ]}, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertIn("潮汐窗口过后，交换无法再发生", w.facts)
        self.assertNotIn("夜间短暂交换行动权限", w.facts)
        ev = [e for e in w.events if e.kind == "fact_changed"]
        self.assertTrue(ev)
        self.assertEqual(ev[-1].cause, "潮汐塔停机，连接两人的信号中断了")
        self.assertTrue(any("设定变了" in s for s in summaries))

    def test_fact_change_without_why_rejected(self):
        """有因才变：没写为什么的设定变更被驳回。"""
        llm, w = self._world()
        w.facts = ["夜间短暂交换行动权限"]
        w.pass_time(6)
        plan = json.dumps({
            "events": [], "npc_plans": [], "influences": [],
            "daily_bits": [], "new_scenes": [], "item_patches": [],
            "new_npcs": [], "crowds": [],
            "fact_changes": [
                {"op": "remove", "old": "夜间短暂交换行动权限", "why": ""},
            ]}, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertTrue(any("原因必填" in s for s in summaries))
        self.assertEqual(w.facts, ["夜间短暂交换行动权限"])  # 没动

    def test_no_change_updates_are_dropped(self):
        """无变化不重复入账：事实改写成原样、天气覆写同状态都被拒。"""
        llm, w = self._world()
        w.facts = ["夜间短暂交换行动权限"]
        w.weather = "暴雨"
        w.pass_time(6)
        plan = json.dumps({
            "events": [{"type": "weather_shift",
                        "params": {"to": "暴雨", "intensity": 0.9}}],
            "npc_plans": [], "influences": [],
            "daily_bits": [], "new_scenes": [], "item_patches": [],
            "new_npcs": [], "crowds": [],
            "fact_changes": [
                {"op": "change", "old": "夜间短暂交换行动权限",
                 "fact": "夜间短暂交换行动权限", "why": "想强调一遍"},
            ]}, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertFalse(any(e.kind == "weather_shift"
                             for e in w.events))  # 同状态天气不重复
        self.assertFalse(any(e.kind == "fact_changed"
                             for e in w.events))  # 无变化事实不重复
        self.assertEqual(w.facts, ["夜间短暂交换行动权限"])

    def test_acting_in_scene_marks_it_seen(self):
        """你做过事的地方不是「你不在的地方」：行动推进 seen 进度。

        通用规则（与互换无关）：在借来的身体里做事，醒来后
        回到该场景，不该收到「你不在时做了什么」的通知。
        """
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.player["location"] = "s-cafe"
        emit(w, "item_changed",
             {"item": "i-letter", "location": "s-station",
              "note": "信封变了"}, cause="测试")
        before = w.turn
        w.pass_time(12)  # 世界往前走
        interpreter.dialogue_turn(llm, w, arin, "你在等谁？")  # 玩家在这里做事
        seen = w.player.get("seen", {})
        self.assertEqual(seen.get(w.player["location"]), w.turn)
        # 之后回到该场景：不会有「你不在时」的旧闻（since 已推进）
        from worldledger import main as main_mod
        main_mod._print_scene_changes(w, w.player["location"])
        self.assertEqual(seen[w.player["location"]], w.turn)

    def test_sweep_protects_fact_and_goal_referenced_items(self):
        """引用即保鲜：被 facts 或活跃目标引用的物品不被时间扫走。"""
        llm, w = self._world()
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("id") == "i-letter")
        w.facts = ["一封信是找到真相的唯一钥匙"]
        arin = w.npcs["n-arin"]
        arin.goals[0]["targets"] = ["item:i-letter"]
        letter["last_turn"] = w.turn - 500  # 早该被扫走
        w.sweep_items()
        self.assertIn(letter, w.scenes["s-station"].items)  # 钥匙保住了
        # 未被引用的物品照常被扫走
        other = {"id": "i-junk", "name": "没人记得的旧扫帚",
                 "note": "", "last_turn": w.turn - 500}
        w.scenes["s-station"].items.append(other)
        w.sweep_items()
        self.assertNotIn(other, w.scenes["s-station"].items)

    def test_moment_fires_once_at_due_day(self):
        """既定时刻：到点必发、只发一次、有因留痕。"""
        llm, w = self._world()
        w.heartbeat = 1.0 / 6  # 6 心跳 = 1 天
        w.moments = [{"due_day": 3, "what": "潮汐窗口开启，旧堤被淹",
                      "done": False}]
        for _ in range(6):
            w.pass_time(6)
            evolution.world_pulse(llm, w)
            if w.moments[0]["done"]:
                break
        self.assertTrue(w.moments[0]["done"])
        evs = [e for e in w.events if e.kind == "world_event"
                and "潮汐窗口开启" in e.summary]
        self.assertEqual(len(evs), 1)  # 只发一次
        self.assertEqual(evs[0].cause, "既定时刻")

    def test_clock_advances_by_duration(self):
        """世界钟：时间由运动决定——事件时长累加，钟点戳留痕。"""
        llm, w = self._world()
        w.log("world_event", "他赶了半天的路", "测试",
              {"event_params": {"title": "x", "location": "s-station"}},
              duration=0.5)
        self.assertAlmostEqual(w.clock, 0.5)
        self.assertAlmostEqual(w.events[-1].day, 0.0)  # 发生时盖戳
        self.assertAlmostEqual(w.events[-1].duration, 0.5)
        self.assertEqual(w.day, 1)  # 第 0.5 天 = 第 1 天
        w.log("dialogue", "聊到深夜", "测试", duration=0.6)
        self.assertAlmostEqual(w.clock, 1.1)
        self.assertEqual(w.day, 2)      # 钟走过了一天
        self.assertEqual(w.phase, 0)    # 0.1 天 = 清晨

    def test_heartbeat_sets_world_granularity(self):
        """世界粒度：快世界一个心跳半天，慢世界一个心跳一小时。"""
        llm, w = self._world()
        w.heartbeat = 0.5  # 快世界
        w.pass_time(6)
        self.assertAlmostEqual(w.clock, 3.0)
        self.assertEqual(w.day, 4)
        llm2, w2 = self._world()
        w2.heartbeat = 0.01  # 慢世界
        w2.pass_time(6)
        self.assertAlmostEqual(w2.clock, 0.06)
        self.assertEqual(w2.day, 1)

    def test_clock_roundtrips(self):
        """时钟持久化：clock / heartbeat / 事件戳 都存得回来。"""
        llm, w = self._world()
        w.heartbeat = 0.25
        w.pass_time(4)
        w.log("world_event", "黄昏之约", "测试",
              {"event_params": {"title": "x", "location": "s-station"}},
              duration=0.25)
        data = w.to_dict()
        w2 = type(w).from_dict(data)
        self.assertAlmostEqual(w2.clock, w.clock)
        self.assertAlmostEqual(w2.heartbeat, w.heartbeat)
        self.assertAlmostEqual(w2.events[-1].day, w.events[-1].day)
        self.assertAlmostEqual(w2.events[-1].duration,
                               w.events[-1].duration)

    def test_moment_without_location_is_global(self):
        """既定时刻不默认落在玩家脚下：无地点 = 全局事件。"""
        llm, w = self._world()
        w.heartbeat = 1.0 / 6
        w.player["location"] = "s-cafe"
        w.moments = [{"due_day": 2, "what": "天文台警报响起", "done": False}]
        for _ in range(4):
            w.pass_time(6)
            evolution.world_pulse(llm, w)
            if w.moments[0]["done"]:
                break
        ev = [e for e in w.events if e.kind == "world_event"
              and "天文台" in e.summary][-1]
        params = (ev.payload or {}).get("event_params", {})
        self.assertNotIn("location", params)  # 不是 s-cafe，也不在玩家脚下
        # 可选地点：写明地点才带 location
        llm2, w2 = self._world()
        w2.heartbeat = 1.0 / 6
        w2.moments = [{"due_day": 2, "what": "钟楼的钟响了",
                       "location": "s-station", "done": False}]
        for _ in range(4):
            w2.pass_time(6)
            evolution.world_pulse(llm2, w2)
            if w2.moments[0]["done"]:
                break
        ev2 = [e for e in w2.events if e.kind == "world_event"
               and "钟楼的钟" in e.summary][-1]
        params2 = (ev2.payload or {}).get("event_params", {})
        self.assertEqual(params2.get("location"), "s-station")

    def test_worldgen_moment_location_survives_to_fire(self):
        """闭环：worldgen 落库保留地点 → 到点触发事件仍带地点。"""
        base = json.loads(MockLLM().chat("TASK:WORLDGEN", DESC))
        base["moments"] = [{"due_day": 2, "what": "钟楼的钟响了",
                            "location": "s-station"}]
        base["heartbeat"] = 1.0 / 6
        llm = ScriptedLLM([json.dumps(base, ensure_ascii=False)])
        w = generate_world(llm, "世界1", DESC)
        # 落库：location 不丢
        self.assertEqual(w.moments[0].get("location"), "s-station")
        # 到点触发：事件 payload 仍带地点
        pulse_llm = MockLLM()
        for _ in range(4):
            w.pass_time(6)
            evolution.world_pulse(pulse_llm, w)
            if w.moments[0]["done"]:
                break
        self.assertTrue(w.moments[0]["done"])
        ev = [e for e in w.events if e.kind == "world_event"
              and "钟楼的钟" in e.summary][-1]
        params = (ev.payload or {}).get("event_params", {})
        self.assertEqual(params.get("location"), "s-station")

    def test_legacy_save_rebuilds_clock_from_turn(self):
        """兼容迁移：旧档没有 clock → 按 turn/24 重建时间线。"""
        llm, w = self._world()
        w.pass_time(48)
        data = w.to_dict()
        del data["clock"]  # 模拟旧档
        w2 = type(w).from_dict(data)
        self.assertAlmostEqual(w2.clock, w2.turn / 24.0)  # 旧 day_of 关系
        self.assertEqual(w2.day, 3)  # 48 回合 = 2 天 = 第 3 天

    def test_noop_trace_same_note_skipped(self):
        """no-op 守卫：渐变与现状相同 → 不写事件、不刷新活跃时间。"""
        llm, w = self._world()
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("name") == "一封信")
        letter["note"] = "信纸湿了一角"
        w.pass_time(6)
        plan = self._plan(daily_bits=[{
            "detail": "雨滴落下", "location": "s-station",
            "intensity": 0.2,
            "trace": {"item": "i-letter", "change": "信纸湿了一角"}}])
        evolution.world_pulse(ScriptedLLM([plan]), w)
        evs = [e for e in w.events if e.kind == "daily_life"
               and "i-letter" in str((e.payload or {}).get(
                   "event_params", {}).get("item", ""))]
        self.assertEqual(len(evs), 0)  # 无变化：没有事件

    def test_noop_item_change_same_state_skipped(self):
        """no-op 守卫：跃变与现状一致 → 不写事件、不刷新活跃时间。"""
        llm, w = self._world()
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("name") == "一封信")
        letter["note"] = "完好"
        w.pass_time(6)
        before = letter["last_turn"]
        errors = apply_item_patch(w, {"op": "change", "item": "i-letter",
                                      "location": "s-station",
                                      "note": "完好"}, "重复一遍")
        self.assertEqual(errors, [])
        self.assertFalse(any(e.kind == "item_changed"
                             for e in w.events))
        self.assertEqual(letter["last_turn"], before)  # 活跃时间不刷新

    def test_goal_adjudicator_sees_laws_and_facts(self):
        """②类修复：目标裁决器读得到法则与档案（唱歌招海浪不是脑补）。"""
        llm, w = self._world()
        w.law_profile.laws = [Law(id="sing", trigger="有人唱歌",
                                  effect="海浪涌来", intensity=0.6)]
        w.facts = ["渔村建在鲸背上"]
        arin = w.npcs["n-arin"]
        arin.goals = [{"id": "g-1", "text": "唱一首歌", "progress": 0.0,
                       "targets": [], "because": ""}]
        w.pass_time(12)
        seen = []

        class SpyLLM:
            name = "spy"

            def chat(self, system, user):
                seen.append(user)
                return json.dumps({})

            def chat_json(self, system, user):
                return json.loads(self.chat(system, user))

        evolution.world_pulse(SpyLLM(), w)
        goal_payload = next(u for u in seen if "g-1" in u)
        self.assertIn("海浪涌来", goal_payload)
        self.assertIn("渔村建在鲸背上", goal_payload)

    def test_goal_payload_filters_to_npc_knowledge(self):
        """地基第 4 条：裁决载荷只含他场景里/他参与的事件。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.location = "s-cafe"  # 人在咖啡店
        w.pass_time(6)
        emit(w, "world_event", {"title": "秘密纸条", "detail": "码头见",
                                "location": "s-station", "intensity": 0.4},
             cause="测试")  # 车站的秘密：他看不见
        emit(w, "world_event", {"title": "门口的猫", "detail": "猫叼走了鱼",
                                "location": "s-cafe", "intensity": 0.3},
             cause="测试")  # 咖啡店：他看得见
        arin.goals = [{"id": "g-1", "text": "留意怪事", "progress": 0.0,
                       "targets": [], "because": "测试"}]
        seen = []

        class SpyLLM:
            name = "spy"

            def chat(self, system, user):
                seen.append(user)
                return json.dumps({"events": [], "goal_updates": {}})

            def chat_json(self, system, user):
                return json.loads(self.chat(system, user))

        evolution.propose_proactive(SpyLLM(), w, arin)
        payload = json.loads(seen[-1])
        evs = payload.get("recent_events", [])
        self.assertFalse(any("码头" in s for s in evs))  # 别处的秘密不在视野
        self.assertTrue(any("猫" in s for s in evs))     # 自己场景的在视野

    def test_all_single_npc_payloads_filter_to_knowledge(self):
        """对话、动作结局、主动开口与目标裁决共用同一知识边界。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.location = "s-cafe"
        w.player["location"] = "s-cafe"
        w.pass_time(6)
        emit(w, "world_event", {"title": "站台密信", "detail": "只约在码头见",
                                "location": "s-station", "intensity": 0.4},
             cause="测试")
        emit(w, "world_event", {"title": "门口的猫", "detail": "猫叼走了鱼",
                                "location": "s-cafe", "intensity": 0.3},
             cause="测试")
        seen = []

        class SpyLLM:
            name = "spy"

            def chat_json(self, system, user):
                seen.append(json.loads(user))
                if "TASK:DIALOGUE" in system:
                    return {"reply": "嗯。", "choices": [], "law_ids": [],
                            "relationship_delta": 0, "mood_delta": 0.0,
                            "memory": "", "memory_importance": 0.5}
                if "TASK:ACTIONRESOLVE" in system:
                    return {"outcome": "办妥了。"}
                return {"open": False}

        arin.state.last_turn = w.turn  # 跳过读取补算
        w.social[f"{arin.id}->proactive"] = w.turn  # 只检查对话裁决
        interpreter.dialogue_turn(SpyLLM(), w, arin, "发生什么了？")
        arin.state.action.text = "整理账本"
        arin.state.action.location = "s-cafe"
        arin.state.action.started_clock = w.clock - 1.0
        arin.state.action.due_clock = w.clock
        arin.state.action.progress = 1.0
        evolution.advance_action(SpyLLM(), w, arin, 1)
        w.social.pop(f"{arin.id}->player", None)
        evolution.propose_opener(SpyLLM(), w, arin)

        payloads = ([p["world"] for p in seen if "player_input" in p]
                    + [p for p in seen if "recent_events" in p])
        self.assertEqual(len(payloads), 3)
        for payload in payloads:
            events = payload["recent_events"]
            summaries = [e.get("summary", "") if isinstance(e, dict) else e
                         for e in events]
            self.assertFalse(any("码头" in s for s in summaries))
            self.assertTrue(any("猫" in s for s in summaries))

    def test_visibility_sees_departures_and_collisions(self):
        """可见性闭环：别人从自己场景离开、自己是碰撞一方都算看见。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.location = "s-cafe"
        w.pass_time(6)
        man = w.npcs["n-man"]
        man.state.location = "s-cafe"  # 两人同在咖啡店
        evolution.move_npc(w, man, "s-station", cause="测试")  # 他离开
        seen = []

        class SpyLLM:
            name = "spy"

            def chat(self, system, user):
                seen.append(user)
                return json.dumps({"events": [], "goal_updates": {}})

            def chat_json(self, system, user):
                return json.loads(self.chat(system, user))

        arin.goals = [{"id": "g-1", "text": "留意怪事", "progress": 0.0,
                       "targets": [], "because": "测试"}]
        evolution.propose_proactive(SpyLLM(), w, arin)
        evs = json.loads(seen[-1]).get("recent_events", [])
        self.assertTrue(any("来到" in s for s in evs))  # 看见他离开
        # 碰撞一方也算参与
        emit(w, "collision",
             {"a": arin.id, "b": man.id, "thing": "一封信"}, cause="测试")
        evolution.propose_proactive(SpyLLM(), w, arin)
        evs2 = json.loads(seen[-1]).get("recent_events", [])
        self.assertTrue(any("一封信" in s for s in evs2))

    def test_goal_adjudicator_sees_target_snapshot(self):
        """NPC 的目标裁决读得到被引实体的当前快照（引用 = 眼睛）。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.goals[0]["targets"] = ["item:i-letter"]
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("id") == "i-letter")
        letter["note"] = "信封被撕开过"
        cap = _CaptureLLM()
        evolution.propose_proactive(cap, w, arin)
        user = json.loads(cap.last_user)
        snaps = user.get("targets_now", [])
        self.assertTrue(snaps)
        self.assertEqual(snaps[0]["ref"], "item:i-letter")
        self.assertIn("被撕开过", snaps[0]["snapshot"])  # 她看得见现状
        self.assertIn("旧车站", snaps[0]["snapshot"])

    def test_goal_blocked_freezes_until_unblocked(self):
        """受阻态：被卡住时进度冻结，阻碍解除后才能推进。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        # 被一封信卡住（引用验真）
        goal_json = json.dumps({
            "events": [], "new_goals": [],
            "goal_updates": {"find-letter": {
                "progress": 0.9, "blocked_by": "item:i-letter",
                "blocked_note": "信被人拿走了"}}}, ensure_ascii=False)
        evolution.propose_proactive(ScriptedLLM([goal_json]), w, arin)
        goal = next(g for g in arin.goals if g["id"] == "find-letter")
        self.assertEqual(goal["blocked_by"], "item:i-letter")
        self.assertLess(float(goal["progress"]), 0.9)  # 冻结：没推进
        # 阻碍解除：blocked_by 留空 + because
        unblock = json.dumps({
            "events": [], "new_goals": [],
            "goal_updates": {"find-letter": {
                "progress": 0.9, "blocked_by": "",
                "because": "信又回到了车站"}}}, ensure_ascii=False)
        evolution.propose_proactive(ScriptedLLM([unblock]), w, arin)
        self.assertEqual(goal["blocked_by"], "")
        self.assertEqual(float(goal["progress"]), 0.9)  # 解冻后推进

    def test_goal_progress_falls_back_to_act_cause(self):
        """机械兜底：模型没写 because，但本轮真有动作 → 取动作当因。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        old = float(arin.goals[0]["progress"])
        plan = json.dumps({
            "events": [{"type": "npc_acted",
                        "params": {"npc": "n-arin",
                                   "action": "去车站打听那封信的下落",
                                   "location": "s-station"}}],
            "goal_updates": {"find-letter": {"progress": old + 0.2}},
            "new_goals": []}, ensure_ascii=False)
        summaries = evolution.propose_proactive(
            ScriptedLLM([plan]), w, arin)
        self.assertFalse(any("必须有因" in s for s in summaries))
        self.assertGreater(float(arin.goals[0]["progress"]), old)  # 推进了
        self.assertTrue(any("打听" in m.content for m in arin.memories))

    def test_goal_progress_without_cause_rejected(self):
        """有因才推进：进度提升没有 because 被驳回。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        old = float(arin.goals[0]["progress"])
        no_cause = json.dumps({
            "events": [], "new_goals": [],
            "goal_updates": {"find-letter": {"progress": old + 0.2}}},
            ensure_ascii=False)
        summaries = evolution.propose_proactive(
            ScriptedLLM([no_cause]), w, arin)
        self.assertTrue(any("必须有因" in s for s in summaries))
        self.assertEqual(float(arin.goals[0]["progress"]), old)  # 没动

    def test_actionresolve_payload_includes_item_context(self):
        """结局裁决拿得到现场物品与目标快照——patch 不用靠猜。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(text="去车站取那封信",
                                        location="s-station", progress=0.99)
        arin.goals[0]["targets"] = ["item:i-letter"]
        seen = []

        class SpyLLM:
            name = "spy"

            def chat(self, system, user):
                seen.append(user)
                return json.dumps({"outcome": "做完了。"})

            def chat_json(self, system, user):
                return json.loads(self.chat(system, user))

        evolution.advance_action(SpyLLM(), w, arin, 6)
        payload = json.loads(seen[-1])
        ids = [i.get("id") for i in payload.get("scene_items", [])]
        self.assertIn("i-letter", ids)  # 现场物品表给了真实 id
        self.assertIn("i-letter", payload.get("used_item_ids", []))
        refs = [t.get("ref") for t in payload.get("targets_now", [])]
        self.assertIn("item:i-letter", refs)  # 目标物品快照也在

    def test_action_outcome_patch_lands_in_state(self):
        """结局补丁：叙事说发生的事，patch 让它真发生（有因、验引用）。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(text="去车站取那封信",
                                        location="s-station", progress=0.99)
        w.pass_time(6)
        plan = json.dumps({"outcome": "他取走了信，拆开了封蜡。",
                           "patch": {"op": "change", "item": "i-letter",
                                     "location": "s-station",
                                     "note": "被拆开了"}},
                          ensure_ascii=False)
        summaries = evolution.advance_action(ScriptedLLM([plan]), w, arin, 6)
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("id") == "i-letter")
        self.assertIn("被拆开了", letter["note"])  # 状态真的变了
        ev = [e for e in w.events if e.kind == "item_changed"
              and "信" in e.summary][-1]
        self.assertTrue(ev.cause.startswith("动作结局"))  # 有因
        self.assertTrue(any("被拆开了" in s for s in summaries))

    def test_action_outcome_taking_item_sets_holder(self):
        """结局说拿走，补丁必须把物品的归属也落进状态。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(text="去车站取那封信",
                                        location="s-station", progress=0.99)
        w.pass_time(6)
        plan = json.dumps({"outcome": "阿凛把信放进口袋。",
                           "patch": {"op": "change", "item": "i-letter",
                                     "location": "s-station",
                                     "held_by": "npc:n-arin"}},
                          ensure_ascii=False)
        evolution.advance_action(ScriptedLLM([plan]), w, arin, 6)
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("id") == "i-letter")
        self.assertEqual(letter["held_by"], "npc:n-arin")
        self.assertNotIn("i-letter", [i["id"] for i in active_items(
            w.scenes["s-station"])])
        self.assertTrue(any(e.kind == "item_transfer" for e in w.events))

    def test_action_outcome_can_materialize_held_item(self):
        """角色携带、首次显现的持续物品也必须进入物态账本。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(text="取出铜钥匙查看", location="s-station",
                                        progress=0.99)
        w.pass_time(6)
        plan = json.dumps({"outcome": "阿凛取出铜钥匙后又收进口袋。",
                           "patch": {"op": "add", "item": "i-copper-key",
                                     "name": "铜钥匙", "location": "s-station",
                                     "note": "锈迹斑斑，刚从阿凛口袋里取出",
                                     "held_by": "npc:n-arin"}},
                          ensure_ascii=False)
        evolution.advance_action(ScriptedLLM([plan]), w, arin, 6)
        key = next(i for i in w.scenes["s-station"].items
                   if i.get("id") == "i-copper-key")
        self.assertEqual(key["held_by"], "npc:n-arin")
        self.assertTrue(any(e.kind == "item_added" for e in w.events))
        self.assertTrue(any(e.kind == "item_transfer" for e in w.events))

    def test_action_outcome_new_scene_item_needs_explicit_holder(self):
        """行动结局新增持久物时，不能把归属含混地留在叙事里。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(text="整理站台遗物", location="s-station",
                                        progress=0.99)
        w.pass_time(6)
        plan = json.dumps({"outcome": "他在长椅下找到一枚旧徽章。",
                           "patch": {"op": "add", "item": "i-badge",
                                     "name": "旧徽章", "location": "s-station"}},
                          ensure_ascii=False)
        summaries = evolution.advance_action(ScriptedLLM([plan]), w, arin, 6)
        self.assertTrue(any("held_by" in s for s in summaries))
        self.assertFalse(any(i.get("id") == "i-badge"
                             for i in w.scenes["s-station"].items))

    def test_action_outcome_new_scene_item_has_no_fake_transfer(self):
        """新物品明确留在现场，只记诞生，不伪造一次放回。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(text="整理站台遗物", location="s-station",
                                        progress=0.99)
        w.pass_time(6)
        plan = json.dumps({"outcome": "他把找到的旧徽章留在长椅上。",
                           "patch": {"op": "add", "item": "i-badge",
                                     "name": "旧徽章", "location": "s-station",
                                     "held_by": ""}}, ensure_ascii=False)
        evolution.advance_action(ScriptedLLM([plan]), w, arin, 6)
        badge = next(i for i in w.scenes["s-station"].items
                     if i.get("id") == "i-badge")
        self.assertFalse(badge.get("held_by"))
        added = [e for e in w.events if e.kind == "item_added"
                 and "徽章" in e.summary]
        transferred = [e for e in w.events if e.kind == "item_transfer"
                       and "徽章" in e.summary]
        self.assertTrue(added)
        self.assertFalse(transferred)

    def test_action_outcome_patch_bad_ref_rejected(self):
        """结局补丁引用不存在的物品 → 驳回，动作完成照常入账。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(text="去取信",
                                        location="s-station", progress=0.99)
        w.pass_time(6)
        plan = json.dumps({"outcome": "做完了。",
                           "patch": {"op": "change", "item": "i-ghost",
                                     "location": "s-station",
                                     "note": "x"}}, ensure_ascii=False)
        summaries = evolution.advance_action(ScriptedLLM([plan]), w, arin, 6)
        self.assertTrue(any("驳回结局补丁" in s for s in summaries))
        self.assertTrue(any(e.kind == "action_done" for e in w.events))
        self.assertFalse(any(e.kind == "item_changed" for e in w.events))

    def test_npc_links_grow_with_interaction(self):
        """NPC↔NPC 关系账本：搭话让两人的链接向正方向移动。"""
        llm, w = self._world()
        arin, zhou = w.npcs["n-arin"], w.npcs["n-zhou"]
        evolution.move_npc(w, zhou, "s-cafe", cause="测试")
        for _ in range(3):
            evolution.heartbeat(llm, w, arin)
        self.assertGreater(float(arin.links.get("n-zhou", 0.0)), 0)
        self.assertGreater(float(zhou.links.get("n-arin", 0.0)), 0)
        self.assertLessEqual(abs(arin.links["n-zhou"]), 1.0)  # 有界

    def test_interaction_writes_both_parties_memory(self):
        """对话是双方的事实：听者也留下「对我说」的自己版本。"""
        llm, w = self._world()
        arin, zhou = w.npcs["n-arin"], w.npcs["n-zhou"]
        evolution.move_npc(w, zhou, "s-cafe", cause="测试")
        for _ in range(6):
            evolution.heartbeat(llm, w, arin)
        said = [m for m in arin.memories if "对 老周 说" in m.content]
        heard = [m for m in zhou.memories if "对我说" in m.content
                 and "阿凛" in m.content]
        if said:  # 只要说者记了，听者就必须也记了
            self.assertTrue(heard)
        ev = [e for e in w.events if e.kind == "npc_interaction"]
        if ev:
            loc = (ev[-1].payload or {}).get("event_params", {}).get(
                "location", "")
            self.assertEqual(loc, arin.state.location)  # 带地点

    def test_interaction_visible_to_same_scene_bystanders(self):
        """搭话带地点：同场景的人看得见，别处的人看不见。"""
        from worldledger.store import Event
        llm, w = self._world()
        arin, zhou, man = w.npcs["n-arin"], w.npcs["n-zhou"], w.npcs["n-man"]
        arin.state.location = "s-cafe"
        zhou.state.location = "s-cafe"     # 旁观者：同场景
        man.state.location = "s-station"   # 别处
        ev = Event(turn=w.turn + 1, kind="npc_interaction",
                   summary="x 对 y 说", cause="测试",
                   payload={"event_params": {"npc": "n-arin",
                                             "target": "n-x",
                                             "line": "……",
                                             "location": "s-cafe"}})
        self.assertTrue(evolution._npc_visible(w, zhou, ev))   # 同场景看见
        self.assertFalse(evolution._npc_visible(w, man, ev))   # 别处看不见

    def test_dialogue_reply_visible_to_same_scene_bystanders(self):
        """现场对话带地点：在场者能从账本看到，不在场者不能。"""
        from worldledger.store import Event
        llm, w = self._world()
        arin, zhou, man = w.npcs["n-arin"], w.npcs["n-zhou"], w.npcs["n-man"]
        arin.state.location = "s-cafe"
        zhou.state.location = "s-cafe"
        man.state.location = "s-station"
        ev = Event(turn=w.turn + 1, kind="dialogue", summary="阿凛：……",
                   cause="测试", payload={"event_params": {
                       "npc": "n-arin", "reply": "……", "location": "s-cafe"}})
        self.assertTrue(evolution._npc_visible(w, zhou, ev))
        self.assertFalse(evolution._npc_visible(w, man, ev))

    def test_player_said_enters_ledger(self):
        """玩家的话进账本：世界里有「你」。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.player["location"] = "s-cafe"
        interpreter.dialogue_turn(llm, w, arin, "你在等谁？")
        ev = [e for e in w.events if e.kind == "player_said"]
        self.assertTrue(ev)
        self.assertIn("你在等谁", ev[-1].summary)
        self.assertEqual(ev[-1].cause, "玩家对话")
        self.assertEqual((ev[-1].payload or {}).get("event_params", {}).get(
            "location"), w.player["location"])

    def test_goal_targets_persist_collide_once(self):
        """真实路径：targets 落库 → 碰撞只入账一次（心跳不刷屏）。"""
        llm, w = self._world()
        arin, man = w.npcs["n-arin"], w.npcs["n-man"]
        evolution.move_npc(w, man, "s-cafe", cause="测试")
        w.pass_time(12)
        plan = self._plan(npc_plans=[
            {"npc": "n-arin",
             "state": {"activity": "翻账本", "mood": "平静",
                       "location": "s-cafe"},
             "action": None, "interaction": None, "goal_updates": {},
             "new_goals": [{"id": "g-new", "text": "找到那封信",
                            "progress": 0.0, "targets": ["一封信"]}]},
            {"npc": "n-man",
             "state": {"activity": "等信", "mood": "平静",
                       "location": "s-cafe"},
             "action": None, "interaction": None, "goal_updates": {},
             "new_goals": [{"id": "g-new2", "text": "等自己的信",
                            "progress": 0.0, "targets": ["一封信"]}]}])
        evolution.world_pulse(ScriptedLLM([plan]), w)
        g1 = next(g for g in arin.goals if g["id"] == "g-new")
        self.assertEqual(g1["targets"], ["item:i-letter"])  # 写入即规范化
        self.assertNotIn("collision", [e.kind for e in w.events])
        # 下一次脉冲：碰撞浮出（引用首次形成）
        w.pass_time(6)
        evolution.world_pulse(ScriptedLLM([self._plan()]), w)
        self.assertEqual([e.kind for e in w.events].count("collision"), 1)
        # 再来两次：状态没变 → 不重复刷屏
        for _ in range(2):
            w.pass_time(6)
            evolution.world_pulse(ScriptedLLM([self._plan()]), w)
        self.assertEqual([e.kind for e in w.events].count("collision"), 1)
        # 状态变化（信被移动/变化）→ 新碰撞
        letter = next(i for i in w.scenes["s-station"].items
                      if i.get("id") == "i-letter")
        letter["note"] = "信封被撕开过"
        w.pass_time(6)
        evolution.world_pulse(ScriptedLLM([self._plan()]), w)
        self.assertEqual([e.kind for e in w.events].count("collision"), 2)

    def test_collision_refires_after_release_and_return(self):
        """解除后重现：引用消失 → 记录清除；再引用 → 新碰撞。"""
        llm, w = self._world()
        arin, man = w.npcs["n-arin"], w.npcs["n-man"]
        evolution.move_npc(w, man, "s-cafe", cause="测试")
        arin.goals[0]["targets"] = ["一封信"]
        man.goals[0]["targets"] = ["item:i-letter"]  # 裸名与 id 引用同实体
        w.pass_time(12)
        evolution.world_pulse(ScriptedLLM([self._plan()]), w)
        self.assertEqual([e.kind for e in w.events].count("collision"), 1)
        # 解除：man 不再引用
        man.goals[0]["targets"] = []
        w.pass_time(6)
        evolution.world_pulse(ScriptedLLM([self._plan()]), w)
        # 重现：man 又盯上那封信
        man.goals[0]["targets"] = ["一封信"]
        w.pass_time(6)
        evolution.world_pulse(ScriptedLLM([self._plan()]), w)
        self.assertEqual([e.kind for e in w.events].count("collision"), 2)

    def test_same_world_diverges_with_different_behavior(self):
        llm, w = self._world()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "u.json"
            u = Universe(worlds={w.name: w}, current=w.name)
            u.save(p)
            w2 = Universe.load(p).here
        interpreter.change_law(llm, w, "这座城的人不再撒谎")  # 只有 w 天变
        for world in (w, w2):
            for _ in range(4):
                world.pass_time(6)
                evolution.world_pulse(llm, world)
        self.assertGreater(w.law_profile.version, w2.law_profile.version)
        self.assertIn("world_react", [e.kind for e in w.events])
        self.assertNotIn("world_react", [e.kind for e in w2.events])
        self.assertNotEqual(
            [e.summary for e in w.events[-10:]],
            [e.summary for e in w2.events[-10:]])
        for world in (w, w2):  # 分叉不破坏不变量
            for npc in world.npcs.values():
                self.assertIn(npc.state.location, world.scenes)


class _BoomLLM:
    """测试用：chat 一律抛异常，模拟 API 故障。"""

    def chat(self, system: str, user: str) -> str:
        raise ConnectionError("API 炸了")

    def chat_json(self, system: str, user: str, attempts: int = 2) -> dict:
        raise ConnectionError("API 炸了")


class _RecordingLLM:
    """测试用：包装 MockLLM，记录最后一次载荷（量成本代理）。"""

    def __init__(self, inner):
        self.inner = inner
        self.last_payload = ""

    def chat(self, system: str, user: str) -> str:
        return self.inner.chat(system, user)

    def chat_json(self, system: str, user: str, attempts: int = 2) -> dict:
        self.last_payload = user
        return self.inner.chat_json(system, user, attempts)


class TestHardening(unittest.TestCase):
    """审查修复钉住：心跳不丢、原因必填、世界结构校验、路径安全。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        return llm, w

    def test_pulse_llm_failure_does_not_consume_heartbeat(self):
        llm, w = self._world()
        w.pass_time(6)
        before = w.pulse_last_turn
        with self.assertRaises(ConnectionError):
            evolution.world_pulse(_BoomLLM(), w)
        self.assertEqual(w.pulse_last_turn, before)  # 心跳没被吞掉

    def test_pulse_does_not_overwrite_travel_with_new_action(self):
        """真实故障链：脉冲中的新行动不能覆盖尚未抵达的跨场景行动。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.state.action = ActionState(text="去旧车站找线索",
                                        location="s-station", progress=0.5)
        w.pass_time(6)
        before = len(w.events)
        plan = json.dumps({
            "events": [], "influences": [], "daily_bits": [],
            "npc_plans": [{"npc": arin.id,
                           "state": {"activity": "赶路", "mood": "专注",
                                     "location": arin.state.location},
                           "action": {"type": "npc_acted", "params": {
                               "npc": arin.id,
                               "action": "推开旧车站的门，向老周说明线索",
                               "location": "s-station"}}}],
            "new_scenes": [], "item_patches": []}, ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertEqual(arin.state.location, "s-cafe")
        self.assertEqual(arin.state.action.text, "去旧车站找线索")
        self.assertFalse(any(e.kind in ("npc_acted", "action_aborted")
                             for e in w.events[before:]))
        self.assertTrue(any("暂缓" in s for s in summaries))

    def test_emit_requires_nonempty_cause(self):
        llm, w = self._world()
        with self.assertRaises(ValueError):
            emit(w, "daily_life",
                 {"detail": "某件事", "location": "s-station",
                  "intensity": 0.2}, cause="")

    def test_worldgen_structural_validation_catches_bad_refs(self):
        bad = {
            "atmosphere": "雨·永续",
            "laws": [{"trigger": "有人撒谎", "effect": "漏出真心话"}],
            "scenes": [
                {"id": "s-a", "name": "甲地", "description": "x",
                 "exits": ["s-b"], "npcs": ["n-1"], "hint": "",
                 "items": [{"name": "伞"}]},
            ],
            "npcs": [{"id": "n-1", "name": "某人",
                      "persona": "人设", "goals": []}],
        }
        with self.assertRaises(worldgen.WorldGenError):
            worldgen._build_world("世界", DESC, bad)  # 出口 s-b 不存在

    def test_worldgen_validation_catches_isolated_scene(self):
        bad = {
            "atmosphere": "雨·永续",
            "laws": [{"trigger": "有人撒谎", "effect": "漏出真心话"}],
            "scenes": [
                {"id": "s-a", "name": "甲地", "description": "x",
                 "exits": [], "npcs": ["n-1"], "hint": "", "items": []},
                {"id": "s-b", "name": "乙地", "description": "x",
                 "exits": [], "npcs": [], "hint": "", "items": []},
            ],
            "npcs": [{"id": "n-1", "name": "某人",
                      "persona": "人设", "goals": []}],
        }
        with self.assertRaises(worldgen.WorldGenError) as ctx:
            worldgen._build_world("世界", DESC, bad)
        self.assertIn("孤岛", str(ctx.exception))

    def test_worldgen_birth_alignment_is_explicit(self):
        """同场景陌生人不自动相识；初始经历与关系只保留显式输入。"""
        base = {
            "atmosphere": "平静", "laws": [],
            "scenes": [{"id": "s-cafe", "name": "咖啡店",
                        "description": "店内", "npcs": ["n-a", "n-b"],
                        "exits": [], "items": []}],
            "npcs": [
                {"id": "n-a", "name": "路人甲", "persona": "刚进店避雨",
                 "goals": [], "memories": []},
                {"id": "n-b", "name": "路人乙", "persona": "第一次来这座城",
                 "goals": [], "memories": []},
            ],
            "facts": [], "moments": [], "heartbeat": 1.0 / 24.0,
        }
        strangers = worldgen._build_world("陌生人", DESC, base)
        self.assertEqual(strangers.npcs["n-a"].links, {})
        self.assertEqual(strangers.npcs["n-b"].links, {})
        self.assertEqual(strangers.npcs["n-a"].memories, [])

        base["npcs"][0]["links"] = {"n-b": 0.4}
        base["npcs"][0]["memories"] = [{
            "content": "我和路人乙一起跑船多年。",
            "projection": {"age_days": None, "scene": None,
                           "items": [], "current_states": []}}]
        aligned = worldgen._build_world("旧友", DESC, base)
        self.assertEqual(aligned.npcs["n-a"].links, {"n-b": 0.4})
        self.assertEqual(aligned.npcs["n-b"].links, {})
        self.assertGreater(aligned.npcs["n-a"].memories[0].turn, 0)
        self.assertTrue(any(e.kind == "memory_projected"
                            for e in aligned.events))

    def test_worldgen_rejects_periodic_law_without_clock_commitment(self):
        """周期规律不能只存在于散文；必须有世界钟可执行的时刻锚。"""
        base = {
            "atmosphere": "梦·交错", "laws": [],
            "scenes": [{"id": "s-a", "name": "甲地", "description": "x",
                        "npcs": ["n-a", "n-b"], "exits": [], "items": []}],
            "npcs": [
                {"id": "n-a", "name": "甲", "persona": "人设", "goals": []},
                {"id": "n-b", "name": "乙", "persona": "人设", "goals": []},
            ],
            "facts": ["两人的行动归属会变化"], "moments": [],
            "heartbeat": 1.0 / 24.0,
        }
        with self.assertRaises(worldgen.WorldGenError) as ctx:
            worldgen._build_world("世界", "两人隔天交换身体", base)
        self.assertIn("repeat_days", str(ctx.exception))

        base["moments"] = [{
            "due_day": 1, "repeat_days": 2,
            "what": "两人的行动归属在醒来时发生变化",
        }]
        with self.assertRaises(worldgen.WorldGenError) as ctx:
            worldgen._build_world("世界", "两人隔天交换身体", base)
        self.assertIn("agency_patches", str(ctx.exception))

    def test_worldgen_preserves_periodic_clock_commitment(self):
        """结构化周期时刻进入存档，运行时才有机会兑现它。"""
        base = {
            "atmosphere": "梦·交错", "laws": [],
            "scenes": [{"id": "s-a", "name": "甲地", "description": "x",
                        "npcs": ["n-a", "n-b"], "exits": [], "items": []}],
            "npcs": [
                {"id": "n-a", "name": "甲", "persona": "人设", "goals": []},
                {"id": "n-b", "name": "乙", "persona": "人设", "goals": []},
            ],
            "facts": ["两人的行动归属会变化"],
            "moments": [{
                "due_day": 1, "repeat_days": 2,
                "what": "两人的行动归属在醒来时发生变化",
                "agency_patches": [{
                    "body": "n-a", "actor": "n-b", "duration_days": 1,
                    "why": "周期性世界承诺",
                }, {
                    "body": "n-b", "actor": "n-a", "duration_days": 1,
                    "why": "周期性世界承诺",
                }],
            }],
            "heartbeat": 1.0 / 24.0,
        }
        world = worldgen._build_world(
            "世界", "两人隔天交换身体", base)
        self.assertEqual(world.moments[0]["repeat_days"], 2.0)
        self.assertEqual(len(world.moments[0]["agency_patches"]), 2)

    def test_worldgen_requires_named_moment_npcs_in_refs(self):
        """具名时刻必须把角色写入 refs，不能依赖散文绕过事件校验。"""
        base = {
            "atmosphere": "黄昏", "laws": [],
            "scenes": [{"id": "s-a", "name": "旧站台", "description": "x",
                        "npcs": ["n-a", "n-b"], "exits": [], "items": []}],
            "npcs": [
                {"id": "n-a", "name": "甲", "persona": "人设", "goals": []},
                {"id": "n-b", "name": "乙", "persona": "人设", "goals": []},
            ],
            "facts": [], "heartbeat": 1.0 / 24.0,
            "moments": [{"due_day": 1, "what": "甲与乙在黄昏相见"}],
        }
        with self.assertRaises(worldgen.WorldGenError) as ctx:
            worldgen._build_world("世界", DESC, base)
        self.assertIn("refs", str(ctx.exception))

        base["moments"][0]["refs"] = ["npc:n-a", "npc:n-b", "scene:s-a"]
        world = worldgen._build_world("世界", DESC, base)
        self.assertEqual(world.moments[0]["refs"],
                         ["npc:n-a", "npc:n-b", "scene:s-a"])

    def test_worldgen_rejects_invalid_moment_clock_and_name_binding(self):
        """既定时刻不能接受非法日期或与文字不一致的身体引用。"""
        base = {
            "atmosphere": "梦·交错", "laws": [],
            "scenes": [{"id": "s-a", "name": "甲地", "description": "x",
                        "npcs": ["n-c", "n-b"], "exits": [],
                        "items": []}],
            "npcs": [
                {"id": "n-c", "name": "角色丙",
                 "persona": "人设", "goals": []},
                {"id": "n-b", "name": "角色乙",
                 "persona": "人设", "goals": []},
            ],
            "facts": [], "moments": [], "heartbeat": 1.0 / 24.0,
        }
        bad_clock = json.loads(json.dumps(base))
        bad_clock["moments"] = [{"due_day": 0, "what": "开始",
                                  "agency_patches": []}]
        with self.assertRaisesRegex(worldgen.WorldGenError, "due_day"):
            worldgen._build_world("世界", DESC, bad_clock)

        bad_binding = json.loads(json.dumps(base))
        bad_binding["moments"] = [{
            "due_day": 1, "what": "第一次互换",
            "agency_patches": [{
                 "body": "n-c", "actor": "n-b",
                 "duration_days": 1, "why": "角色乙通过角色甲的身体行动",
            }],
        }]
        with self.assertRaisesRegex(worldgen.WorldGenError, "身体文字与引用"):
            worldgen._build_world("世界", DESC, bad_binding)

        missing_restore = json.loads(json.dumps(base))
        missing_restore["scenes"][0]["npcs"] = ["n-c", "n-b"]
        missing_restore["moments"] = [{
            "due_day": 1, "what": "第一次互换",
            "agency_patches": [{
                 "body": "n-b", "actor": "n-c",
                "duration_days": 1, "why": "行动归属交换",
            }, {
                 "body": "n-c", "actor": "n-b",
                "duration_days": 1, "why": "行动归属交换",
            }],
        }, {
            "due_day": 2, "what": "互换结束，恢复行动主体",
            "agency_patches": [],
        }]
        with self.assertRaisesRegex(worldgen.WorldGenError, "归位/恢复 moment"):
            worldgen._build_world("世界", "两个角色交换身体", missing_restore)

        valid = json.loads(json.dumps(base))
        valid["scenes"][0]["npcs"] = ["n-a", "n-b"]
        valid["npcs"][0] = {"id": "n-a", "name": "角色甲",
                              "persona": "人设", "goals": []}
        valid["moments"] = [{
            "due_day": 1, "what": "第一次互换",
            "agency_patches": [{
                 "body": "n-a", "actor": "n-b",
                 "duration_days": 1, "why": "角色乙通过角色甲的身体行动",
            }],
        }]
        worldgen._build_world("世界", DESC, valid)

    def test_worldgen_rejects_invalid_birth_references(self):
        """关系和既定时刻的悬空引用不能静默改义后落库。"""
        base = {
            "atmosphere": "平静", "laws": [],
            "scenes": [{"id": "s-a", "name": "甲地", "description": "x",
                        "npcs": ["n-a", "n-b"], "exits": [], "items": []}],
            "npcs": [
                {"id": "n-a", "name": "甲", "persona": "人设", "goals": []},
                {"id": "n-b", "name": "乙", "persona": "人设", "goals": []},
            ],
            "facts": [], "moments": [], "heartbeat": 1.0 / 24.0,
        }
        for links in ({"n-ghost": 0.2}, {"n-a": 0.2}, {"n-b": 1.2}):
            bad = json.loads(json.dumps(base))
            bad["npcs"][0]["links"] = links
            with self.subTest(links=links), self.assertRaises(worldgen.WorldGenError):
                worldgen._build_world("世界", DESC, bad)
        bad_moment = json.loads(json.dumps(base))
        bad_moment["moments"] = [{"due_day": 2, "what": "钟响",
                                   "location": "s-ghost"}]
        with self.assertRaises(worldgen.WorldGenError):
            worldgen._build_world("世界", DESC, bad_moment)

    def test_worldgen_validation_catches_npc_in_two_scenes(self):
        bad = {
            "atmosphere": "雨·永续",
            "laws": [{"trigger": "有人撒谎", "effect": "漏出真心话"}],
            "scenes": [
                {"id": "s-a", "name": "甲地", "description": "x",
                 "exits": ["s-b"], "npcs": ["n-1"], "hint": "", "items": []},
                {"id": "s-b", "name": "乙地", "description": "x",
                 "exits": ["s-a"], "npcs": ["n-1"], "hint": "", "items": []},
            ],
            "npcs": [{"id": "n-1", "name": "某人",
                      "persona": "人设", "goals": []}],
        }
        # validate_world 严格报告问题……
        w = World(name="t", description=DESC, law_profile=LawProfile(
            expectation=DESC, atmosphere="x", laws=[]),
            scenes={s["id"]: Scene.from_dict(s) for s in bad["scenes"]},
            npcs={n["id"]: NPC.from_dict(n) for n in bad["npcs"]})
        problems = worldgen.validate_world(w)
        self.assertTrue(any("同时属于" in p for p in problems))
        # 生成器不替世界选择归属：重复位置必须拒绝，而不是保留第一个。
        with self.assertRaises(worldgen.WorldGenError):
            worldgen._build_world("世界", DESC, bad)

    def test_load_rejects_structurally_invalid_save(self):
        """读档只验证；旧档矛盾不能被静默带回引擎继续运行。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.scenes["s-station"].npcs.append(arin.id)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            Universe(worlds={w.name: w}, current=w.name).save(path)
            with self.assertRaisesRegex(ValueError, "结构校验失败"):
                Universe.load(path)

    def test_worldgen_structural_validation_catches_duplicate_ids(self):
        bad = {
            "atmosphere": "雨·永续",
            "laws": [{"trigger": "有人撒谎", "effect": "漏出真心话"}],
            "scenes": [
                {"id": "s-a", "name": "甲地", "description": "x",
                 "exits": [], "npcs": ["n-1"], "hint": "",
                 "items": []},
                {"id": "s-a", "name": "乙地", "description": "x",
                 "exits": [], "npcs": [], "hint": "", "items": []},
            ],
            "npcs": [{"id": "n-1", "name": "某人",
                      "persona": "人设", "goals": []}],
        }
        with self.assertRaises(worldgen.WorldGenError):
            worldgen._build_world("世界", DESC, bad)

    def test_npc_name_path_traversal_rejected(self):
        llm, w = self._world()
        errors = cards.create_npc(w, "..", "试图逃出目录的 NPC")
        self.assertTrue(errors)
        errors = cards.create_npc(w, "a/b", "带斜杠的 NPC")
        self.assertTrue(errors)
        self.assertNotIn("..", w.npcs)
        self.assertNotIn("a/b", w.npcs)

    def test_export_npc_sanitizes_filename(self):
        llm, w = self._world()
        bad = NPC(id="n-x", name="../邪名", persona="人设")
        p2 = cards.export_npc(bad)  # 默认路径：名字净化后落盘
        self.assertNotIn("..", p2.name)
        self.assertEqual(p2.name, "邪名.json")
        p2.unlink()

    def test_npc_fades_into_fog_and_releases_heartbeat(self):
        """退入雾中：不在场、不占心跳、名录可查（不是删除）。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.remember(arin, "我记得那封信的事。", cause="测试", kind="npc_memory")
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [], "influences": [],
                           "daily_bits": [],
                           "fade_npcs": [{"npc": "n-arin",
                                          "where": "南方的城",
                                          "why": "攒够了钱，去寻那封信的寄信人"}],
                           "new_npcs": [], "new_scenes": [],
                           "item_patches": [], "crowds": []},
                          ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertTrue(arin.in_fog)
        self.assertNotIn(arin.id, w.scenes["s-cafe"].npcs)  # 不在场
        self.assertTrue(any(e.kind == "npc_faded" for e in w.events))
        stamp = arin.state.last_turn
        w.pass_time(12)
        evolution.world_pulse(llm, w)
        self.assertEqual(arin.state.last_turn, stamp)  # 不占心跳
        self.assertTrue(any("那封信" in m.content
                            for m in arin.memories))  # 记忆保留（不删）

    def test_fade_without_why_rejected(self):
        llm, w = self._world()
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [], "influences": [],
                           "daily_bits": [],
                           "fade_npcs": [{"npc": "n-arin", "why": ""}],
                           "new_npcs": [], "new_scenes": [],
                           "item_patches": [], "crowds": []},
                          ensure_ascii=False)
        summaries = evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertTrue(any("原因必填" in s for s in summaries))
        self.assertFalse(w.npcs["n-arin"].in_fog)

    def test_npc_returns_from_fog(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.in_fog = True
        arin.fog_note = "出海。为了找那封信的寄信人"
        for scene in w.scenes.values():
            if arin.id in scene.npcs:
                scene.npcs.remove(arin.id)
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [], "influences": [],
                           "daily_bits": [],
                           "return_npcs": [{"npc": "n-arin",
                                            "why": "听说雨快停了，回来看看",
                                            "to": "s-cafe"}],
                           "new_npcs": [], "new_scenes": [],
                           "item_patches": [], "crowds": []},
                          ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertFalse(arin.in_fog)
        self.assertIn(arin.id, w.scenes["s-cafe"].npcs)
        self.assertTrue(any(e.kind == "npc_returned" for e in w.events))

    def test_fog_npc_survives_roundtrip(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        arin.in_fog = True
        arin.fog_note = "被雨吞掉。为了找那封信"
        u = Universe(worlds={w.name: w}, current=w.name)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "u.json"
            u.save(p)
            u2 = Universe.load(p)
        a2 = u2.here.npcs["n-arin"]
        self.assertTrue(a2.in_fog)
        self.assertEqual(a2.fog_note, "被雨吞掉。为了找那封信")

    def test_pulse_plan_new_goal_logs_goal_emerged(self):
        """P1-1：脉冲目标完成路径的新目标也要入账（出生与死亡都有痕）。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        ensure_scene(llm, w, "s-alley")  # 生成后：距咖啡店 1 跳，6 回合到期
        arin.state.location = "s-alley"
        w.player["location"] = "s-cafe"
        w.pass_time(6)
        plan = json.dumps({
            "events": [], "influences": [], "daily_bits": [],
            "npc_plans": [{"npc": "n-arin",
                           "state": {"activity": "想事情",
                                     "mood": "平静",
                                     "location": "s-alley"},
                           "new_goals": [{"id": "g-new",
                                          "text": "找到那封信的寄信人",
                                          "progress": 0.0}]}],
            "new_scenes": [], "item_patches": []}, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertTrue(any(e.kind == "goal_emerged" for e in w.events))

    def test_pulse_plan_generic_action_executes(self):
        """P1-2：脉冲计划的 action 是任意已注册事件（note_left 不再被吞）。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        ensure_scene(llm, w, "s-alley")  # 生成后：距咖啡店 1 跳，6 回合到期
        arin.state.location = "s-alley"
        w.player["location"] = "s-cafe"
        w.pass_time(6)
        plan = json.dumps({
            "events": [], "influences": [], "daily_bits": [],
            "npc_plans": [{"npc": "n-arin",
                           "state": {"activity": "写字条",
                                     "mood": "平静",
                                     "location": "s-alley"},
                           "action": {"type": "note_left",
                                      "params": {
                                          "npc": "n-arin",
                                          "location": "s-alley",
                                          "content": "今晚别走，等我。"}}}],
            "new_scenes": [], "item_patches": []}, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertTrue(any(e.kind == "note_left" for e in w.events))

    def test_emergent_npc_keeps_goal_after_dialogue(self):
        """互动不会重置人物已有目标。"""
        llm, w = self._world()
        w.player["location"] = "s-cafe"
        errors, hid = cards.emerge_npc(
            w, "货郎", "挑担路过的货郎。",
            {"id": "g-1", "text": "卖完这担针线", "progress": 0.0},
            "s-cafe", "集市日，货郎经过")
        self.assertEqual(errors, [])
        huo = w.npcs[hid]
        self.assertEqual(huo.goals[0]["text"], "卖完这担针线")  # 目标潜伏
        interpreter.dialogue_turn(llm, w, huo, "你卖什么？")
        self.assertEqual(huo.goals[0]["text"], "卖完这担针线")

    def test_duplicate_npc_name_rejected(self):
        llm, w = self._world()
        self.assertTrue(cards.create_npc(w, "阿凛", "假阿凛"))
        errors, _ = cards.emerge_npc(w, "阿凛", "另一个阿凛。", None,
                                     "s-cafe", "测试")
        self.assertTrue(errors)

    def test_character_action_lands_on_new_npc_by_id(self):
        """B1：action 按 id 定位，不按名字（同名已禁，但身份必须按 id）。"""
        llm, w = self._world()
        proposal = {"name": "修表匠·禾",
                    "persona": "蹲在钟楼下修表的人。",
                    "goal": {"id": "g-1", "text": "修好怀表", "progress": 0.0},
                    "location": "s-cafe", "reason": "钟楼的表停了",
                    "activity": "用镊子检查一块停摆的怀表"}
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [], "influences": [],
                           "daily_bits": [], "new_npcs": [proposal],
                           "new_scenes": [], "item_patches": [],
                           "crowds": []}, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        he = next(n for n in w.npcs.values() if n.name == "修表匠·禾")
        self.assertTrue(he.state.activity.startswith("用镊子"))
        self.assertEqual(he.state.action.text, "")
        others = [n for n in w.npcs.values() if n.id != he.id]
        self.assertTrue(all(not n.state.action.text for n in others))

    def test_social_stays_bounded_and_fog_cleans_keys(self):
        """social 回收：退雾清键，载荷只带活跃 NPC 相关键 + 全局键。"""
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        # 造一些键：互动冷却、开口、主动
        world = w
        world.social["n-arin->n-zhou"] = world.turn
        world.social["n-zhou->n-arin"] = world.turn
        world.social["n-arin->player"] = world.turn
        world.social["n-arin->proactive"] = world.turn
        world.social["new-npc"] = world.turn  # 全局键
        self.assertIn("n-arin->n-zhou", evolution.active_social(world))
        self.assertIn("new-npc", evolution.active_social(world))
        # 退雾 → 键被清
        w.pass_time(6)
        plan = json.dumps({"events": [], "npc_plans": [], "influences": [],
                           "daily_bits": [],
                           "fade_npcs": [{"npc": "n-arin",
                                          "why": "出城了"}],
                           "new_npcs": [], "new_scenes": [],
                           "item_patches": [], "crowds": []},
                          ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertNotIn("n-arin->n-zhou", world.social)
        self.assertNotIn("n-zhou->n-arin", world.social)
        self.assertNotIn("n-arin->player", world.social)
        self.assertNotIn("n-arin->proactive", world.social)
        self.assertIn("new-npc", world.social)  # 全局键不动
        self.assertNotIn("n-arin->", str(evolution.active_social(world)))

    def test_fog_npc_frees_population_slot(self):
        """B2：雾中人不占活跃人口名额——容量是「同时在场上限」。"""
        llm, w = self._world()
        for i in range(9):  # 3 + 9 = 12，填满活跃名额
            errors = cards.create_npc(w, f"路人{i}", "普通人。")
            self.assertEqual(errors, [])
        errors, _ = cards.emerge_npc(w, "新人", "来晚了的人。", None,
                                     "s-cafe", "测试")
        self.assertTrue(errors)  # 满员
        arin = w.npcs["n-arin"]
        arin.in_fog = True  # 阿凛退入雾中 → 名额释放
        for scene in w.scenes.values():
            if arin.id in scene.npcs:
                scene.npcs.remove(arin.id)
        errors, _ = cards.emerge_npc(w, "新人", "来晚了的人。", None,
                                     "s-cafe", "测试")
        self.assertEqual(errors, [])  # 名额回来了

    def test_pulse_payload_stays_bounded_over_long_runs(self):
        """载荷有界：200 次心跳后，事件类型集合与成本代理不随历史增长。"""
        rec = _RecordingLLM(MockLLM())
        w = generate_world(rec, "世界1", DESC)
        w.player["location"] = "s-alley"  # 远离 NPC，让脉冲干活
        for _ in range(200):
            w.pass_time(6)
            evolution.world_pulse(rec, w)
        self.assertGreater(len(w.events), 200)  # 世界真的活了很久
        payload = json.loads(rec.last_payload)
        kinds = payload["world"]["past_event_types"]
        self.assertLessEqual(len(kinds), 30)  # 类型集合有界，不带历史长度
        self.assertLessEqual(len(w.events) - payload["world"]["event_count"],
                             2)  # 载荷之后最多再落 1-2 条（涌现等）
        self.assertLess(len(rec.last_payload), 20000)  # 成本代理有界


class TestHolding(unittest.TestCase):
    """持有关系：物品被持有 → 隐去、转手留痕、放下回归。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        return llm, w

    def _letter(self, w):
        return next(i for i in w.scenes["s-station"].items
                    if i.get("id") == "i-letter")

    def test_held_item_hidden_from_scene_items(self):
        llm, w = self._world()
        letter = self._letter(w)
        errors = transfer_item(w, letter, "npc:n-arin", "测试")
        self.assertEqual(errors, [])
        self.assertEqual(letter["held_by"], "npc:n-arin")
        visible = [i["id"] for i in active_items(w.scenes["s-station"])]
        self.assertNotIn("i-letter", visible)  # 跟着持有者走了

    def test_transfer_emits_event_and_put_down_returns(self):
        llm, w = self._world()
        letter = self._letter(w)
        transfer_item(w, letter, "npc:n-arin", "测试")
        ev = [e for e in w.events if e.kind == "item_transfer"]
        self.assertTrue(ev)
        self.assertIn("拿走", ev[-1].summary)
        transfer_item(w, letter, "", "测试")  # 放下
        visible = [i["id"] for i in active_items(w.scenes["s-station"])]
        self.assertIn("i-letter", visible)  # 回到场景
        ev2 = [e for e in w.events if e.kind == "item_transfer"]
        self.assertIn("放回", ev2[-1].summary)

    def test_invalid_holder_rejected(self):
        llm, w = self._world()
        letter = self._letter(w)
        errors = transfer_item(w, letter, "item:i-letter", "测试")  # 物品不能持有
        self.assertTrue(errors)
        errors = transfer_item(w, letter, "npc:n-ghost", "测试")  # 引用不存在
        self.assertTrue(errors)

    def test_patch_change_with_held_by(self):
        llm, w = self._world()
        errors = apply_item_patch(
            w, {"op": "change", "item": "i-letter",
                "location": "s-station", "held_by": "player"},
            cause="测试")
        self.assertEqual(errors, [])
        letter = self._letter(w)
        self.assertEqual(letter["held_by"], "player")
        self.assertIn("item_transfer", [e.kind for e in w.events])

    def test_held_item_survives_roundtrip(self):
        llm, w = self._world()
        transfer_item(w, self._letter(w), "npc:n-arin", "测试")
        u = Universe(worlds={w.name: w}, current=w.name)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "u.json"
            u.save(p)
            w2 = Universe.load(p).here
        letter2 = next(i for i in w2.scenes["s-station"].items
                       if i.get("id") == "i-letter")
        self.assertEqual(letter2["held_by"], "npc:n-arin")

    def test_held_item_survives_idle_sweep(self):
        """角色携带的物品不能被场景的闲置清扫扫掉。"""
        llm, w = self._world()
        letter = self._letter(w)
        transfer_item(w, letter, "npc:n-arin", "测试")
        letter["last_turn"] = 1
        w.turn = 1 + ITEM_MAX_IDLE + 1
        w.sweep_items()
        self.assertIn(letter, w.scenes["s-station"].items)


class TestWorldMood(unittest.TestCase):
    """世界氛围：数值 + 事件驱动 + 时间回温 + 标签投影。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-alley"
        return llm, w

    def test_push_and_clamp(self):
        llm, w = self._world()
        evolution.push_world_mood(w, -1.5, "测试")
        self.assertEqual(w.mood_value, -1.0)  # 钳制
        self.assertEqual(w.mood_reason, "测试")

    def test_decay_returns_to_baseline(self):
        llm, w = self._world()
        evolution.push_world_mood(w, -0.8, "大事")
        evolution.decay_world_mood(w, 200)  # 很久以后：回温
        self.assertGreater(w.mood_value, -0.2)

    def test_big_event_darkens_daily_bits_warm(self):
        llm, w = self._world()
        w.pass_time(6)
        plan = json.dumps({
            "events": [{"type": "world_event",
                        "params": {"title": "大事", "detail": "天空裂开",
                                   "location": "s-station",
                                   "intensity": 0.9}}],
            "npc_plans": [], "influences": [],
            "daily_bits": [{"detail": "猫躲雨", "location": "s-station",
                            "intensity": 0.2}],
            "new_scenes": [], "item_patches": []}, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertLess(w.mood_value, 0)  # 大事压低
        self.assertEqual(mood_now(w), "平静")  # 情绪词由 AI 自由提案，默认平静

    def test_goal_completion_warms_world(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.remember(arin, "听说「一封信」出现了。", cause="流言",
                   importance=0.6)
        arin.goals[0]["progress"] = 0.5
        arin.relationship = 10
        w.pass_time(6)
        evolution.heartbeat(llm, w, arin)
        self.assertGreater(w.mood_value, 0)  # 有人达成目标 → 回暖
        self.assertEqual(w.mood_reason, "有人达成了目标")

    def test_law_change_darkens_world(self):
        llm, w = self._world()
        interpreter.change_law(llm, w, "这座城的人从不撒谎")
        self.assertLess(w.mood_value, 0)
        self.assertEqual(w.mood_reason, "天变")

    def test_roundtrip(self):
        llm, w = self._world()
        evolution.push_world_mood(w, 0.4, "回暖")
        u = Universe(worlds={w.name: w}, current=w.name)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "u.json"
            u.save(p)
            u2 = Universe.load(p)
        self.assertEqual(u2.here.mood_value, 0.4)
        self.assertEqual(u2.here.mood_reason, "回暖")

    def test_weather_shift_updates_weather_state(self):
        """天气与情绪分账：weather_shift 改天气，不动情绪。"""
        llm, w = self._world()
        w.pass_time(6)
        plan = json.dumps({
            "events": [{"type": "weather_shift",
                        "params": {"to": "雷雨", "intensity": 0.8}}],
            "npc_plans": [], "influences": [], "daily_bits": [],
            "new_scenes": [], "item_patches": []}, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertEqual(w.weather, "雷雨")
        self.assertEqual(weather_now(w), "雷雨")  # 不带固定形容词，质感交给文本
        self.assertEqual(w.mood_value, 0.0)  # 情绪纹丝不动

    def test_world_mood_word_is_ai_proposed(self):
        """情绪词由 AI 自由提案，引擎不设词典。"""
        llm, w = self._world()
        w.pass_time(6)
        plan = json.dumps({
            "events": [], "npc_plans": [], "influences": [],
            "daily_bits": [], "new_scenes": [], "item_patches": [],
            "world_mood_word": "全城惴惴不安"}, ensure_ascii=False)
        evolution.world_pulse(ScriptedLLM([plan]), w)
        self.assertEqual(mood_now(w), "全城惴惴不安")

    def test_old_save_weather_falls_back_to_base(self):
        llm, w = self._world()
        self.assertEqual(weather_now(w), w.law_profile.atmosphere)


class TestEmergentPlaces(unittest.TestCase):
    """新地名涌现：NPC 去玩家不知道的地方——地名级，不细化。"""

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-cafe")
        w.player["location"] = "s-station"
        return llm, w

    def test_npc_can_enter_fog_scene(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        moved = evolution.move_npc(w, arin, "s-alley", cause="测试")
        self.assertTrue(moved)  # 雾中场景是真实地方，NPC 可以进去
        self.assertEqual(arin.state.location, "s-alley")
        self.assertIn("n-arin", w.scenes["s-alley"].npcs)

    def test_act_to_unknown_place_emerges_fog_scene(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        w.remember(arin, "听说「一封信」出现了。", cause="流言",
                   importance=0.6)
        arin.relationship = 5
        w.pass_time(6)
        # Mock 目标裁决：用 ScriptedLLM 直接灌一个去新地方的行动
        goal = json.dumps({
            "events": [{"type": "npc_acted",
                        "params": {"npc": "n-arin",
                                   "action": "去河边的旧磨坊打听那封信",
                                   "location": "s-99",
                                   "place": "河边的旧磨坊"}}],
            "goal_updates": {"find-letter": 0.5},
            "new_goals": []}, ensure_ascii=False)
        evolution.propose_proactive(ScriptedLLM([goal]), w, arin)
        self.assertIn("s-99", w.scenes)          # 新地名涌现
        place = w.scenes["s-99"]
        self.assertFalse(place.generated)        # 不细化（雾中）
        self.assertEqual(place.name, "河边的旧磨坊")
        self.assertEqual(arin.state.location, "s-cafe")  # 还在路上
        self.assertEqual(arin.state.action.location, "s-99")
        ev = next(e for e in w.events if e.kind == "scene_extended")
        self.assertEqual(ev.cause, "NPC 去往新地方")

    def test_move_to_nonexistent_rejected(self):
        llm, w = self._world()
        arin = w.npcs["n-arin"]
        moved = evolution.move_npc(w, arin, "s-nowhere", cause="测试")
        self.assertFalse(moved)  # 不存在的目的地：先走地名涌现
        self.assertEqual(arin.state.location, "s-cafe")


class TestItemCausalityCarHitsPlayer(unittest.TestCase):
    """最小物品因果回归：行驶中的车撞到玩家。

    验证四件事：
    1. 车辆、玩家、地点能被同一个多实体事件同时引用；
    2. 事件提交后，车辆状态、玩家受伤状态、地点痕迹一起改变；
    3. 缺失实体 / 位置不符 → 整件事拒绝，零部分写入；
    4. 事件可通过 cause 回放，玩家视角不泄漏内部字段。
    已知缺口（不扩展引擎）：车辆「速度」只是 note 文本，没有数值
    属性与前置校验——「速度不满足时拒绝」属于属性层，当前不存在，
    由 test_speed_is_narrative_not_mechanical 固定当前行为。
    """

    def _world(self):
        llm = MockLLM()
        w = generate_world(llm, "世界1", DESC)
        ensure_scene(llm, w, "s-alley")
        w.player["location"] = "s-alley"
        errs = apply_item_patch(w, {"op": "add", "item": "i-car",
                                    "name": "面包车",
                                    "note": "行驶中，速度很快",
                                    "location": "s-alley"},
                                cause="测试布置")
        self.assertEqual(errs, [])
        return w

    def _crash_proposal(self, car_id="i-car"):
        return {
            "title": "行驶中的车撞上了玩家",
            "detail": "面包车刹不住，车头撞上玩家的小腿，"
                      "车轮在湿路面上拖出两道痕迹。",
            "location": "s-alley",
            "intensity": 0.9,
            "participants": [f"item:{car_id}", "player", "scene:s-alley"],
            "item_patches": [{"op": "change", "item": car_id,
                              "location": "s-alley",
                              "note": "撞停，车头凹陷"}],
            "actor_patches": [{"target": "player", "can_act": False,
                               "condition": "被车撞伤，小腿剧痛"}],
            "scene_state_patches": [{"scene": "scene:s-alley", "op": "add",
                                     "fact": "tire-marks",
                                     "text": "路面留下两道轮胎擦痕",
                                     "duration_days": 1}],
        }

    def _car(self, w):
        return next(i for s in w.scenes.values() for i in s.items
                    if i["id"] == "i-car")

    def test_one_event_references_car_player_place(self):
        """验证 1：车辆、玩家、地点被同一个事件同时引用。"""
        w = self._world()
        n0 = len(w.events)
        summaries = evolution.commit_entity_event(w, self._crash_proposal())
        self.assertFalse(any("驳回" in s for s in summaries), summaries)
        root = w.events[n0]
        self.assertEqual(root.kind, "world_event")
        self.assertEqual(root.cause, "多实体事件")
        params = root.payload["event_params"]
        self.assertEqual(set(params["refs"]),
                         {"item:i-car", "player", "scene:s-alley"})

    def test_consequences_all_land_together(self):
        """验证 2：车辆状态、玩家受伤、地点痕迹一起改变。"""
        w = self._world()
        evolution.commit_entity_event(w, self._crash_proposal())
        self.assertIn("撞停", self._car(w)["note"])          # 车辆
        self.assertIs(w.player.get("can_act"), False)         # 玩家受伤
        self.assertIn("被车撞伤", w.player.get("condition", ""))
        alley = w.scenes["s-alley"]
        self.assertTrue(any(f.get("id") == "tire-marks"
                            for f in alley.state_facts))      # 地点痕迹

    def test_missing_entity_rejected_without_partial_state(self):
        """验证 3a：缺少实体 → 整件事拒绝，零部分写入。"""
        w = self._world()
        proposal = self._crash_proposal(car_id="i-ghost-car")
        n0 = len(w.events)
        note_before = self._car(w)["note"]
        summaries = evolution.commit_entity_event(w, proposal)
        self.assertTrue(any("驳回" in s for s in summaries), summaries)
        self.assertEqual(len(w.events), n0)                    # 无新事件
        self.assertEqual(self._car(w)["note"], note_before)    # 车辆未变
        self.assertNotIn("can_act", w.player)                  # 玩家未变
        self.assertFalse(w.scenes["s-alley"].state_facts)      # 场景未变

    def test_wrong_location_rejected_without_partial_state(self):
        """验证 3b：位置不符（玩家不在场）→ 整件事拒绝。"""
        w = self._world()
        w.player["location"] = "s-cafe"
        n0 = len(w.events)
        summaries = evolution.commit_entity_event(w, self._crash_proposal())
        self.assertTrue(any("驳回" in s for s in summaries), summaries)
        self.assertEqual(len(w.events), n0)
        self.assertNotIn("can_act", w.player)
        self.assertFalse(w.scenes["s-alley"].state_facts)

    def test_replayable_and_player_view_clean(self):
        """验证 4：cause 回放 + 玩家视角不泄漏内部字段。"""
        w = self._world()
        evolution.commit_entity_event(w, self._crash_proposal())
        root = next(e for e in reversed(w.events)
                    if e.kind == "world_event"
                    and e.cause == "多实体事件")
        cons = root.payload.get("consequences", {})
        self.assertTrue(cons["item_patches"])       # 后果随根事件存档
        self.assertTrue(cons["actor_patches"])
        self.assertTrue(cons["scene_state_patches"])
        for bad in ("refs", "actor_patches", "can_act", "consequences",
                    "scene_state_patches", "item_patches"):
            self.assertNotIn(bad, root.summary)     # 渲染层无内部字段
        self.assertIn("车", root.summary)

    def test_speed_is_narrative_not_mechanical(self):
        """能力缺口固定：速度只是 note 文本，引擎不校验速度前置。
        断言当前行为（提交成功），防止将来误以为引擎校验了速度。"""
        w = self._world()
        self._car(w)["note"] = "停在路边，熄火状态"  # 速度不满足
        summaries = evolution.commit_entity_event(w, self._crash_proposal())
        self.assertFalse(any("驳回" in s for s in summaries), summaries)

    def test_collision_bridges_to_state_runtime_before_world_commit(self):
        """Fresh projection -> prepare -> one WorldLedger commit."""
        w = self._world()
        before = len(w.events)
        summaries = evolution.commit_entity_event(
            w, self._crash_proposal(), use_state_runtime=True)
        self.assertFalse(any("驳回" in s for s in summaries), summaries)
        self.assertEqual(len(w.events) - before, 4)  # root + three projections
        root = w.events[before]
        audit = root.payload["state_runtime"]
        self.assertEqual(audit["status"], "validated")
        self.assertTrue(audit["proposal_created"])
        self.assertEqual(audit["validation"], "passed")
        self.assertTrue(audit["prepared"])
        self.assertEqual(audit["mode"], "validator")
        self.assertEqual(set(audit["entity_ids"]), {
            "item:i-car", "player", "scene:s-alley"})

    def test_state_runtime_rejection_writes_neither_side(self):
        """Current projection rejects before either ledger changes."""
        w = self._world()
        w.player["location"] = "s-cafe"
        before_events = len(w.events)
        before_note = self._car(w)["note"]
        summaries = evolution.commit_entity_event(
            w, self._crash_proposal(), use_state_runtime=True)
        self.assertTrue(any("驳回" in s for s in summaries), summaries)
        self.assertEqual(len(w.events), before_events)
        self.assertEqual(self._car(w)["note"], before_note)
        self.assertNotIn("can_act", w.player)
        self.assertFalse(w.scenes["s-alley"].state_facts)

    def test_state_runtime_projection_is_rebuilt_after_world_changes(self):
        """A later validation reads current WorldLedger state, not old state."""
        from worldledger.state_runtime_adapter import (
            prepare_current_world_event,
        )
        w = self._world()
        normalized, errors = evolution._normalize_entity_event(
            w, self._crash_proposal())
        self.assertEqual(errors, [])
        prepare_current_world_event(w, normalized)
        w.player["location"] = "s-cafe"
        with self.assertRaises(ValueError):
            prepare_current_world_event(w, normalized)


class TestDemoRuns(unittest.TestCase):
    def test_demo_smoke(self):
        from tools import demo
        # 测试强制 Mock：不烧真实 API，且确定性
        demo.demo(llm=MockLLM())  # 不抛异常即通过

    def test_pilot_runs_and_all_checks_pass(self):
        """python -m tools.pilot 的可运行性 + 21 项断言全部通过。"""
        from tools import pilot
        with tempfile.TemporaryDirectory() as tmp:
            result = pilot.run(Path(tmp))
            self.assertEqual(result["total"], 21)
            self.assertEqual(result["passed"], 21,
                             [c for c in result["checks"] if not c["passed"]])
            for name in ("pilot_ledger.json", "pilot_player_view.txt",
                         "pilot_audit.md"):
                self.assertTrue((Path(tmp) / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
