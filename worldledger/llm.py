"""LLM 抽象层：可插拔。

- MockLLM：确定性内核（引擎的离线模拟器）。模板库按描述关键字选择
  世界套件（雨城 / 沙漠 / 中性）——描述决定世界，机制层零感知。
  雨城套件是测试夹具：描述含「雨」时的行为与历史版本一致。
- OpenAICompatLLM：环境变量配置的 OpenAI 兼容接口。
- get_llm()：有 key 用真实 API，否则自动回落 Mock。
"""
from __future__ import annotations

import json
import math
import os
import urllib.request

# ---------------- 基础 ----------------


class LLMError(Exception):
    pass


def extract_json(text: str) -> str:
    """从 LLM 输出中提取 JSON 片段（容忍代码块围栏与前后杂文）。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError(f"输出中找不到 JSON：{text[:200]}")
    return text[start:end + 1]


class BaseLLM:
    name = "base"

    def chat(self, system: str, user: str) -> str:
        raise NotImplementedError

    def chat_json(self, system: str, user: str, attempts: int = 2) -> dict:
        last_err: Exception | None = None
        for _ in range(attempts):
            text = self.chat(system, user)
            try:
                data = json.loads(extract_json(text))
                if isinstance(data, dict):
                    return data
            except (LLMError, json.JSONDecodeError) as e:
                last_err = e
        raise LLMError(f"LLM 输出 {attempts} 次均无法解析为 JSON：{last_err}")


# ---------------- Mock 规则引擎 ----------------


class MockLLM(BaseLLM):
    name = "mock"

    def chat(self, system: str, user: str) -> str:
        if "TASK:WORLDGEN" in system:
            return json.dumps(mock_worldgen(user), ensure_ascii=False)
        if "TASK:LAWCHANGE" in system:
            return json.dumps(mock_lawchange(user), ensure_ascii=False)
        if "TASK:DIALOGUE" in system:
            return json.dumps(mock_dialogue(user), ensure_ascii=False)
        if "TASK:PLAYERACT" in system:
            return json.dumps(mock_playact(user), ensure_ascii=False)
        if "TASK:NPCCATCHUP" in system:
            return json.dumps(mock_catchup(user), ensure_ascii=False)
        if "TASK:NPCHEARTBEAT" in system:
            return json.dumps(mock_heartbeat(user), ensure_ascii=False)
        if "TASK:SCENEGEN" in system:
            return json.dumps(mock_scenegenerate(user), ensure_ascii=False)
        if "TASK:RUMOR" in system:
            return json.dumps(mock_rumor(user), ensure_ascii=False)
        if "TASK:NPCGOAL" in system:
            return json.dumps(mock_npcgoal(user), ensure_ascii=False)
        if "TASK:NPCINITIATE" in system:
            return json.dumps(mock_npcinitiate(user), ensure_ascii=False)
        if "TASK:ACTIONRESOLVE" in system:
            return json.dumps(mock_actionresolve(user), ensure_ascii=False)
        if "TASK:WORLDPULSE" in system:
            return json.dumps(mock_worldpulse(user), ensure_ascii=False)
        if "TASK:MEMORYALIGN" in system:
            return json.dumps(mock_memoryalign(user), ensure_ascii=False)
        raise LLMError(f"Mock 不认识的任务：{system[:80]}")


# —— 法则模板库（关键词 → 法则条目，世界无关）——

_LAW_TEMPLATES = [
    # (关键词列表, id, 触发, 后果, 强度)
    (["谎", "真心"], "lie", "有人撒谎", "撒谎者当场说漏一件真心话", 1.0),
    (["火", "燃烧", "灼"], "fire", "火焰点燃", "火焰蔓延速度加倍", 0.8),
    (["诚实", "不撒谎", "从不撒谎", "真话"], "honesty", "人不会撒谎",
     "每一句话都是完整的真相", 1.0),
    (["冰", "寒冷"], "ice", "温度低于冰点", "说出的话会凝结成冰晶", 0.6),
]


def _match_laws(text: str) -> list[dict]:
    laws = []
    for keywords, law_id, trigger, effect, intensity in _LAW_TEMPLATES:
        if any(k in text for k in keywords):
            laws.append({"id": law_id, "trigger": trigger, "effect": effect,
                         "intensity": intensity})
    return laws


_ATMOSPHERE_KEYWORDS = [("雨", "雨"), ("永远", "永续"), ("永不", "永续"),
                        ("夜", "夜"), ("黑暗", "夜"), ("沙漠", "沙漠"),
                        ("落日", "落日"), ("雷", "雷"), ("雪", "雪")]


def _atmosphere(text: str) -> str:
    found = [v for k, v in _ATMOSPHERE_KEYWORDS if k in text]
    return "·".join(dict.fromkeys(found)) if found else "平凡"


# ---------------- 世界套件（描述决定世界） ----------------
# 每个套件自带场景、人物、作息、台词、互动、日历。
# 雨城套件 = 测试夹具：行为与历史版本完全一致。

_SUITES = {
    "rain": {
        # (id, 名称, 线索, 描述[, 可选关键词])
        "scene_templates": [
            ("s-station", "旧车站", "褪色的站台，时刻表停在午夜",
             "褪色的站台，长椅积水，时刻表永远停在午夜。"),
            ("s-cafe", "街角咖啡店", "暖黄灯光的街角咖啡店",
             "暖黄灯光，雨点敲窗，吧台后蒸汽升腾。"),
            ("s-alley", "积水小巷", "积水的小巷，路灯忽明忽灭",
             "窄巷两侧旧墙，路灯忽明忽灭，水洼倒映着天。"),
            ("s-coast", "海边栈桥", "伸进灰海的海边栈桥",
             "栈桥伸进灰海，浪声闷响，海风带着盐味。", ["海", "船"]),
        ],
        "links": [("s-station", "s-cafe"), ("s-cafe", "s-alley"),
                  ("s-alley", "s-coast")],
        "npcs": [
            {"id": "n-arin", "name": "阿凛",
             "home": "s-cafe", "secretive": True,
             "persona": "街角咖啡店的店长，藏着秘密，雨天总是望着窗外。",
             "traits": {"沉静": True},
             "goals": [{"id": "find-letter", "progress": 0.0,
                        "text": "查明车站那封信是谁寄的"},
                       {"id": "cafe-alive", "progress": 0.3,
                        "text": "守住这家咖啡店，别让它在雨里关门"}],
             "relationship": 0},
            {"id": "n-zhou", "name": "老周",
             "home": "s-station", "secretive": False,
             "persona": "旧车站的守夜人，说话慢，喜欢打听旅客的去向。",
             "traits": {"健谈": True}, "relationship": 0},
            {"id": "n-man", "name": "小满",
             "home": "s-alley", "secretive": False,
             "persona": "雨巷里的少年，总在等一封不会来的信。",
             "traits": {"执拗": True},
             "goals": [{"id": "wait-letter", "progress": 0.0,
                        "text": "等一封寄给自己的信"}],
             "relationship": 0},
        ],
        "schedules": {
            "n-arin": {0: ("s-cafe", "开店准备"), 1: ("s-cafe", "煮咖啡"),
                       2: ("s-cafe", "望着窗外"), 3: ("s-station", "看时刻表")},
            "n-zhou": {0: ("s-station", "扫地"), 1: ("s-station", "打盹"),
                       2: ("s-station", "查看时刻表"),
                       3: ("s-station", "守夜")},
            "n-man": {0: ("s-alley", "等信"), 1: ("s-alley", "看雨"),
                      2: ("s-alley", "徘徊"), 3: ("s-cafe", "躲雨")},
        },
        "replies": {
            "n-arin": {
                "ask_secret": "等一个不会来的人。",
                "order": "今天的豆子带着雨味，要一杯吗？",
                "generic": "雨天让客人变少，也让人变诚实。",
                "leave": "伞在门边，别忘了。",
            },
            "n-zhou": {
                "ask_secret": "我在这里守了三十年，见过每一班不会停靠的车。",
                "order": "这里只有热水，和等车的人。",
                "generic": "这雨下了多少年，就有多少人没等到人。",
                "leave": "慢走，下一班车也快到了。",
            },
            "n-man": {
                "ask_secret": "等一封信。寄信的人说，雨天就会到。",
                "order": "我不喝咖啡，喝雨。",
                "generic": "你听见了吗？雨落在旧墙上，像有人敲门。",
                "leave": "别让信淋湿了。",
            },
        },
        "interact_lines": {
            "n-arin": "今天的雨还是不停。",
            "n-zhou": "停了，就不是这座城了。",
            "n-man": "你也在等吗？",
        },
        "honesty_lines": {
            "n-arin": "你来得正好。我等的信，寄信人就是你。",
        },
        "calendar": {
            # (事件类型, 最早回合, 参数)——世界自己的日历
            "events": [
                ("world_event", 48, {
                    "title": "午夜的钟声",
                    "detail": "旧车站的钟在无人听见的时刻敲了十三下。",
                    "location": "s-station", "intensity": 0.7}),
                ("world_event", 24, {
                    "title": "人们在旧车站发现了一封信",
                    "detail": "信封被雨打湿了一角，没人知道是谁放在那儿的。",
                    "location": "s-station", "intensity": 0.6}),
            ],
            # (最早回合, 生长参数)——世界生长
            "extension": (72, {"from": "s-station", "name": "雾中的旧码头",
                               "hint": "河边的旧码头，船早已不靠岸"}),
        },
        # 初始物品：世界生成时就在场景里（状态，不是事件）
        "initial_items": {
            "s-station": [
                {"id": "i-letter", "name": "一封信",
                 "note": "信封被雨打湿了一角",
                 "cause": "世界生成"},
            ],
        },
        "letter_place": "s-station",
        "letter_action": "披上外套，去了车站打听那封信的下落",
        "wait_action": "听见信的消息，一路跑去了车站",
        "rumor_line": "你听说了吗？车站那封信的事。",
        "daily_pool": ["晾衣绳上的衣服被收了进去",
                       "水洼里的倒影被风吹碎",
                       "檐角的猫躲进纸箱里",
                       "店门口的伞架上少了一把伞"],
        "life_newcomer": {"name": "学徒·小晴",
                          "persona": "开学季新来的学徒，总在伞铺门口探头探脑，"
                                     "对雨有着说不清的好奇。",
                          "goal": {"id": "g-a1", "text": "学会修好第一把伞",
                                   "progress": 0.0},
                           "location": "s-station",
                           "reason": "开学季，新学徒入学",
                           "activity": "蹲在站台边，用粉笔画着伞骨的构造图"},
    },

    "desert": {
        "scene_templates": [
            ("s-d-market", "绿洲市集", "沙暴边缘的绿洲市集",
             "市集搭在绿洲边缘，风一过，遮阳的布蓬就哗哗作响。"),
            ("s-d-oasis", "月牙泉边", "月牙泉边的芦苇丛",
             "泉水清得发亮，芦苇丛里藏着昼伏夜出的小兽。"),
            ("s-d-ruins", "沙丘残垣", "被沙半埋的旧城残垣",
             "半截石柱从沙里伸出来，刻着没人认得的字。"),
        ],
        "links": [("s-d-market", "s-d-oasis"),
                  ("s-d-oasis", "s-d-ruins")],
        "npcs": [
            {"id": "n-d-hana", "name": "哈娜",
             "home": "s-d-market", "secretive": True,
             "persona": "绿洲市集的香料商人，眼角有风沙磨出的细纹。",
             "traits": {"沉静": True},
             "goals": [{"id": "find-letter", "progress": 0.0,
                        "text": "查明市集那封被沙掩埋的信是谁寄的"},
                       {"id": "market-alive", "progress": 0.3,
                        "text": "让市集熬过这个旱季"}],
             "relationship": 0},
            {"id": "n-d-old", "name": "驼翁",
             "home": "s-d-market", "secretive": False,
             "persona": "赶了一辈子骆驼的老人，认得每一只驼铃。",
             "traits": {"健谈": True}, "relationship": 0},
            {"id": "n-d-boy", "name": "沙孩",
             "home": "s-d-oasis", "secretive": False,
             "persona": "在泉边长大的少年，总说自己在等一封从北方来的信。",
             "traits": {"执拗": True},
             "goals": [{"id": "wait-letter", "progress": 0.0,
                        "text": "等一封寄给自己的信"}],
             "relationship": 0},
        ],
        "schedules": {
            "n-d-hana": {0: ("s-d-market", "摆摊准备"),
                         1: ("s-d-market", "叫卖香料"),
                         2: ("s-d-market", "望着沙丘"),
                         3: ("s-d-oasis", "打水")},
            "n-d-old": {0: ("s-d-market", "喂骆驼"),
                        1: ("s-d-market", "打盹"),
                        2: ("s-d-market", "数驼铃"),
                        3: ("s-d-market", "守夜")},
            "n-d-boy": {0: ("s-d-oasis", "等信"),
                        1: ("s-d-oasis", "看泉"),
                        2: ("s-d-oasis", "徘徊"),
                        3: ("s-d-market", "躲风")},
        },
        "replies": {
            "n-d-hana": {
                "ask_secret": "我在等一支不会再来的驼队。",
                "order": "香料称两，不还价。",
                "generic": "风一停，市集就安静得能听见沙子走路。",
                "leave": "日头毒，带块盐再走。",
            },
            "n-d-old": {
                "ask_secret": "我数了一辈子驼铃，从没数错过。",
                "order": "骆驼只喝水，不卖。",
                "generic": "北边的沙暴又要来了，你闻闻这风。",
                "leave": "顺着驼铃走，不会迷路。",
            },
            "n-d-boy": {
                "ask_secret": "一封从北方来的信。说好沙暴停的那天到。",
                "order": "泉水分你一半，别喝光。",
                "generic": "你看泉底，有一小块天。",
                "leave": "别把脚印留在沙丘上，风会记得。",
            },
        },
        "interact_lines": {
            "n-d-hana": "这风里，有雨的味道。",
            "n-d-old": "有雨，就不是这片沙了。",
            "n-d-boy": "你也在等吗？",
        },
        "honesty_lines": {},
        "calendar": {
            "events": [
                ("world_event", 48, {
                    "title": "永不熄灭的火盆",
                    "detail": "市集中央的火盆在无风的夜里自己燃了起来。",
                    "location": "s-d-market", "intensity": 0.7}),
                ("world_event", 24, {
                    "title": "人们在市集发现了一封被沙掩埋的信",
                    "detail": "信封上写着没人认得的名字。",
                    "location": "s-d-market", "intensity": 0.6}),
            ],
            "extension": (72, {"from": "s-d-market",
                               "name": "沙暴尽头的旧驿站",
                               "hint": "沙暴尽头，半埋在沙里的旧驿站"}),
        },
        "initial_items": {
            "s-d-market": [
                {"id": "i-letter", "name": "一封被沙掩埋的信",
                 "note": "信封上写着没人认得的名字",
                 "cause": "世界生成"},
            ],
        },
        "letter_place": "s-d-market",
        "letter_action": "顶着日头，去了市集打听那封信的下落",
        "wait_action": "听见信的消息，一路跑去了市集",
        "rumor_line": "你听说了吗？市集那封信的事。",
        "daily_pool": ["摊主把遮阳布重新系紧",
                       "风把沙粒吹进没盖的茶杯",
                       "骆驼在墙影下打了个盹",
                       "卖冰水的小贩擦着玻璃瓶"],
    },

    "neutral": {
        "scene_templates": [
            ("s-x-market", "老市集", "青石板铺成的老市集",
             "青石板路两侧摆满摊子，叫卖声混着旧货的气味。"),
            ("s-x-tower", "旧钟楼", "广场边的旧钟楼",
             "钟楼外墙爬满藤蔓，钟面的指针停在黄昏。"),
            ("s-x-bank", "河岸", "长满芦苇的河岸",
             "河水很缓，芦苇随风倒向同一个方向。"),
        ],
        "links": [("s-x-market", "s-x-tower"),
                  ("s-x-tower", "s-x-bank")],
        "npcs": [
            {"id": "n-x-shop", "name": "店主人",
             "home": "s-x-market", "secretive": True,
             "persona": "老市集里开杂货铺的人，话不多，记性极好。",
             "traits": {"沉静": True},
             "goals": [{"id": "find-letter", "progress": 0.0,
                        "text": "查明市集那封无人认领的信是谁寄的"},
                       {"id": "keep-shop", "progress": 0.3,
                        "text": "守住这间老铺子"}],
             "relationship": 0},
            {"id": "n-x-watch", "name": "守夜人",
             "home": "s-x-tower", "secretive": False,
             "persona": "旧钟楼的守夜人，习惯在整点咳嗽一声。",
             "traits": {"健谈": True}, "relationship": 0},
            {"id": "n-x-kid", "name": "少年",
             "home": "s-x-bank", "secretive": False,
             "persona": "常在河岸发呆的少年，说自己在等一封不会到的信。",
             "traits": {"执拗": True},
             "goals": [{"id": "wait-letter", "progress": 0.0,
                        "text": "等一封寄给自己的信"}],
             "relationship": 0},
        ],
        "schedules": {
            "n-x-shop": {0: ("s-x-market", "开门摆货"),
                         1: ("s-x-market", "理账"),
                         2: ("s-x-market", "望着街口"),
                         3: ("s-x-tower", "听钟")},
            "n-x-watch": {0: ("s-x-tower", "扫地"),
                          1: ("s-x-tower", "打盹"),
                          2: ("s-x-tower", "擦钟面"),
                          3: ("s-x-tower", "守夜")},
            "n-x-kid": {0: ("s-x-bank", "等信"),
                        1: ("s-x-bank", "看水"),
                        2: ("s-x-bank", "徘徊"),
                        3: ("s-x-market", "躲雨")},
        },
        "replies": {
            "n-x-shop": {
                "ask_secret": "我在等一个很多年前说要回来的人。",
                "order": "杂货都在架子上，自己看。",
                "generic": "这条街的人来来去去，只有钟声不变。",
                "leave": "天晚了，路上当心。",
            },
            "n-x-watch": {
                "ask_secret": "我守了三十年，见过每一场不该响的钟声。",
                "order": "钟楼不卖东西。",
                "generic": "整点的钟声，比人诚实。",
                "leave": "慢走，别误了整点。",
            },
            "n-x-kid": {
                "ask_secret": "一封信。寄信的人说，河水涨起来的那天到。",
                "order": "我不买东西，只看水。",
                "generic": "你听见了吗？芦苇在学人说话。",
                "leave": "别把信丢进河里。",
            },
        },
        "interact_lines": {
            "n-x-shop": "今天的钟声，慢了半拍。",
            "n-x-watch": "慢半拍，才是这座城的钟。",
            "n-x-kid": "你也在等吗？",
        },
        "honesty_lines": {},
        "calendar": {
            "events": [
                ("world_event", 48, {
                    "title": "钟楼的钟敲了十三下",
                    "detail": "旧钟楼的钟在无人听见的时刻敲了十三下。",
                    "location": "s-x-tower", "intensity": 0.7}),
                ("world_event", 24, {
                    "title": "人们在市集发现了一封无人认领的信",
                    "detail": "信封被露水打湿了一角。",
                    "location": "s-x-market", "intensity": 0.6}),
            ],
            "extension": (72, {"from": "s-x-market", "name": "更远处的石桥",
                               "hint": "河的下游，一座很久没人走的石桥"}),
        },
        "initial_items": {
            "s-x-market": [
                {"id": "i-letter", "name": "一封无人认领的信",
                 "note": "信封被露水打湿了一角",
                 "cause": "世界生成"},
            ],
        },
        "letter_place": "s-x-market",
        "letter_action": "去了市集打听那封信的下落",
        "wait_action": "听见信的消息，一路跑去了市集",
        "rumor_line": "你听说了吗？市集那封信的事。",
        "daily_pool": ["钟楼的指针在风里晃了晃",
                       "河边的芦苇被风压弯",
                       "市集收摊的竹筐堆成两摞",
                       "谁家的窗帘在风里鼓起来"],
    },
}


def _suite_key_for_desc(desc: str) -> str:
    # 只有「雨」是雨城套件（测试夹具）；雪/雷不冒充雨城
    if "雨" in desc:
        return "rain"
    if any(k in desc for k in ("沙漠", "沙", "落日", "绿洲", "旱")):
        return "desert"
    return "neutral"


def _suite_key_for_atmo(atmo: str) -> str:
    # 与 desc 版本一致：只有「雨」是雨城套件
    if "雨" in atmo:
        return "rain"
    if "沙漠" in atmo:
        return "desert"
    return "neutral"


# 合并查找表：运行时按实体 id 直接查，机制层不感知套件
_ALL_SCENE_DESCS: dict[str, str] = {}
_ALL_SCHEDULES: dict[str, dict] = {}
_ALL_REPLIES: dict[str, dict] = {}
_ALL_INTERACT_LINES: dict[str, str] = {}
_ALL_HONESTY_LINES: dict[str, str] = {}
for _s in _SUITES.values():
    for _tpl in _s["scene_templates"]:
        _ALL_SCENE_DESCS[_tpl[0]] = _tpl[3]
    _ALL_SCHEDULES.update(_s.get("schedules", {}))
    _ALL_REPLIES.update(_s.get("replies", {}))
    _ALL_INTERACT_LINES.update(_s.get("interact_lines", {}))
    _ALL_HONESTY_LINES.update(_s.get("honesty_lines", {}))


def mock_worldgen(desc: str) -> dict:
    """一句话 → 世界 JSON（确定性）。套件随描述选择；出生场景之外雾中。"""
    suite = _SUITES[_suite_key_for_desc(desc)]
    laws = _match_laws(desc)
    has_lie = any(law["id"] == "lie" for law in laws)

    scenes = []
    for tpl in suite["scene_templates"]:
        sid, name, hint, sdesc = tpl[:4]
        optional = tpl[4] if len(tpl) > 4 else []
        if optional and not any(k in desc for k in optional):
            continue
        scenes.append({"id": sid, "name": name, "description": sdesc,
                       "hint": hint, "npcs": [], "exits": [],
                       "items": [dict(i) for i in
                                 suite.get("initial_items", {}).get(sid, [])]})
    by_id = {s["id"]: s for s in scenes}
    for a, b in suite["links"]:
        if a in by_id and b in by_id:
            by_id[a]["exits"].append(b)
            by_id[b]["exits"].append(a)

    npcs = []
    for tpl in suite["npcs"]:
        npc = {
            "id": tpl["id"], "name": tpl["name"],
            "persona": tpl["persona"],
            "traits": {"爱说谎": True}
            if (has_lie and tpl.get("secretive"))
            else dict(tpl.get("traits", {})),
            "goals": [dict(g) for g in tpl.get("goals", [])],
            "relationship": int(tpl.get("relationship", 0)),
            "home": tpl["home"],
        }
        npcs.append(npc)
    for npc in npcs:
        for scene in scenes:
            if scene["id"] == npc["home"]:
                scene["npcs"].append(npc["id"])
    for i, scene in enumerate(scenes):
        if i > 0:  # 贴片式：出生场景之外全部雾中
            scene["description"] = ""
    return {
        "atmosphere": _atmosphere(desc),
        "laws": laws,
        "scenes": scenes,
        "npcs": npcs,
    }


def mock_lawchange(user: str) -> dict:
    """改法则请求 → 新法则集（替换式），氛围不变。

    user 为 JSON：{current: {atmosphere, laws}, request: 文本}
    """
    state = json.loads(user)
    current = state["current"]
    request = state["request"]
    new_laws = _match_laws(request)
    if not new_laws:
        new_laws = current.get("laws", [])  # 没识别出关键词 → 保持原法则
    # 天变语义是替换式重立法则：声明「从不撒谎」时，旧的「谎」法则必须废除。
    if any(law["id"] == "honesty" for law in new_laws):
        new_laws = [law for law in new_laws if law["id"] != "lie"]
    return {"atmosphere": current.get("atmosphere", "平凡"), "laws": new_laws}


# —— 对话裁决规则 ——


def _intent(text: str) -> str:
    if any(k in text for k in ("谁", "什么", "为什么", "信", "等")):
        return "ask_secret"
    if any(k in text for k in ("再见", "离开", "走了")):
        return "leave"
    if any(k in text for k in ("咖啡", "喝", "买", "点")):
        return "order"
    return "generic"


_CHOICES = ["追问下去", "换个话题", "道别离开"]

# 自定义 NPC（玩家导入/创建，无模板台词）的通用回复
_GENERIC_REPLIES = {
    "ask_secret": "你为什么会想知道这个？",
    "order": "这里没有你想要的东西。",
    "generic": "……（静静地看着你）",
    "leave": "慢走。",
}


def mock_playact(user: str) -> dict:
    """玩家动作裁决（确定性夹具）：按关键词给一个简单分寸。

    注意：这是 Mock 自己的夹具启发式——真实模式的引擎没有类型表，
    分寸由 AI 按角色判断。
    """
    state = json.loads(user)
    npc = state["npc"]
    text = state["action"]
    rel = int(npc.get("relationship", 0))
    intimate = any(k in text for k in ("亲", "吻", "抱"))
    rough = "推" in text
    if rough:
        accepted, rel_delta, mood_delta = True, -10, -0.5
        reply = "……（后退了半步，眼神冷了下来）"
    elif intimate and rel < 50:
        accepted, rel_delta, mood_delta = False, 0, -0.1
        reply = "……（侧身躲开了你的动作）"
    else:
        accepted, rel_delta, mood_delta = True, 3, 0.2
        reply = "……（没有躲开）"
    return {"accepted": accepted, "reply": reply,
            "relationship_delta": rel_delta, "mood_delta": mood_delta,
            "memory_importance": 0.7,
            "memory": f"他对我做了「{text[:20]}」",
            "law_ids": []}


def mock_dialogue(user: str) -> dict:
    """对话裁决（确定性）。

    user 为 JSON：{world: {atmosphere, laws, turn}, npc: {id, name,
    persona, traits, relationship, memories}, player_input}
    返回：{reply, choices, law_ids, relationship_delta, mood_delta,
    memory_importance, memory}
    """
    state = json.loads(user)
    npc = state["npc"]
    laws = state["world"]["laws"]
    player_input = state["player_input"]
    intent = _intent(player_input)
    npc_id = npc["id"]
    reply = _ALL_REPLIES.get(npc_id, _GENERIC_REPLIES).get(intent, "……")

    law_ids: list[str] = []
    lie_law = next((l for l in laws if l["id"] == "lie"), None)
    if (intent == "ask_secret" and lie_law
            and npc.get("traits", {}).get("爱说谎")):
        law_ids.append("lie")
    honesty_law = next((l for l in laws if l["id"] == "honesty"), None)
    if intent == "ask_secret" and honesty_law:
        reply = _ALL_HONESTY_LINES.get(npc_id, "我全都告诉你，不瞒你。")
        law_ids.append("honesty")
    if intent == "order" and any(l["id"] == "fire" for l in laws):
        law_ids.append("fire")

    delta = {"ask_secret": 2, "order": 1, "generic": 0, "leave": -1}[intent]
    mood_delta = {"ask_secret": 0.15, "order": 0.05, "generic": 0.0,
                  "leave": -0.05}[intent]
    importance = {"ask_secret": 0.7, "order": 0.5, "generic": 0.4,
                  "leave": 0.3}[intent]
    memory = f"你问「{player_input}」，{npc['name']} 答「{reply}」"
    return {
        "reply": reply,
        "choices": _CHOICES,
        "law_ids": law_ids,
        "relationship_delta": delta,
        "mood_delta": mood_delta,
        "memory_importance": importance,
        "memory": memory,
        "memory_keywords": [],
        "item_patches": [],
    }


# —— NPC 演化规则（确定性作息 + 情绪漂移 + 互动）——

_PHASE_MOOD = {0: "清醒", 1: "平静", 2: "沉静", 3: "倦怠"}

INTERACT_COOLDOWN = 6  # 同一对 NPC 两次搭话的最小回合间隔


def _schedule_for(npc_id: str, phase: int) -> tuple[str | None, str]:
    return _ALL_SCHEDULES.get(npc_id, {}).get(phase, (None, "待机"))


def _mood_for(phase: int, weather: str) -> str:
    mood = _PHASE_MOOD.get(phase, "平静")
    if "雨" in weather and phase == 2:
        mood = "忧郁"
    return mood


def mock_catchup(user: str) -> dict:
    """读取补算：把 NPC 状态推进到当前回合。

    user 为 JSON：{phase, day, weather, elapsed_turns,
                    npc: {id, name, state: {location, activity, mood}},
                    scenes: {id: name}}
    返回：{location, activity, mood, moved, memory}
    """
    state = json.loads(user)
    npc = state["npc"]
    phase = state["phase"]
    old_loc = npc["state"]["location"]
    loc, activity = _schedule_for(npc["id"], phase)
    mood = _mood_for(phase, state["weather"])
    if loc is None:
        loc = old_loc
    moved = loc != old_loc
    memory = ""
    if moved and state["elapsed_turns"] >= 24:
        scene_name = state["scenes"].get(loc, loc)
        memory = f"过了些日子，我来到了「{scene_name}」。"
    return {"location": loc, "activity": activity, "mood": mood,
            "moved": moved, "memory": memory}


def mock_heartbeat(user: str) -> dict:
    """心跳演化：玩家在场场景里，NPC 每回合的活法。

    user 为 JSON：{phase, weather, npc: {id, name, state}, present: [id],
                    can_interact: bool}
    返回：{activity, mood, location,
            interaction: {with: id, line: str} 或 null}
    """
    state = json.loads(user)
    npc = state["npc"]
    phase = state["phase"]
    loc, activity = _schedule_for(npc["id"], phase)
    mood = _mood_for(phase, state["weather"])
    if loc is None:
        loc = npc["state"]["location"]
    interaction = None
    if state.get("can_interact") and state.get("present"):
        line = _ALL_INTERACT_LINES.get(npc["id"], "……")
        interaction = {"with": state["present"][0], "line": line}
    return {"activity": activity, "mood": mood, "location": loc,
            "interaction": interaction}


def mock_npcgoal(user: str) -> dict:
    """驱动力裁决（确定性）：条件成熟 → 提案 0-1 个主动事件。

    目标规则按 id 匹配（find-letter / wait-letter），动作文本与地点
    来自世界套件——引擎对具体世界零感知。

    user 为 JSON：{npc: {id, goals, relationship, mood_value, state},
                   memories: [...], player_traces: [...], atmosphere: str}
    返回：{"events": [...], "goal_updates": {goal_id: progress},
           "new_goals": [...]}
    """
    state = json.loads(user)
    npc = state["npc"]
    suite = _SUITES[_suite_key_for_atmo(state.get("atmosphere", ""))]
    letter_loc = suite["letter_place"]
    memories = state.get("memories", [])
    rel = npc.get("relationship", 0)
    mood = npc.get("mood_value", 0.0)
    npc_loc = npc.get("state", {}).get("location", "")
    events = []
    updates = {}
    new_goals = []
    for goal in npc.get("goals", []):
        gid = goal.get("id", "")
        progress = float(goal.get("progress", 0.0))
        if gid == "find-letter" and progress < 1.0:
            heard = any(("一封信" in m) or ("信" in m and "听说" in m)
                        for m in memories)
            if heard and progress == 0.0 and (rel >= 5 or mood >= 0.2):
                events.append({"type": "npc_acted",
                               "params": {"npc": npc["id"],
                                          "action": suite["letter_action"],
                                          "location": letter_loc}})
                updates["find-letter"] = {"progress": 0.5,
                                          "because": "打听到了那封信的消息"}
            elif heard and progress >= 0.5 and rel >= 10:
                events.append({"type": "note_left",
                                      "params": {"npc": npc["id"],
                                          "location": npc_loc,
                                          "content": "我查到一些关于那封信的事。今晚别走，等我。"}})
                updates["find-letter"] = {"progress": 1.0,
                                          "because": "查清了信的线索"}
                new_goals.append({"id": "find-sender",
                                  "text": "找到那封信的寄信人",
                                  "progress": 0.0})
        elif gid == "wait-letter" and progress < 1.0:
            if any("一封信" in m for m in memories):
                events.append({"type": "npc_acted",
                               "params": {"npc": npc["id"],
                                          "action": suite["wait_action"],
                                          "location": letter_loc}})
                updates["wait-letter"] = {"progress": 1.0,
                                          "because": "亲眼看到了那封信"}
    return {"events": events, "goal_updates": updates,
            "new_goals": new_goals}


def mock_worldpulse(user: str) -> dict:
    """统一心跳（确定性）：套件日历事件 + 到期 NPC 计划 + 流言 + 世界生长。"""
    state = json.loads(user)
    w = state["world"]
    suite = _SUITES[_suite_key_for_atmo(w.get("atmosphere", ""))]
    turn = int(w["turn"])
    phase = (turn // 6) % 4
    past = w.get("past_event_types", [])
    weather = w.get("atmosphere", "")
    letter_loc = suite["letter_place"]

    events = []
    for kind, min_turn, params in suite["calendar"]["events"]:
        if turn >= min_turn and kind not in past:
            events.append({"type": kind, "params": dict(params)})

    due = state.get("due_npcs", [])
    npc_plans = []
    for npc in due:
        nid = npc["id"]
        cur_loc = npc["state"]["location"]
        loc, activity = _schedule_for(nid, phase)
        if loc is None:
            loc = cur_loc
        plan = {"npc": nid,
                "state": {"activity": activity,
                          "mood": _mood_for(phase, weather),
                          "location": loc}}
        rel = npc.get("relationship", 0)
        memories = npc.get("memories", [])
        for goal in npc.get("goals", []):
            gid = goal.get("id", "")
            progress = float(goal.get("progress", 0.0))
            if gid == "find-letter" and progress < 1.0:
                heard = any(("一封信" in m) or ("信" in m and "听说" in m)
                            for m in memories)
                if heard and progress == 0.0 and (rel >= 5
                                                  or npc.get("mood_value", 0) >= 0.2):
                    plan["action"] = {"type": "npc_acted",
                                      "params": {"npc": nid,
                                                 "action": suite["letter_action"],
                                                 "location": letter_loc}}
                    plan["goal_updates"] = {"find-letter": {
                        "progress": 0.5, "because": "打听到了那封信的消息"}}
                elif heard and progress >= 0.5 and rel >= 10:
                    plan["action"] = {"type": "note_left",
                                      "params": {"npc": nid,
                                                 "location": cur_loc,
                                                 "content": "我查到一些关于那封信的事。今晚别走，等我。"}}
                    plan["goal_updates"] = {"find-letter": {
                        "progress": 1.0, "because": "查清了信的线索"}}
                    plan["new_goals"] = [{"id": "find-sender",
                                           "text": "找到那封信的寄信人",
                                           "progress": 0.0}]
            elif gid == "wait-letter" and progress < 1.0:
                if any("一封信" in m for m in memories):
                    plan["action"] = {"type": "npc_acted",
                                      "params": {"npc": nid,
                                                 "action": suite["wait_action"],
                                                 "location": letter_loc}}
                    plan["goal_updates"] = {"wait-letter": {
                        "progress": 1.0, "because": "亲眼看到了那封信"}}
        # 日程跨场景变化先形成在途行动；state.location 只回填身体现状，
        # 不直接把身体瞬移到行动者或日程的目的地。
        if loc != cur_loc and not plan.get("action"):
            plan["action"] = {
                "type": "npc_acted",
                "params": {"npc": nid, "action": "开始前往新的地点",
                            "location": loc, "days": 1.0},
            }
        others = [o for o in due
                  if o["id"] != nid and o["state"]["location"] == loc]
        if others and not plan.get("action"):
            plan["interaction"] = {"with": others[0]["id"],
                                   "line": _ALL_INTERACT_LINES.get(nid, "……")}
        npc_plans.append(plan)

    new_scenes = []
    ext_turn, ext_params = suite["calendar"]["extension"]
    if turn >= ext_turn and "scene_extended" not in past:
        new_scenes.append(dict(ext_params))
    # 日常小事：确定性轮换（平淡生活是世界的底色）
    pool = suite.get("daily_pool", ["日子照常，没什么特别的事发生。"])
    start_sid = suite["scene_templates"][0][0]
    daily_bits = [{"detail": pool[(turn // 6) % len(pool)],
                   "location": start_sid, "intensity": 0.2}]
    # 人口生态：生活流入（确定性——第 48 回合，新学徒入学）
    new_npcs = []
    life = suite.get("life_newcomer")
    if life and turn >= 48 and "npc_emerged" not in past:
        new_npcs.append(dict(life))
    return {"events": events, "npc_plans": npc_plans,
            "new_scenes": new_scenes,
            "item_patches": [], "daily_bits": daily_bits,
            "new_npcs": new_npcs}


def mock_actionresolve(user: str) -> dict:
    """动作结局（确定性）：按动作关键词给结局（世界无关）。"""
    state = json.loads(user)
    action = state.get("action", "")
    if "打听" in action:
        return {"outcome": "问了一圈，各有各的说法，也多了几个疑团。"}
    if "信" in action or "等" in action:
        return {"outcome": "那件事有了新的眉目，也有了新的疑团。"}
    return {"outcome": "这件事做完了，心里的一块石头落了地。"}


def mock_npcinitiate(user: str) -> dict:
    """开口裁决（确定性）：世界状态变化驱动，优先级从上到下。

    user 为 JSON：{npc: {relationship, mood_value}, law_recent, has_rumor,
                   atmosphere}
    返回：{"open": bool, "line": str}
    """
    state = json.loads(user)
    npc = state.get("npc", {})
    suite = _SUITES[_suite_key_for_atmo(state.get("atmosphere", ""))]
    if state.get("law_recent"):
        return {"open": True, "line": "刚才的天变，你也感觉到了吗？"}
    if state.get("has_rumor"):
        return {"open": True, "line": suite["rumor_line"]}
    if npc.get("relationship", 0) >= 10:
        return {"open": True, "line": "你来得正好，我有话想对你说。"}
    if npc.get("mood_value", 0.0) <= -0.3:
        return {"open": True, "line": "……陪我坐一会儿，好吗？"}
    return {"open": False, "line": ""}


def mock_rumor(user: str) -> dict:
    """流言转述（确定性）：事件 → 带盐的传闻版本（允许走样）。"""
    state = json.loads(user)
    etype = state.get("event_type", "")
    params = state.get("params", {})
    if etype == "item_arrive":
        return {"content": f"听说「{params.get('item', '一件东西')}」出现了，"
                           f"可没人说得清是给谁的。"}
    if etype == "world_event":
        return {"content": f"都说{params.get('title', '出了怪事')}，"
                           f"数过的人信誓旦旦说自己没数错。"}
    return {"content": "城里在传一件怪事，越传越真。"}


def mock_scenegenerate(user: str) -> dict:
    """贴片生成（确定性）：按场景 id 返回套件模板描述。

    user 为 JSON：{scene: {id, name, hint}, world: {...}, neighbors: [...]}
    """
    state = json.loads(user)
    sid = state["scene"]["id"]
    desc = _ALL_SCENE_DESCS.get(sid)
    if desc:
        return {"description": desc, "atmosphere": ""}
    return {"description": f"「{state['scene']['name']}」，"
                           f"{state['scene']['hint']}。",
            "atmosphere": ""}


def mock_memoryalign(user: str) -> dict:
    """离线夹具只确认每条记忆已审阅；不靠关键词猜造世界事实。"""
    state = json.loads(user)
    return {"memories": [
        {"memory_id": memory.get("id", ""), "age_days": None,
         "duration_days": None, "embodied_as": "self",
         "accessible": True, "access_cause": "",
         "scene": None, "items": [], "current_states": []}
        for memory in state.get("memories", [])
    ]}


# ---------------- 真实 LLM（OpenAI 兼容） ----------------

def _load_config() -> dict:
    """读取项目根目录 worldledger_config.json（不提交）。"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "worldledger_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


class OpenAICompatLLM(BaseLLM):
    name = "openai-compat"

    def __init__(self):
        # 优先环境变量，其次项目根目录 worldledger_config.json（不提交）
        cfg = _load_config()
        self.api_key = (os.environ.get("WORLDLEDGER_API_KEY")
                        or str(cfg.get("api_key", "")))
        self.base = (os.environ.get("WORLDLEDGER_BASE_URL")
                     or str(cfg.get("base_url",
                                    "https://api.openai.com/v1"))).rstrip("/")
        self.model = (os.environ.get("WORLDLEDGER_MODEL")
                      or str(cfg.get("model", "gpt-4o-mini")))
        if not self.api_key:
            raise LLMError("未配置 WORLDLEDGER_API_KEY 或 worldledger_config.json")

    def chat(self, system: str, user: str) -> str:
        url = f"{self.base}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def get_llm() -> BaseLLM:
    try:
        llm = OpenAICompatLLM()
        print(f"[LLM] 已接入真实 API（{llm.model}）")
        return llm
    except (LLMError, KeyError):
        print("[LLM] 未配置 API key，使用 Mock 模式（确定性演示）")
        return MockLLM()


# ---------------- 语义检索（embedding API + 离线兜底） ----------------


def _cosine(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


class SemanticSearch:
    """语义相似度：配置了 embedding 端点就用 API（真语义），
    否则字符二元组重叠（离线兜底，确定性，零成本）。

    config 的 embedding 段（可选）：
    {"embedding": {"base_url": "...", "api_key": "...", "model": "..."}}
    """

    def __init__(self, cfg: dict | None = None):
        self._emb = dict((cfg or {}).get("embedding") or {})

    @property
    def online(self) -> bool:
        return bool(self._emb.get("api_key"))

    def similarities(self, query: str, texts: list[str]) -> list[float]:
        """query 与每个 text 的相似度。在线模式一次 API 调用取全部。"""
        if not texts:
            return []
        if self.online:
            vecs = self._embed([query] + texts)
            q, rest = vecs[0], vecs[1:]
            return [max(0.0, float(_cosine(q, v))) for v in rest]
        from .store import text_similarity  # 离线兜底：字符二元组
        return [text_similarity(query, t) for t in texts]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        base = str(self._emb.get("base_url",
                                 "https://api.openai.com/v1")).rstrip("/")
        body = {"model": self._emb.get("model", "text-embedding-3-small"),
                "input": texts}
        req = urllib.request.Request(
            f"{base}/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._emb['api_key']}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = sorted(data["data"], key=lambda x: x["index"])
        return [r["embedding"] for r in rows]


def get_semantic_search() -> "SemanticSearch":
    return SemanticSearch(_load_config())
