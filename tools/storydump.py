"""故事全文导出：把存档里的完整事件日志渲染成可读的剧情文本。

引擎真实跑过的每个事件、每句对话、每件物品变化，一条不落。
用法：python -m tools.storydump [存档路径]
      （默认 worldledger_save/troupe.json → 输出 worldledger_save/story.txt）
"""
from __future__ import annotations

import sys
from pathlib import Path

from worldledger.store import Universe, game_time


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    src = sys.argv[1] if len(sys.argv) > 1 else "worldledger_save/troupe.json"
    u = Universe.load(src)
    w = u.here
    out = Path("worldledger_save") / f"story_{w.name}.txt"
    lines = [f"世界「{w.name}」完整剧情（存档：{src}）",
             f"描述：{w.description}",
             f"氛围：{w.law_profile.atmosphere}",
             "─" * 56]
    for e in w.events:
        lines.append(f"[{e.turn}·{game_time(e.turn)}] {e.kind}｜"
                     f"{e.summary}｜因：{e.cause}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已导出 {len(w.events)} 条事件 → {out}")


if __name__ == "__main__":
    main()
