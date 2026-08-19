# Tools

这些脚本都不是引擎核心，主要用于演示、回归测试、存档导出和研究评测。

## 推荐入口

- `python -m tools.demo`：无 API key 也能运行的 Mock 演示。
- `python -m tools.tests -q`：本地回归测试。
- `python -m tools.finalstory <save> <output>`：从存档导出玩家视角文本。
- `python -m tools.storydump <save>`：导出完整事件故事。
- `python -m tools.play200 [turns] [description]`：参与式长局脚本，可能调用真实模型。

## 其他脚本

`bench.py`、`showcase.py`、`chain_test.py`、`knowledge_test.py`、`playtest.py` 和 `storyclean.py` 是基准、机制展示或存档处理工具。一次性实验脚本和历史评测脚本已移入本地 `archive/`，不属于公开运行入口。

除 `tools.tests` 和明确标注的 Mock 演示外，脚本的输出不是稳定 API，也不保证跨模型完全复现。需要真实模型时，使用 README 中的环境变量配置，并注意调用费用。
