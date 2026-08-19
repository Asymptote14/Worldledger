# WorldLedger

WorldLedger 是一个由大语言模型驱动、由确定性账本约束的文字世界原型。

它让模型提出 NPC 行动、对话和世界变化，再由一个确定性的状态层负责时间推进、引用校验、事件落账和后果保存。目标不是生成一段一次性的故事，而是维护一个可以继续运行、回放和从玩家视角读取的世界。

当前定位：研究型原型（v0）。它还不是完整物理引擎、通用视频世界模型或多 Agent 系统。

## 为什么做这个

长上下文或 RAG 可以帮助模型找回过去的文字，但“找到了相关文字”不代表世界状态仍然正确：一个人可能知道自己不该知道的秘密，一件已经被拿走的物品可能再次出现在原地，一项尚未完成的行动也可能提前产生结果。

WorldLedger 把这类问题交给状态层处理：LLM 负责理解语义和提出可能发生的事，引擎负责判断它们在当前时间、位置、知识和实体状态下能否成立。

```text
世界状态 + 角色记忆 + 玩家输入
              │
              ▼
       LLM 提出候选事件
              │
              ▼
  引用 / 时间 / 前置条件 / 知识边界校验
              │
              ▼
  原子提交人物、物品、场景与事件账本
              │
              ▼
       玩家可见切片 / 后续世界演化
```

## 一段真实运行切片

下面来自仓库中的通用雨城存档，不是手写演示稿：

```text
第1天·黄昏  你问老陈：“你在等谁？”
第1天·黄昏  老陈翻出一张泛黄的旧照片，又收进怀里。
第2天·清晨  你把伞往他那边挪了挪，问他冷不冷。
第2天·深夜  你离开广场，去了雨巷。
第3天·白昼  墙角水痕继续下淌，青苔边缘被冲出浅沟。
第4天·深夜  你回来问：“我不在的这几天，发生了什么？”
第5天·黄昏  老陈从酒馆回到广场。
```

完整玩家视角见 [`examples/rain_world_story.txt`](examples/rain_world_story.txt)，对应状态账本见 [`examples/rain_world.json`](examples/rain_world.json)。玩家不在场的后台事件不会直接泄漏到这份故事里。

## 它和普通 RAG 有什么不同

RAG 主要解决“从文本库检索相关内容”。WorldLedger 维护的是正在变化的实体状态：NPC、物品、场景、时间、事件原因以及角色各自能知道的内容。

模型负责提出可能的事件；引擎负责检查实体引用、时间和因果，并把通过的变化写入账本。这样可以支持长局回放、知识差异、状态变化和多实体事件。当前的行为丰富度仍然依赖所接入的模型，不能把这些机制等同于已经解决了自主智能。

## 快速开始

项目只使用 Python 标准库，建议 Python 3.10 或更新版本。

```powershell
python -m unittest tools.tests -q
python -m tools.demo
```

没有配置 API key 时，演示使用确定性的 `MockLLM`，不会产生费用。也可以传入自己的世界描述：

```powershell
python -m tools.demo "一座普通的海边小镇，潮湿的石阶通向旧码头"
```

也可以安装为本地包：

```powershell
python -m pip install -e .
worldledger
```

默认存档位于用户数据目录：Windows 使用 `%LOCALAPPDATA%\WorldLedger\universe.json`，Linux 和 macOS 使用 `$XDG_DATA_HOME/worldledger/universe.json` 或 `~/.local/share/worldledger/universe.json`。可以用环境变量指定其他位置：

```powershell
$env:WORLDLEDGER_SAVE_PATH="D:\worldledger-data\universe.json"
python -m worldledger.main
```

## 使用真实模型

通过环境变量配置 OpenAI 兼容接口，不要把密钥写进代码或提交配置文件：

```powershell
$env:WORLDLEDGER_API_KEY="你的密钥"
$env:WORLDLEDGER_BASE_URL="https://api.deepseek.com"
$env:WORLDLEDGER_MODEL="deepseek-chat"
python -m tools.demo
```

`worldledger_config.example.json` 只是格式示例。个人配置文件 `worldledger_config.json` 已被 `.gitignore` 忽略；不配置真实 key 时会自动回落到 Mock。

## 核心部分

- 追加式事件账本：事件携带游戏内时间、原因和相关实体。
- 世界钟与调度：世界可以在玩家离开时继续推进。
- NPC 状态与记忆：人物的经历持续累积，知识边界按角色隔离。
- 场景和物品状态：变化可以留下可回放的局部后果。
- 多实体事件：人物、物品、地点和玩家可以在同一事件中被引用和更新。
- 玩家视角导出：可以只输出玩家实际可见的事件，而不是直接展示后台账本。
- 可插拔 LLM：`MockLLM` 用于零成本测试，`OpenAICompatLLM` 用于真实模型。

## 当前验证范围

- 仓库包含 315 项本地回归测试，覆盖存档往返、引用完整性、知识边界、行动时序、记忆归属、物品转移和多实体事件等机制。
- Mock 模式是确定性的，适合验证引擎纪律和快速体验，不代表真实模型的生成质量。
- 公开雨城样例展示了一次实际长局切片，但单个样例不能证明跨模型、跨世界的普遍性能。
- 对照实验应固定模型、预算、玩家输入和评分规则；具体协议见 [`docs/EVALUATION_PROTOCOLS.md`](docs/EVALUATION_PROTOCOLS.md)。

## 示例

`examples/rain_world.json` 是一局不依赖现成作品版权角色的通用雨城存档；`examples/rain_world_story.txt` 是从同一存档整理出的玩家视角文本。它是功能示例，不是对所有世界或所有模型的性能保证。

## 目录

```text
worldledger/       状态、事件、世界生成、演化和 LLM 适配层
tools/demo.py      一键演示入口
tools/tests.py     本地回归测试
docs/              设计和评测文档
examples/          可公开阅读的最小示例
```

实验存档、请求日志和一次性脚本不属于运行时核心，默认被忽略或放在本地 `archive/` 中，不应直接作为公开数据集提交。

## 当前限制

- NPC 的主动性、语言质量和故事推进依赖模型，长局仍可能重复或停滞。
- 物品和环境后果已经进入状态与事件账本，但还不是完整的通用物理模拟。
- 多 Agent 并行裁决、三维空间和视频渲染尚未实现。
- 现有测试主要验证引擎纪律和状态一致性，不能替代跨模型、跨世界的正式评测。

## 许可证

本项目使用 [MIT License](LICENSE)。示例存档和文档中的第三方内容仍需遵守其各自的版权和许可条件。
