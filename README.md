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

下面来自仓库中的原创 Pilot 存档，不是手写演示稿：

```text
第1天·清晨  老周把一封密信收进值班簿下。
第1天·清晨  玩家取走信件，物品归属写入账本。
第1天·白昼  阿凛在另一处场景继续自己的日常，不知道信的内容。
第2天·清晨  一个事件同时引用人物、物品和车站。
第2天·白昼  错误场景的物品补丁被拒绝，没有留下部分状态。
第2天·黄昏  审计确认所有变化都有原因，且同种子可以复现。
```

完整 Pilot 产物见 [`examples/pilot/`](examples/pilot/)。其中包括状态账本、玩家视角和审计报告；玩家不应看到的后台事件不会直接写入玩家视角文件。

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
- 可选 State Runtime 适配器：多实体事件可以先转换为通用 `Proposal`，
  由无副作用的 `StateRuntime.prepare()` 做第二层引用和前置条件校验。
- 玩家视角导出：可以只输出玩家实际可见的事件，而不是直接展示后台账本。
- 可插拔 LLM：`MockLLM` 用于零成本测试，`OpenAICompatLLM` 用于真实模型。

## 当前验证范围

- 仓库包含 325 项本地回归测试，覆盖存档往返、引用完整性、知识边界、行动时序、记忆归属、物品转移和多实体事件等机制。
- Mock 模式是确定性的，适合验证引擎纪律和快速体验，不代表真实模型的生成质量。
- 公开 Pilot 展示了一个原创、低成本、可复现的状态校验案例，但单个样例不能证明跨模型、跨世界的普遍性能。
- 对照实验应固定模型、预算、玩家输入和评分规则；具体协议见 [`docs/EVALUATION_PROTOCOLS.md`](docs/EVALUATION_PROTOCOLS.md)。

## 示例

`examples/pilot/` 是一组不依赖现成作品角色的最小公开示例，包含账本、玩家视角和审计报告。它用于说明引擎纪律，不是对所有世界或所有模型的性能保证。

## 可复现性 Pilot

```powershell
python -m tools.pilot
```

全 Mock、零成本、无 API key，21 项断言覆盖知识边界、物品不回流、原子多实体事件、失败无部分提交、有因可追溯与同种子可复现；产物写入 `examples/pilot/`（账本 JSON、玩家视角文本、审计报告）。

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
- `state-runtime` 桥接目前是可选的增量集成，只作为通用校验器使用；
  WorldLedger 仍是唯一的状态和事件账本，不会自动替换全部旧事件。
- 现有测试主要验证引擎纪律和状态一致性，不能替代跨模型、跨世界的正式评测。

## 许可证

本项目使用 [MIT License](LICENSE)。示例存档和文档中的第三方内容仍需遵守其各自的版权和许可条件。
