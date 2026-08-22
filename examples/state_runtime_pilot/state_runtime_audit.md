# State Runtime Bridge Pilot

- 模式：MockLLM，确定性，无 API key
- WorldLedger 是唯一状态与事件账本
- StateRuntime 仅执行无副作用 `prepare()` 校验

| 检查 | 结果 | 备注 |
| --- | --- | --- |
| WorldLedger 提案归一化 | PASS | [] |
| Proposal 已创建 | PASS | entities=['item:i-pilot-car', 'player', 'scene:s-alley'] |
| StateRuntime.prepare 通过 | PASS | clock=0.0 |
| 真实 world_pulse 放行事件 | PASS | ['行驶中的车撞到玩家：车辆撞到站在巷口的玩家，车轮在湿路面留下擦痕。'] |
| WorldLedger 只提交一次 | PASS | mode=validator |
| 车辆状态改变 | PASS | 撞停，车头凹陷 |
| 玩家状态改变 | PASS | 被车撞伤 |
| 地点状态改变 | PASS |  |
| 当前状态变化后拒绝旧 Proposal | PASS | precondition failed: player.location is 's-station', expected 's-alley' |
| 拒绝时 WorldLedger 零写入 | PASS |  |
