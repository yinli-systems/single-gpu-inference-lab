# Paracloud 当前执行记录

核验时间：2026-10-02T00:12:44.724473+00:00。这是当前快照，不是最终资格结果。

完整有限执行链已经启动，控制器 PID `1725873`，远端目录 `/ssd/scxi253/sgi-complete-qualification-c34d5d9-20261002`。冻结控制器源码 `c34d5d9` 已在实际 Paracloud 环境完成 **214 CPU PASS、10 GPU skip**；全部 852 个源码文件、原始 XML、日志和完整源码包已落盘并逐文件核验 SHA。

首轮正在从头完整重跑两个原有 exposed 开发 case。RTX4090 作业 `1648991 / 1648992`，RTX5090 作业 `1648993 / 1649001`，本快照全部 RUNNING，每个作业真实时限 12 小时、1 GPU、6 CPU。测量源码和协议字节保持原样，全部九个独立流程、每流程 24 单元、所有窗口和原始张量证据均保留。测量源码绑定 `0a5c735`，与控制器源码分别冻结。

自动顺序：完整双卡开发 PASS 和完整归档 → 新正式源码/manifest 绑定 → 双卡 90 smoke → dev2 → 新10 canary → 原48 release → 原12 stress → 完整 kernel 归档 PASS → 四个完整 Qwen 模型、双卡、functional/performance/parity、72 次配对 HTTP allocation → 完整 HTTP 归档。任一步 HOLD 停止后继，已提交作业继续受原定有限时限约束，所有失败保留，不续补缺失单元、不重试、不重采样、不裁剪、不降低 0.99 等门槛。

原始 public 开发轮永久保留 HOLD：4090 一个作业 TIMEOUT 0:0，缺 11 单元。完整归档 1,317,298,697 字节，4,617 个成员已独立核验 SHA。5090 补充分析保留原始数据与全部窗口，修复单 ULP 重算一致性检查和真实 NVIDIA CSV 列名后全部 16 项通过；selected geomean **1.3317×**、worst **1.1055×**。这仅是两项 exposed 5090 数据，不能解释为双卡 qualification 或完整 HTTP 增益。

目前新10 尚未生成，原48/12 尚未消耗，旧 v4.2 HOLD 与旧10已消耗事实均不改变。历史 2/432 token 分歧仍未闭环：原始分歧前状态缺失，后续等值 forced-history replay 无法替代。真实 Resource Graph metadata 更新仍需独立支持与验证；Native Graph 正确不代表 Resource Graph 已通过。两份上游 PR 尚未发布，默认和 serving promotion 均 OFF。即使未来完整 kernel/HTTP PASS，也会停在历史闭环和上游审阅要求之前。

本轮尚未全部完成，正在实际 GPU 计算。即时远端证据见 workflow `controller.json`、public campaign `receipts/controller.json` 和各作业完整原始文件；本地证据位于 `evidence/v42-complete-workflow-cpu-c34d5d9/`。
