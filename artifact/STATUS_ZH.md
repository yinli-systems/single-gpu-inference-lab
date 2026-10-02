# GraphServing／HTTP 最新进展：2026-10-02

本次更新独立于下方历史快照。**完整 Resource HTTP/SLO 资格尚未完成**；原 v4.2 双卡 canary 仍 HOLD，`default_promotion=false`、`serving_promotion=false`，原2/432 token divergence 未闭合。

已经实际完成完整 Qwen2.5-Coder-1.5B BF16 的双卡 Native HTTP 基线和独立 Graph 观察器。每张卡保留全部五工作负载×四评分块、128请求、7,936输出token；每张卡的第一次独立图证据包含2,395次真实GPU输入载入、34次实际图启动及952个Native attention kernel。RTX4090自然输出完全一致；RTX5090第一次有9/128请求、900/7,936 token位置不同，保持 `HOLD_NATURAL_TOKEN_PARITY`。只有未插桩的基线可报告描述性计时，每卡一次分配不支持独立进程置信区间或Resource收益。入口：[可复现 HTTP/SLO 页面](native-graph-http-20261002-r2/index.html)。两次重建的七个输出及manifest逐字节一致，499个原归档成员均核验。

随后两张卡的普通 Native／Native 重启对照分别通过20块、128请求、7,936 token。全局确定性对照没有评分块：4090关闭了radix cache并在缓存warmup失败，5090在DeepGEMM启动编译失败。所有首次失败及762成员原归档保留，未覆盖。这些对照不能归因或闭合首次自然差异。

**新声明的固定单请求缓存图校验已经双卡通过。** 冻结测量源码为 `f4113d9732d44ab1d6f62e29ddfaf0119876cdc6`，实际作业1651560／1651498。两臂保留radix cache、CUDA Graph、普通overlap，关闭全局确定性模式，客户端与实际服务器并发上限均为1。每卡20块、128请求、7,936输出token以及完整top-5 logprobs精确一致；独立CPU/API/GPU关联分别验证194次图启动、5,432个Native attention kernel、16,123次真实输入载入和一个保持图对象／缓冲地址不变而GPU载荷至少变化三次的捕获组。这个结论只适用于该固定调度范围；它不证明普通并发batch invariance，也不闭合原自然差异或原2/432。Resource在这些诊断中完全禁用。证明：[4090](../evidence/v42-graph-http-gated-continuation-ffcbb71/native-verdict-gpu_4090.json)、[5090](../evidence/v42-graph-http-gated-continuation-ffcbb71/native-verdict-gpu_5090.json)。

正式kernel双卡smoke各通过90项真实GPU测试及独立SASS检查，作业1651384／1651385。进入dev后，两张4090分配1651665／1651667在故障节点 `wqd10nba06g6` 以0:53终止；源验证、日志和测量均未开始。原controller保持terminal HOLD。单独补正分支保留原manifest、测量helper、analyzer、阈值、评分窗口和已开始的5090作业1651666／1651668，只补发完全未启动的4090分配1651795／1651796。原消费记录保持，未重新生成fresh cases，未重跑任何已开始的测量。调度初版使用的环境变量没有生效；后来独立审查的5c09ae6补正controller改为显式 `--exclude` 并验证实际Slurm配置，从同一四个作业继续，保留原controller源码／记录且不重置14天预算。其13项CPU约束测试在两端通过；四个真实dev分配仍在运行，尚未有dev资格结论。原HOLD及补正记录：[证据目录](../evidence/v42-formal-dev-infrastructure-correction/)。

完整HTTP推进源码 `ffcbb71e94b4ee7b8a5bbf68fafd7df1aeccee24` 在Paracloud通过192项CPU测试、1项跳过，并已实际启动有限controller。它已独立确认双卡固定调度图校验，正在等待补正正式kernel的dev/canary/release/stress八个独立verdict。只有全部通过，才允许四个完整模型×两卡×三个阶段×三次独立配对分配，共72次HTTP分配、24个阶段结论。普通并发的完整自然token一致性仍是硬门槛，另加固定调度完整token/top-5 logprob校验；吞吐、严格SLO goodput、TTFT／TPOT的p50／p95／p99全部重算，40个metric点及原joint LCB门槛不放宽。原始错误、超时、缓存命中、资源决策、CUPTI、SASS和遥测均保留。任一失败停止后继实验。当前 `full_http_qualified=false`，没有宣称Resource serving／captured Resource Graph收益。

实际服务端根目录：

- 完整第一轮Native基线：`/ssd/scxi253/sgi-native-graph-serving-51be73c-r1-20261002`
- 新固定调度图校验：`/ssd/scxi253/sgi-cached-graph-parity-f4113d9-20261002`
- 正式dev基础设施补正：`/ssd/scxi253/sgi-formal-dev-unstarted-correction-fd8d3e9-20261002`
- 从补正kernel到完整HTTP的有限流程：`/ssd/scxi253/sgi-graph-http-after-dev-infrastructure-correction-ffcbb71-fd8d3e9-20261002`

以下为早期冻结快照，原文保留；它不代表上述实验后的最新状态。

---

# 本轮系统研究产物与证据边界

本轮源代码提交：`8ef76bbc7354f1ae7f14de76648c8fabcfbbf597`。Paracloud 独立重建目录：`/ssd/scxi253/sgi-system-artifact-b9cfd557-20261002`。

完成了 `make reproduce-v42-paper`，可从保留的真实原始归档重建六张 SVG/PNG 图、三张 CSV/HTML 表和完整 JSON 分析；第六张图与第三张表明确展示 serving 证据缺口，不包含虚构的 SLO 性能。原始负面结果和失败归档完整保留。

实现了按实际进程与完整平衡块进行的分层 bootstrap，保留固定的镜像窗口；补充 median/mean/P90/P95/P99/worst 和超过 0.5%/1%/5% 的 regret 统计，分别报告现有 available-arm excess、有符号独立计时差异、冻结候选标签的 oracle 查表和实际 Native+Cap 候选集合。实现了离线精确风险上界与训练数据限定的结构先验原型。它们不生成可执行 Resource 证书。

关键发现：原始完整 RTX5090 补充数据共有 144 个固定评分记录，其中 36 个没有实际测到 Native+Cap 两个候选。现有 available-arm excess 的 P99 为 0.0424682461%，不能解释为全候选 oracle regret 保证。现有两个几何组不足以验证结构先验泛化，因此所有模式报告 `INSUFFICIENT_GEOMETRY_SUPPORT`；没有捏造拟合概率或剪枝收益。

严格风险目标也未被支持。以一个完整进程中出现任一平衡块比 Native 慢超过 1% 为事件，三个零失败进程对应的 95% 单侧上界仍约 63.16%；五个为约 45.07%。在固定环境、独立可交换进程的假设下，上界严格低于 0.1% 需要 2,995 个真实零失败进程；20,000 次 bootstrap 不能代替真实试验。这不是生产请求层面的概率保证。

验证：八项有意义的统计/隔离约束测试通过；原始缓存字节变更和冻结 authority 输入变更均被拒绝。同一环境两次重建所有 19 个输出文件逐字节一致。Paracloud CPU 环境使用隔离的锁定依赖，一键命令真实成功，全部源文件和输出哈希复核通过；两端六张 SVG 字节一致、PNG 解码像素一致，1,604 个浮点值的最大差异为 2.22e-16。完整服务器产物归档 23 个成员均已独立核验。

**仍未完成：** 原始 v4.2 双卡 canary 仍是 HOLD；`default_promotion=false`、`serving_promotion=false`；原始 2/432 token divergence 未闭合。GraphStep/GraphServing、32/128 device connections、H100/B200、真实 serving trace、完整多基线与严格 SLO/Pareto 比较仍需要各自的真实实验。先验目前是结构特征 logistic 原型，尚不是已验证的解析 occupancy/cache 模型。新增分层区间是回顾性的边际 95% 区间，不是正式同时置信资格下界。

截至服务器快照 2026-10-02 02:39:17 UTC，四个完整资格作业 1648991/1648992/1648993/1649001 仍 RUNNING，有限 controller 为 EXPOSED_COMPLETE_REPETITION、terminal=false。其原始三进程协议、阈值、样本和推进顺序保持冻结；没有将新建议的五进程协议偷偷替换进去。完整 HTTP 尚未取得资格结果。此次 CPU 重建未派发新 GPU 作业。

入口：[证据图表](persisted-reproduction/index.html)、[方法与一键命令](README.md)、[跨平台复核](persisted-reproduction/cross-platform-reproduction.json)、[独立核验程序](verify_reproduction.py)。
