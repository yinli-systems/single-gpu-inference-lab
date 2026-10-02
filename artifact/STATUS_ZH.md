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
