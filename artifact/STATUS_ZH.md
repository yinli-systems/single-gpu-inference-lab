# 最新终态：2026-10-03 00:00 Dubai，作业全部结束并归档，正式资格 HOLD

最后4090作业1651796于10月2日23:43:01 Dubai完成0:0，实际8小时23分03秒，9阶段×24 cells完整。四份dev最终为2份完整完成、2份部分测量后的Slurm启动超时失败；当前本项目GPU占用0。失败1651795缺policy-1/2，失败1651666缺policy-2，合计33/36阶段、792/864 cells已保留，不能计作正式dev PASS。

两个正式归档均已保存本地与Paracloud：原分支SHA256 `71ab1ea16d502e91cb7275c1a25acee942b47123a7b33b9701520bbf80299f80`（3083成员）；补正分支SHA256 `3afa3e6643c06d2f7afc554fc431147139f765746b03c6146be78c309c738037`（2966成员）。共86400条已完成窗口记录核验顺序、时长、记录标志及哈希，32个reference tensor文件哈希保留，两端独立审计逐字节一致。未重新执行数值比较或性能资格门槛；归档成功不能转换为资格PASS。

HTTP controller于23:44:21 Dubai归档后终态`HTTP_SCHEDULER_AMENDMENT_HOLD`，Graph此前已HOLD，所有本链controller已退出。Graph0/8、HTTP0/72、canary/release/stress均未启动。双卡Graph组件PASS保留；完整模型Graph、HTTP/SLO及正式后续目标未完成。控制器终态、源绑定和零HTTP分配证据也已单独归档，SHA256 `265a10cd8c60a965a2898c675b61d805183063ce07b7468bd0efa79aff9b76ed`。

原冻结链在现有“不重跑已开始测量、不跳门槛”约束下无法自动继续。任何新测量或资格规则变更需独立声明的补正协议；原HOLD与全部失败保留。历史2/432及普通Native/observer差异未closure，default/serving promotion均OFF。Heartbeat不宣称成功完成，也不自动重启实验。

[终态汇总与证据](/Users/kevin/Documents/ChatGPT/inference/graph-serving-continuation-20261002/evidence/terminal-summary.json)

---

# 最新核验：2026-10-02 23:00 Dubai，5090完整测量结束，剩余一个4090作业

5090 dev 1651668以0:0完成，实际7小时39分51秒，9阶段×24 cells完整，执行前后源校验通过。另一个5090作业1651666的8阶段结果及policy-2启动超时失败全部保留。旧HOLD分支controller3592786已完成归档并退出；其归档SHA256为`71ab1ea16d502e91cb7275c1a25acee942b47123a7b33b9701520bbf80299f80`，222002448字节、3083文件，已保存本地和Paracloud。

离线复核两份5090数据的408个完整cells、44928条原始窗口记录、16个reference tensor文件哈希及源绑定。核验覆盖窗口顺序、时长、记录一致性和exact/binding标志，未重新执行tensor数值比较或性能资格门槛；此结果不能转换为正式PASS。

当前仅4090作业1651796仍运行，补正分支HTTP archiver仍等待其结束。正式与Graph保持HOLD，Graph0/8、HTTP0/72，后续canary/release/stress未启动；default/serving promotion OFF。下方多作业运行描述是旧快照。

[归档独立核验](/Users/kevin/Documents/ChatGPT/inference/graph-serving-continuation-20261002/evidence/original-formal-hold-archive/independent-verification.json)

---

# 最新核验：2026-10-02 22:00 Dubai，两次 Slurm 步骤启动失败

5090 dev 1651666于21:48:35 Dubai以1:0退出：policy-2启动前srun确认allocation通信超时，与此前4090作业1651795直接错误相同。该5090作业此前8个GPU子步骤均0:0，8个阶段各24 cells完整，源校验全部通过；最后policy-2未启动。36文件诊断归档已在远端保存并逐文件核验；逐cell原始数据和缓存仍待全正式链终态归档，不能宣称全链归档完成。

剩余1651668（5090）及1651796（4090）仍运行。正式与Graph维持HOLD；Graph0/8，HTTP0/72，HTTP controller继续等待已登记作业结束并归档。两次失败不能据此宣称GPU数值错误，也不能视作资格PASS。不重跑已开始测量、不跳门槛，default/serving promotion OFF。

[最新实核记录](/Users/kevin/Documents/ChatGPT/inference/graph-serving-continuation-20261002/evidence/heartbeat/20261002T175937Z-review.json)

---

# 最新异常：2026-10-02 21:30 Dubai，正式链 HOLD

4090 dev 作业1651795在21:28:33 Dubai以1:0退出。直接日志为`Unable to confirm allocation ... Socket timed out on send/recv operation`：policy-1的srun未成功启动。此前7个GPU子步骤均0:0，pristine×3、train×3、policy-0均有完整结果；policy-1/2未测。源校验无失败行。已定位为部分测量后的调度启动失败，尚未证明更底层物理原因，不能冒充数值失败或科学PASS，也不能按完全未启动分配直接补发。

正式controller于17:28:40 UTC终态HOLD，Graph因正式门槛于17:28:48 UTC终态HOLD，0/8，两个controller均已退出。HTTP仍0/72，现有controller等待其余已登记作业结束及失败归档，原WAIT标签不再代表有资格继续。三个dev作业1651666、1651668、1651796仍运行，不停止健康测量；全链归档待其终态。双卡Graph组件PASS保留。

[故障证据及收尾计划](/Users/kevin/Documents/ChatGPT/inference/graph-serving-continuation-20261002/evidence/hold-1651795/incident.json)。不自动重试、不改阈值、不跳过门槛；历史2/432未closure，default/serving promotion均OFF。下方运行状态均为早期快照，以本节及最新证据索引为准。

---

# 最新核验：2026-10-02 21:00 Dubai，Resource Graph 双卡组件 PASS

5090 组件作业 1652167 已 COMPLETED 0:0，实际 4分37秒；与此前4090组成 DUAL_COMPONENT_PASS。组件 watcher 完成归档后正常退出。独立离线重算46个归档成员哈希、每卡3次 CPU/API/GPU 关联的真实 Resource Graph kernel（64KiB）、3组页面/Q/K/V/output变化及过期拒绝后零Resource kernel。精确O/LSE由绑定GPU测试断言及记录支持；未把组件资格当作完整模型或HTTP资格。

原始双卡归档 SHA256：`9d20886b3c3c231f9bc57ddd45e920fad8f5ce823f6a976c202400fbd27cbc18`。两卡各339个helper文件、10597个Native文件的执行前后校验均通过。原始归档已保存本地与Paracloud，独立复核程序为`verify_dual_component_archive.py`。

Graph控制器已自动转为 `WAIT_ALL_EIGHT_FORMAL_KERNEL_VERDICTS`；Graph仍0/8，HTTP仍0/72。四个dev作业仍运行，canary/release/stress尚未启动。原2/432未closure，default/serving promotion均OFF。下方早期“5090排队”信息由本节更新。

[双卡组件独立核验](/Users/kevin/Documents/ChatGPT/inference/graph-serving-continuation-20261002/evidence/dual-component/independent-verification.json)

---

# 后续链已启动：2026-10-02 17:52 Dubai 实核

本节优先于下方旧快照。正式 dev 四作业 1651666／1651668／1651795／1651796 均 RUNNING；fresh canary 0/10、release 0/48、stress 0/12，继续沿原冻结门槛推进。

独立 Resource Graph 组件作业 1652166（4090）已 PASS，3 个真实 page/Q/K/V 更新 epoch、精确 O/LSE、过期物理元数据拒绝及拒绝后零 Resource kernel 均经原始 trace 复核。5090 作业 1652167 仍 PENDING Priority。组件 PASS 不是完整模型 GraphServing PASS。

完整模型 Resource Graph 新有限链已经启动，PID 2916963，根目录 `/ssd/scxi253/sgi-resource-full-graph-chain-4444210-20261002`。源码归档 SHA256 `44442103256712c43becd4e7f680b7b2cc7483bdfab34622885b9d4dbf19206c`；41 项相关 CPU 测试在两端通过，绑定 7,885 个文件。它等待双卡组件和全部八个正式 kernel verdict；随后最多四模型×两卡=8 次、每次2小时，每模型双卡通过才进入下一模型。现为 WAIT_COMPONENT_AND_FORMAL_GATES，0/8 已启动。固定128-token正例、变长/缓存前缀回退、原始HTTP全token/top-5、每层精确O/LSE、CPU/API/GPU真实Resource Graph关联均要求通过。诊断含额外Native计算和读回，计时无效，不能证明Graph性能。

72次HTTP接续器完成独立调度修正，PID 3241029，根目录 `/ssd/scxi253/sgi-http-scheduler-amendment-0f2831a-20261002`；只把未来sbatch改为显式排除故障节点，并独立验证Slurm配置。5项调度测试两端PASS。冻结测量包仍为 `54dc61eb128cf46e32f2f0a8fe92719c7d3339cca481db4e95fa0e9aecf87a9b`，原30天截止时间不变。原等待PID136377已停止；没有取消测量作业或重试。另一个PID3592786绑定旧终态HOLD分支，冻结代码和真实authorize拒绝已核验，仅保留旧归档工作，不能派发重复Resource HTTP。当前等待八个正式verdict，0/72启动。72次协议依然仅支持eager Resource。

候选覆盖缺口已用84份原始文件逐一核验：4090缺失3条均为duplicate-controls证书拒绝；5090缺失36条全部为Native split-KV，而当前Resource `_plan_support` 明确拒绝该计划。后者需要新实现和新资格；不能强行执行旧runner、补造原始候选计时或追改门槛。全候选P99仍未知。历史全部432个候选请求也重新核验，2次差异仍在（decode-11/output74、decode-13/output114）；原始分歧前状态缺失，未closure。

完整模型Graph功能和性能、完整HTTP资格、split-KV新增支持以及历史归因仍未完成。`default_promotion=false`、`serving_promotion=false`、原v4.2 HOLD均保持。

[本轮状态及可复核证据索引](/Users/kevin/Documents/ChatGPT/inference/graph-serving-continuation-20261002/index.html)

---

以下保留前一轮状态和历史快照，控制器位置以本节为准。

# GraphServing／HTTP 最新进展：2026-10-02

本次更新独立于下方历史快照。**完整 Resource HTTP/SLO 资格尚未完成**；原 v4.2 双卡 canary 仍 HOLD，`default_promotion=false`、`serving_promotion=false`，原2/432 token divergence 未闭合。

**Public-path validator 与 v4.3 冻结已经推进。** 合法`references/`目录误判已由独立修复与原始归档复核关闭；四个既有作业全部完成，双卡通过原冻结开发门槛，4,424个原始测量成员未变。正式`4.3.0`测量源码为`250d98c`，frozen binding为`1092ff04b13a96a1a2b3e17a71acfd6e2e367cc8e56e12f8c207ff227acc596f`，绑定已接受development receipt `fde29802002dd9c51410d07bb4f9e864027a46d0f6c1d8d82e2ef2032149f5d1`。新10个canary已生成冻结、尚未执行；release48/stress12尚未执行。12小时是分配时限，原四个作业实际约7小时14分至7小时57分。

**新增CPU离线诊断：[4090 tail regret原始证据报告](tail-regret-audit-20261002-r2/index.html)。** 两个既有归档SHA及全部使用文件逐一核验，双卡288条记录全部重算。4090的三个高regret记录来自两个已暴露case，全部是eager模式Native回退；每个都由一个训练证书的Native重复控制区间上限超过1.005触发（分别1.00564924、1.00571052、1.00540446），证书及managed Resource资格因此缺失。完整调用收益均值和下界检查本身通过。这定位了直接决策原因，尚未证明其物理原因，也未获得新的优化收益。两卡可用oracle实际包含Resource的记录分别为141/144、108/144；低available-arm regret不能当作全候选最优性保证。更强P99目标需要独立声明候选覆盖与统计单位，不能追改当前冻结门槛。本次诊断未派发GPU、未消费fresh case。

**Resource GraphServing仍有独立缺口。** 当前72次HTTP协议允许eager Resource与Native捕获图共存，其审计明确拒绝captured Resource kernel。因此该HTTP流程即使PASS，也不能自动证明真实SGLang捕获图内的Resource执行；后者仍需要单独集成与资格证据。

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
