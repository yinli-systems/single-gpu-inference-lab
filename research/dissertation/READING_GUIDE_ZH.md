# 阅读、复核与答辩指南

这份英文研究稿约 8,800 词。它把已经落盘的机制实验、静态策略反例、各版本资格判定、完整性故障、serving 准备及历史 parity 调查串成一个研究论证。学校本年度 rubric、字数和提交要求暂未提供，因此它是可审阅的研究稿，不是最终提交认证，也不预测分数。

截至 2026-10-02，必须保留以下结论：

```text
v4.2 original dual canary = HOLD
default_promotion = false
serving_promotion = false
historical_token_divergence_resolved = false
```

## 应当如何读

先读 Abstract、第 5 章结果、第 7 章讨论和第 9 章结论，确认论文真正支持的结论。再读第 3 章机制与实现、第 4 章方法、第 6 章 serving 和历史 parity。第 8 章和本目录的 `closure_plan.json` 给出剩余工作的验收条件。

本稿最重要的研究主线是：

1. 实际 launch reservation 从 48 KiB 变成 64 KiB，在两个已暴露的长 prefix 条件下明显降低诊断 kernel 时间；仅提高 attribute ceiling，没有同样变化。短 suffix 反而变慢。因此值得研究的是条件性资源选择。
2. v3.2.2 static selector 的 fresh release 即便 selected 平均值好、operator 数值完全一致，也未通过 worst-case / controls。它是被测规则的反例，不是所有静态策略不可能有效的证明。
3. v4.2 exposed dev 的 PASS 没有延伸为 canary PASS。旧双卡 canary 仍 HOLD；后续 RTX 5090 public-path 补充 PASS 也只有它自己的已暴露开发范围。
4. 架构、缓存、生命周期和 CPU 检查有实质进展，但完整 HTTP Resource 性能、真实 Resource Graph 元数据更新、旧 2/432 因果闭环是独立要求。

## 最容易被追问的区别

| 两个概念 | 应当给出的解释 |
|---|---|
| attribute ceiling 与 launch reservation | 前者允许请求多少，后者是这次 launch 实际请求多少。CUPTI 记录实际 launch；不能由 ceiling 推断 reservation。 |
| 相同 SASS 与相同性能 | 指令一致约束了实现差异，资源、调度、host overhead 和环境仍可改变性能。 |
| 容量上界与 achieved occupancy | shared-memory 容量可以排除某些配置；没有合格的硬件 counters，就不能把它写成实际 residency。 |
| selected gain 与 whole-policy gain | selected 只含被选中的记录；真实 policy 还包含 lookup、校验和 Native fallback。 |
| candidate Native 与 pristine | 同模块内部 Native-normalised 比较无法证明 candidate package 引入没有绝对成本。 |
| geometry、fold、process、block、call | 144 folds 不是 144 个 fresh geometries；调用数不是独立 GPU 数。 |
| exact O/LSE 与 full-token parity | 前者是有限输入上的 operator 性质，后者是完整模型整个请求的值一致性。 |
| scheduler COMPLETED 与 qualification PASS | 完成只解决执行状态；仍要完整 cell、source、SASS、numerics、controls、statistics 和 archive。 |
| shell exit 0 与 Slurm TIMEOUT | shell 退出码不能覆盖 scheduler timeout。原实验 853/864 cells 仍不完整。 |
| forced-history equality 与历史 closure | 重建的同状态下相等，没有复现原来失败，也没有找到原来 first differing state。 |

P99 regret 的 public-path 定义是 `chosen_latency / available_oracle_latency - 1`。5090 补充结果的 fraction `0.00042468246096653167` 转换成 **0.042468246096653167%**。旧 v4.2 dev JSON 的 reversed regret 定义没有混入比较。1.33 倍 speedup 也不是 latency 降低 33%。

## 这轮实际交付了什么

| 文件 | 用途 |
|---|---|
| `resource_sensitive_attention.tex` | 独立、可编辑的英文 LaTeX。表格、矢量图和参考文献内嵌，已通过 Codex 内置编译器。 |
| `output/pdf/resource_sensitive_attention_research_draft.pdf` | 同一稿件的 ReportLab PDF 阅读版；与内置 LaTeX 预览是两个独立排版结果。 |
| `manuscript.txt` | 英文稿件的生成源，含章节、表格和事实模板。 |
| `claim_ledger.json` | 90 项关键事实，包含范围、源文件、JSON pointer、单位和原始值。 |
| `inputs/` | 十份原始 JSON 的完整字节快照；不改原始实验文件。 |
| `verification.json` | 输入 SHA、pointer、单位转换及 HOLD/promotion 不变量复核结果。 |
| `closure_plan.json` | 剩余工作的明确依赖、通过条件、失败处置与证据要求。 |
| `source_inventory.json` | 额外功能、CPU、serving 准备和 ownership 证据的位置与文件 SHA。 |
| `resource_intervention.svg` | 来自诊断 receipt 的可导出矢量图；不表示资格区间或 HTTP 增益。 |
| `build_inventory.json` | 稿件字数、产物 SHA 和构建范围。 |
| `native_compile_receipt.json`、`pdf_qa.json` | 内置编译结果与 PDF 排版检查记录。 |

## 如何复核与重新生成

在研究仓库根目录执行：

```sh
TASK_PY=/Users/kevin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
"$TASK_PY" research/dissertation/build_evidence.py
"$TASK_PY" research/dissertation/build_report.py
```

第一步验证已冻结的十份输入和 90 项事实。第二步验证同一账本再生成 `.tex`、PDF、SVG 与 inventory，不运行 GPU，不改 promotion，不向外部发布。字体与 Python 库使用本机 bundled runtime；换机器需提供相应字体和 reportlab。

`--freeze` 会显式替换报告输入快照，仅在决定制作新的 dated revision 时使用。默认复核冻结快照，不把将来改变的 live STATUS 当作旧稿的新事实。修改生成脚本、稿件或输入后，必须重新执行 PDF 排版 QA；LaTeX 源改变后须重新内置编译。`pdf_qa.json` 绑定具体产物 SHA，不能被用于另一版 PDF。

## 剩余工作怎样算真正完成

1. 当前 exposed complete repetition 四任务全部实际结束，完整 864 cells，所有严格双卡 gates / 独立 SASS / raw archive PASS。若失败，留存 HOLD，停止后继阶段，不补跑 missing tail 来拼成原实验。
2. 之后才冻结新 formal source 和 manifest，按 exact dual smoke → dev2 → new10 canary → original48 release → original12 stress 执行。new10 在本稿 cutoff 尚未生成。旧 consumed10 不恢复 fresh。
3. 所有 kernel stages 和 archive PASS 后执行四个 full Qwen 模型、双卡、三阶段、三 paired allocations，共 72 HTTP allocations / 24 scoped verdicts。必须实际看到 Resource、完整 tokens、有效 TTFT/TPOT/throughput/goodput、raw launch 和 telemetry。
4. 单独实现并资格验证真实 Resource Graph metadata 更新。Native Graph 固定 pointer 测试和 minimal epoch driver 的 Resource hits 都不能代替这个实验。
5. 单独调查旧 2/432：复现、捕获完整同状态、定位第一个差异、修复，再验证原 workload 的 full-token parity。复现不了时保持历史 unresolved，不能用新的零 mismatch 宣称旧 root cause 已找到。
6. 最后整理两份 upstream diff 和 repo 要求，再进入发布审阅。没有发表 PR 不影响目前本地研究稿的复核，但不能声称 upstream 已接收。

## 答辩时怎样回答“项目是不是失败了”

可以回答：项目得到一个有严格边界的资源干预机制、一个 fresh 静态策略反例、一个隔离 Native 的可运行实现，以及可以阻止不合格优化进入后继阶段的证据链。部署资格仍然 HOLD。研究问题得到了有条件的答案，而不是把没有通过的实验隐藏掉。

不要回答“已经生产安全”“一定提升多少 HTTP”“强制同 history 相等就证明旧 mismatch 无害”或“能保证 90 分以上”。学校评分、真实服务收益和原失败归因都需要各自的证据。

## 学校要求收到后再做的最终调整

补齐 actual rubric、word/page limit、deadline、引用规范、作者与 supervisor、贡献与允许的 AI assistance 声明。按这些要求压缩稿件和调整引用，但保留 HOLD、失败记录、scope 与 pending 状态。此处是提交格式的待确认项，不是暂停现有实验的理由。
