# 固定 capture 的 Resource Graph 元数据诊断

这是新的、独立的开发诊断。它不更新原始 v4.2 双卡 canary 的 HOLD，不授权默认或 serving promotion，不闭环原始 2/432 token divergence。

## 已完成并可复核

- 冻结诊断源码 `c3f5a8cba2b382b151a9908477bf3728f60ffea7`；11 个归档成员，9 个源文件均取自该 Git commit。
- 本地 13 项 CPU 边界/归档测试通过；新增执行代码静态检查及 sbatch 语法检查通过。
- Paracloud 实际 Python3.12 / Torch2.13.0+cu130 / 普通 candidate FlashInfer75544a17 环境：13 项 CPU 测试通过，16 项 GPU 测试因不可见 CUDA 而跳过。跳过不构成 GPU 通过。
- 独立复制 455 个已校验的测试依赖文件；pytest8.4.2。共享 Python 环境及 candidate package 未安装或改写。
- 保存第一次预检缺少 pytest 的失败记录；该失败发生在任何新 GPU 提交之前。新版预检是新的源码快照。
- 本地重新核验 CPU 原始归档、每个源文件、提交回执；测试 XML 和源码包的实际损坏负例均被拒绝。

## 已实际提交的有限作业

| GPU | Slurm job | 源码 | GPU 单元 | 初始期限 |
|---|---|---|---:|---|
| RTX4090 | 1649088 | c3f5a8c | 16 | 1小时 |
| RTX5090 | 1649089 | c3f5a8c | 16 | 1小时 |

诊断 root：`/ssd/scxi253/sgi-graph-epochs-c3f5a8c-20261002`。
控制器 PID3514328；最多等待6小时，超时仅取消自己提交的两项作业。禁止重复提交、失败补尾或替换原始失败记录。只有所有作业停止后才能完整归档。

调度环境请求排除现有四项正式重复实验的节点；Slurm 的 `ExcNodeList` 回执为 null，因此不能声称排除限制已被调度器确认。实际新节点为 wqd10nba07g8 / wqd10nah09g3，与四项现有作业节点不同，当前没有共置。详见原始 `dispatch-intent.json` 和 `dispatch-snapshot.json`。

截至 `2026-10-02T01:41:12Z`，5090 的 `float16-NHD-tuple-graph1` 首个单元已完成三个 epoch；三份原始 trace 各4个实际 ResourceKernel，shared memory 均65536字节。原始 trace 和结果已独立归档、逐成员校验，单元功能检查通过。这是单个诊断单元，尚无双卡终态/SASS结论。详见 `first-cell-independent-verification.json`。

GPU 结果应以终态控制器及逐成员验证的完整原始归档为准；本文不把 RUNNING/0:0 当作成功。原始四项完整重复实验仍由原来的独立控制器管理，冻结源码和判据不变。

## 本轮必须证明的边界

两卡各覆盖 FP16/BF16、NHD/HND、packed/tuple、Graph1/Graph16。每个单元使用既有 exposed 几何，不消耗 fresh case；同一个 captured graph 和所有 owned buffer 地址必须经历三次真实物理页映射、Q/K/V 更新及原始 Native 规划。

每个 epoch 在 Resource replay 之前建立 Native 对照；Resource O/LSE 与之后的 Native 都必须完全匹配该对照。原始 CUPTI trace 必须证明实际 Resource attention launches 和65536字节 shared memory。未通知的 inference metadata 写入、更新中的调用，以及相同总 Q 下的不同 ordered geometry 都必须拒绝旧 Graph，必要时回到当前 Native。

事务计时包含 GPU 元数据读回、同步、Native planning、前置 Native 对照、payload 更新、全 workspace/metadata copy 和四次 Graph 调用。它含验证对照，不能解读为部署一步的延迟或 serving 提升，也不提供性能置信区间。

即使本轮功能全部通过，也只证明此最小 driver 的固定几何元数据转移。它不认证公共 managed Graph runner，不证明真实 SGLang Graph 生命周期、正式新鲜测试、HTTP 性能、历史 divergence 或上游接受。`default_promotion=false`、`serving_promotion=false` 始终保留。

运行 `python verify_evidence.py` 可独立复核本地 CPU、源码和提交字节完整性。GPU 终态尚未包含在此 CPU 完整性回执内。
