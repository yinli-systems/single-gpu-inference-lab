# 固定 capture 的 Resource Graph 元数据诊断：双卡 PASS

本轮固定 exposed 几何的最小 driver 诊断已完成；Paracloud 与本地均复核通过。它关闭了这一具体实验的功能验证，不改变原始 v4.2 双卡 canary 的 HOLD，不授权默认或 serving promotion，不闭环原始 2/432 token divergence。

| GPU | Slurm job / 终态 | GPU 单元 | 元数据 epoch | pytest | 独立 Native/Resource SASS |
|---|---|---:|---:|---|---|
| RTX4090 | 1649088 / COMPLETED0:0 | 16 | 48 | 24通过，0跳过 | 176对完全一致 |
| RTX5090 | 1649089 / COMPLETED0:0 | 16 | 48 | 24通过，0跳过 | 176对完全一致 |

32 个 GPU 单元覆盖 FP16/BF16、NHD/HND、packed/tuple、Graph1/Graph16。每个单元的同一个 captured graph 和全部 owned buffer 地址经历三次真实物理页映射与 Q/K/V 更新。96 份原始 CUDA trace 共包含3264次实际 ResourceKernel attention launch，shared memory 全部为65536字节。每个 epoch 的 Resource O/LSE 及之后的 Native 均精确匹配在 Resource 执行前建立的当前 Native 对照；物理页与输出哈希真实变化。

更新期间调用被拒绝；未通知的 inference metadata 写入，以及总 Q 相同但 ordered geometry 不同的情况，均拒绝旧 Graph 并按契约回到当前 Native。测试以禁止 replay 的毒化 Graph 对象验证不会误执行过期 capture。

## 可复核证据

冻结诊断源码：`c3f5a8cba2b382b151a9908477bf3728f60ffea7`。
普通 FlashInfer candidate package：`75544a17ce0019ca877f50354d95451ee089f859`，未改写共享 package 或 Python 环境。

完整原始归档：`raw-complete.tar.gz`，87570053字节，1994个成员。
SHA256：`16e72deb189876f23f0a0d992e132c50c945334c91e84fc06776cd97b0cb070b`。
Paracloud 原始 root：`/ssd/scxi253/sgi-graph-epochs-c3f5a8c-20261002`。

远端有限控制器完成所有作业终态、完整矩阵、原始 traces、helper/package/SDK/private pytest 依赖前后校验、独立 SASS 和逐成员归档验证。本地再次顺序检查全部1994成员，独立重解析全部96份 trace，并比对实际 ResourceKernel 名称、64KiB launch、完整32单元、O/LSE契约及实际binary/SASS哈希；见 `local-full-verification.json` 和 `verify_gpu_archive.py`。

`build-recipes.tar.gz` 另保存主归档过滤器未收录的8份原始 Ninja 构建命令与生成 C++；25成员均再次校验，含 nvcc / cuobjdump 路径、实际二进制哈希与版本输出。它只补充可重建性，不重新执行或替换任何 GPU 结果。

本地与 Paracloud CPU 预检分别13项通过；Paracloud预检的16项 CUDA skip 仅说明当时没有GPU。GPU作业每卡另执行8项CPU边界加16项GPU单元，实际均无skip。第一次缺少pytest的预检失败完整保留；它发生在任何新GPU提交之前。依赖455个文件由已有校验过的wheel测试支持目录复制到诊断私有目录。

`verify_evidence.py` 检查CPU/源码/提交字节；`verify_gpu_archive.py` 检查GPU完整归档。CPU XML、源码包损坏负例及仅使用Git已提交文件的复核均通过。

## 结果边界与继续执行的工作

这是单个固定 exposed 几何、每卡一个进程、每单元三个相邻 epoch 的功能诊断。它不提供跨进程性能置信区间、不测不同几何间泛化、不认证公共 managed Graph runner，也不覆盖真实SGLang Graph生命周期。

每次更新实际复制151245084字节，约144.24MiB。事务计时还包含元数据GPU读回、同步、Native planning、前置Native对照、payload更新和四次Graph调用。因此它含验证对照和保守全workspace复制，不能当作部署一步延迟或serving提升。后续必须独立完成实际serving所有权/lease集成、成本降低、部署边界匹配的成对计时与新鲜qualification，才能扩大支持边界。

原有四项完整重复实验1648991 /1648992 /1648993 /1649001仍由原来的有限控制器管理，冻结源码0a5c735/c34d5d9及0.99判据不变；见 `final-progress-snapshot.json`。新的诊断不消耗fresh case，不修改已消耗的原始10-case canary或尚未开放的原始48+12。

诊断节点为wqd10nba07g8 /wqd10nah09g3，实际与四项原实验不同。提交环境请求排除原实验节点，但Slurm的ExcNodeList为null，故不能声称该排除约束已获调度器确认；原始请求和实际分配均保留。

`default_promotion=false`；`serving_promotion=false`；原始canary=HOLD；原始历史2/432仍未closure。本轮PASS不能替代这些独立门槛。
