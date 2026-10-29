---
translation_status: translated
---

# 缓存服务管理 { #cache-service-management }

缓存服务提供共享 KV 缓存，模型部署可以接入它，而无需各自独立保留 KV 缓存。缓存位于推理引擎之外，因此同一 Worker 上的部署和副本能够复用彼此的前缀；L2 存储后端还可将这种复用扩展到重启后和跨节点场景。

缓存服务仅支持内置 vLLM 和 SGLang 后端，部署只能接入所在集群内的缓存服务。

## 创建缓存服务 { #create-a-cache-service }

1. 进入 `Cache Service` 页面。
2. 单击 `Add Cache Service` 按钮并选择提供商。
3. 输入 `Name`。
4. 选择 `Version`，或选择 `Custom` 并输入 `Container Image`。
5. 填写提供商设置，例如 `RAM Size (GiB)`，即缓存服务器内存中的缓存容量。
6. _（可选）_ 设置 `Worker Label Selector`，限制运行该服务的 Worker。
7. _（可选）_ 添加 `L2 Storage Backend`，将缓存溢写到更大的存储层（本地目录、Redis 或 Valkey，或兼容 S3 的对象存储）。多个条目按顺序确定优先级：读取优先使用第一个条目，写入则会写入所有条目。
8. _（可选）_ 在 `Advanced` 中添加 `Parameters` 和 `Environment Variables`。如果服务运行多个组件，每个组件都保留独立的参数列表，因为各组件运行不同的二进制程序。
9. 单击 `Save` 按钮。

!!! tip

    对现有缓存服务的更改，只有在删除并重新创建其实例后才会生效。

## 接入部署 { #attach-a-deployment }

1. 使用内置 vLLM 或 SGLang 后端创建或编辑模型部署。
2. 选择 `Enable Extended KV Cache`。
3. 将 `Cache Backend` 设置为目标缓存服务。列表只显示同一集群内且与所选后端兼容的缓存服务。
4. 单击 `Save` 按钮。

每个实例都会报告其接入的缓存服务及近期命中率。如果实例启动时没有接入缓存，例如其所在 Worker 上没有可用缓存实例，它会报告共享 KV 缓存未激活，而不会因此失败。

## 使用 LMCache 运行混合注意力模型 { #running-hybrid-attention-models-with-lmcache }

混合模型将循环层（Mamba 或 Gated-DeltaNet（GDN）线性注意力）与全注意力层交错排列。Qwen3.5 和 Qwen3.6 系列、Qwen3-Next、Kimi-Linear 及 Kimi K3 都属于此类模型。

循环层保存固定大小的状态，而非为每个 token 保存一对键值，并且只能在块结束处创建状态快照。因此，引擎会根据模型本身确定块大小，其大小远超过通常的少量 token；只有 LMCache 的缓存存储单元与这些块边界一致时，才能为模型提供缓存服务。双方都不会自动完成这种协调：使用默认设置时，引擎要么拒绝启动，要么虽然接入缓存却始终没有命中。

### 步骤 1：确定引擎选择的块大小 { #step-1-find-the-block-size-the-engine-chose }

vLLM 会增大注意力块大小，直到注意力页至少与循环状态页一样大，并在部署启动时记录结果：

```
Setting attention block size to 544 tokens to ensure that attention page size is >= mamba page size.
```

该数值即为 `N`。它由模型、数据类型和并行方式共同决定，因此请从部署自身的日志中读取，不要自行假定。下表数值来自使用模型自身精度的单 GPU 运行，可用于检查所观察到的数值是否合理，而不应作为直接查用的配置表：

| 模型                                     | `N` |
| ----------------------------------------- | --- |
| `Qwen/Qwen3.5-0.8B`                       | 544 |
| `Qwen/Qwen3.6-27B`                        | 784 |
| `moonshotai/Kimi-Linear-48B-A3B-Instruct` | 944 |
| `moonshotai/Kimi-K3`                      | 768 |

### 步骤 2：配置缓存服务 { #step-2-configure-the-cache-service }

- 将 `Chunk Size` 设置为 `N` 或其倍数。缓存服务器的 chunk 大小必须是引擎块大小的倍数，否则缓存注册会失败。
- 在缓存服务器的 `Parameters` 中添加 `--separate-object-groups`，使每个注意力窗口存储为独立的对象组。缓存服务器默认关闭此选项。

!!! note

    chunk 大小属于缓存服务配置，而不是部署配置。按某个 `N` 调整后的服务仍可用于块大小能整除它的普通模型，但两个 `N` 不同的混合模型需要各自独立的缓存服务。

### 步骤 3：将模型部署到该服务 { #step-3-deploy-the-model-against-the-service }

按照[接入部署](#attach-a-deployment)中的说明部署模型，将步骤 2 中的缓存服务选为其 `Cache Backend`。运行中的实例会报告已接入的服务和当前命中率。

### 参考资料 { #reference }

[混合注意力模型](https://docs.lmcache.ai/mp/hybrid_models.html)——LMCache 官方对此要求的说明，以及缓存服务器为此提供的参数。

## 在昇腾上使用 MemCache { #using-memcache-on-ascend }

MemCache 是昇腾基于 MemFabric 构建的分布式 KV Cache 池。它与 LMCache 的区别在于缓存所在的位置：GPUStack 只运行一个负责索引缓存池的 `MetaService`，每个 vLLM-Ascend 引擎在自己的进程内贡献主机内存，并自行在 NPU 内存与缓存池之间搬运 KV 块。它仅适用于 vLLM-Ascend 部署，通过 vllm-ascend 的 `AscendStoreConnector` 接入。

使用要求：

- vLLM-Ascend 0.23.0 及以上，运行在 Ascend A2、A3 或 950PR/950DT 的 worker 上。
- 无需选择镜像。MemCache 随 vLLM-Ascend 镜像一同发布（`memcache_hybrid` 与 `memfabric_hybrid`），因此缓存服务的`版本`就是 vLLM-Ascend 的发布版本，`MetaService` 会运行该版本对应 worker 昇腾型号的镜像，镜像仓库与 vLLM-Ascend 后端一致（`GPUSTACK_VLLM_ASCEND_IMAGE_REPOSITORY`）。仓库中发布的 0.23.0 及以后的版本都可选，在服务端启动或提供商来源变更时读取；0.23.0 始终在列表中，仓库不可达时仍可用。请选择部署所用的版本。
- 按所选传输方式完成宿主机准备，参见 [vllm-ascend KV 池指南](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/kv_cache_pool_ascend_store.html)与 MemCache 的 [A2 PD 分离最佳实践](https://gitcode.com/Ascend/memcache/wiki/MMC%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5%E2%80%94DSV4-Flash+A2-PD%E5%88%86%E7%A6%BB.md)：A2 使用 `device_rdma` 时，NPU 驱动固件不低于 25.5.1，OS 页大小为 4K（`getconf PAGESIZE`，64K 无法使用 MemCache），并需要 `/etc/hccn.conf`；A3 上使用 `device_sdma` 需要灵衢计算网络 1.5 及以上。静态大页（`vm.nr_hugepages`）适用于主机内存类协议，`device_rdma` 不需要。推荐使用 MemCache 1.1.5。

按[创建缓存服务](#create-a-cache-service)的步骤创建，并设置：

- `单卡内存贡献（GiB）`：每个引擎进程为其使用的每张卡贡献的主机内存。缓存池容量为该值乘以所有接入引擎的卡数。这部分内存由引擎自行占用，调度器不会为其预留，请为系统、引擎和这部分内存在每个 worker 上留足空闲内存。
- `传输协议`：A2 使用 `device_rdma`，A3 经 HCCS 使用 `device_sdma`，950PR/950DT 使用 `device_urma` 或 `device_uboe`。同一服务的所有 worker 共用该选择。
- `配置存储端口`：运行 `MetaService` 的 worker 上的空闲端口，引擎加入缓存池时在此汇合。

按[接入部署](#attach-a-deployment)的步骤接入。在 vLLM-Ascend 的 PD 模式下，MemCache 在两个角色上均被接受，prefill 为生产者，decode 为消费者。

!!! note

    逐层传输、SSD 层以及将缓存池本地服务作为独立进程运行暂不提供。所有共享同一缓存池的引擎必须使用相同的 `PYTHONHASHSEED`，GPUStack 会注入 `0`。

## 自定义缓存提供商 { #customize-cache-providers }

可选择的提供商包括当前版本内置的提供商，以及已安装扩展添加的提供商。平台管理员可以用自己的文档替换整个集合，以固定其他镜像、添加版本，或声明 GPUStack 未内置的提供商。

1. 进入 `Cache Service` 页面。
2. 单击 `Manage Providers` 按钮。
3. 在 `Provider Source` 抽屉中选择以下三种来源之一：

    - `Yaml File`：将内容粘贴到编辑器，或单击 `Import` 将文件加载到编辑器。
    - `URL`：填写 `Source URL`。GPUStack Server 会在保存时及单击 `Update Now` 时获取此文件。URL 必须使用 `http(s)`，不能包含凭据，必须直接指向文件本身而非仓库页面，且文件必须采用 UTF-8 编码，大小不超过 4 MB。
    - `Embedded`：恢复使用当前版本内置的提供商。

4. 单击 `Save` 按钮。

!!! warning

    **请从内置文件开始修改，不要从空白编辑器开始。** 自定义文档会直接替换整个集合，未在其中声明的所有提供商都会停止提供服务，包括已安装扩展提供的提供商。单击编辑器顶部的 `Built-in File` 下载当前安装实际使用的内容，然后在此基础上编辑。

保存时会验证文档。如果其中任何内容无法正常使用，例如当前版本无法读取的声明、会被原样呈现的占位符，或不存在的字段名，整个文档都会被拒绝。最后一种情况尤其需要注意：未知键通常会被直接忽略。例如将组件的 `run_command` 误写为 `run_cmd`，若不验证，该组件就会在没有启动命令的情况下启动。

如果保存会移除现有缓存服务正在使用的提供商或版本，同样会被拒绝。提示消息会列出所有受影响的服务。请先删除这些服务，或将它们迁移到新文档包含的配置上。

!!! note

    运行中的实例会继续使用启动时的声明。新声明在实例重新创建时生效，与 GPUStack 升级时提供商变更的生效方式相同。

    提供商不会按计划自动更新：URL 来源仅在保存时及单击 `Update Now` 时读取。配置自定义文档后，GPUStack 升级或新安装的扩展所添加的提供商，只有在手动合并到文档后才会出现；可以再次下载内置文件查看新增内容。
