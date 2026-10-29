---
translation_status: translated
---

# 基准测试 { #benchmarking }

GPUStack 可以对运行中的模型实例执行基准测试。基准测试由 Worker 在专用基准测试容器镜像中执行，结果和日志存储在 Worker 上。

## 基准测试引擎 { #benchmark-engine }

每个测试点都由 [AISBench](https://github.com/AISBench/benchmark) 在 `aisbench-runner` 镜像中执行，worker 为每次测试启动一个容器。该镜像由 `deploy/build-benchmark-runner.sh` 从 `aisbench-runner/` 构建，必须存在于 worker 上，或通过 `--benchmark-image-repo` 指向可访问的镜像。

一次测试按配置的输入/输出 token 数发送请求（输出由 `ignore_eos` 固定），并发数或请求速率固定。最大吞吐与延迟 SLO 两个预设的搜索方式不变，只是每个测试点的测量方式变了。

此引擎不支持：多轮对话，以及错误数、错误率和过饱和停止条件。在引擎切换前创建的测试可能仍带有这些配置，它们会被记录到日志并忽略。

镜像为本地构建，`gpustack save-images` 不包含它（该命令只从 registry 拉取镜像）。请自行搬运到每台 worker，或推送到内网 registry 后用 `--benchmark-image-repo` 指过去。注意配置了 `--system-default-container-registry` 时默认镜像名会被改写，推送的名字要与之匹配。

## 前提条件 { #prerequisites }

- 模型实例正在运行且状态正常。

### 路由基准测试的网关地址 { #route-benchmark-gateway-address }

路由基准测试使用基准测试 Worker 可访问的 Higress 基础 URL（不含 `/v1`），按以下顺序取值：

1. 执行 Worker 所属集群的 `server_url`。
2. Server 的 `server_external_url`。
3. 创建请求的源地址，且必须由 Server 的 `trusted_hosts` 显式允许。

使用请求源地址回退时不接受通配符。仅填写主机名只允许标准 HTTP/HTTPS 端口；自定义端口必须显式填写，例如 `gateway.example:8443`。如果 URL 包含路径前缀，前置代理必须在转发到 Higress 前剥离该前缀。

目标地址在创建基准测试时固定。修改 URL 后需重新创建基准测试。进度上报继续使用 Worker 的管理地址。

## 创建基准测试 { #create-benchmark }

1. 进入 `Benchmarks` 页面。
2. 单击 `Add Benchmark`。
3. 选择实例并填写配置。
4. 单击 `Save` 按钮。

### ShareGPT 工作负载长度 { #sharegpt-workload-lengths }

对于内置 ShareGPT 数据集，可以选择设置输入 token 长度的最小值、最大值，或同时设置两者。GPUStack 会保留经分词器计算后 token 数满足指定限制的首轮提示词。输出 token 长度为可选项：设置后，所有保留的提示词都会请求相同的最大输出长度；未设置时，每条提示词使用数据集中参考答案的长度。如果没有提示词落在指定范围内，本次运行会在发送基准测试请求前失败。

## 查看基准测试结果 { #view-benchmark-results }

1. 进入 `Benchmarks` 页面。
2. 找到要查看的基准测试。
3. 单击基准测试名称，查看结果和配置快照。

## 导出基准测试结果 { #export-benchmark-results }

1. 进入 `Benchmarks` 页面。
2. 找到要导出的基准测试。
3. 选择一个或多个要导出的基准测试运行记录。
4. 单击 `Export` 按钮下载结果。

## 编辑基准测试 { #edit-benchmark }

1. 进入 `Benchmarks` 页面。
2. 找到要编辑的基准测试。
3. 单击 `Operations` 列中的 `Edit` 按钮。
4. 根据需要修改名称和描述。
5. 单击 `Save` 按钮。

## 删除基准测试 { #delete-benchmark }

1. 进入 `Benchmarks` 页面。
2. 找到要删除的基准测试。
3. 单击 `Operations` 列中的 `Delete` 按钮。
4. 确认删除。

## 使用私有 CA 的 HTTPS GPUStack Server 上的基准测试 { #benchmarks-on-an-https-server-with-a-private-ca }

进度上报和路由推理遵循 Worker 的 TLS 策略。默认启用证书验证，除非 Worker 启用了 `GPUSTACK_INSECURE_TLS` 或 `insecure_tls` 配置选项（`--insecure-tls`）。基准测试没有单独的 TLS 配置。

任一入口使用 HTTPS 且启用验证时，Worker 会将自身信任的 CA 包以文件内容的形式注入基准测试容器，包括 Worker 已导入的私有 CA（参见[额外受信任的 CA](../installation/installation.md#additional-trusted-cas)）。进度客户端使用 `--progress-ca-cert`；HTTPS 推理通过 `SSL_CERT_FILE` 使用注入的 CA 包，并启用后端验证。管理地址为 HTTP 不会阻止为 HTTPS 网关注入 CA。

如果 Worker 显式跳过验证，基准测试的两个通道都会跳过验证，且不注入 CA 包。如果需要验证 HTTPS 推理证书但无法读取 CA 包，基准测试会拒绝启动，而不是使用其他信任库。仅进度上报仍采用尽力而为的方式：上报失败会记录日志，不会丢弃基准测试测量结果。
