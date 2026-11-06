# 部署本仓库(vLLM-Ascend 定制版 GPUStack)

本仓库 = GPUStack v2.3.0 的 fork,包含:

- `gpustack/`:后端,新增独立的 **vLLM-Ascend** 后端,移除了"模型提供商"。
- `ui/`:前端(原 `gpustack-ui` 仓库并入,保留了历史)。
- `aisbench-runner/`:基准测试的执行端,每个测试点由 aisbench 实际压测(替代官方基于 guidellm 的 benchmark-runner)。
- `deploy/`:镜像构建和部署说明(本目录)。

镜像是在官方 `gpustack/gpustack:v2.3.0` 之上叠加本仓库的后端源码和前端构建产物,
所以官方镜像里自带的运行时、工具、文档站、Helm chart、Python 依赖都原样沿用。

## 构建

在**有外网**的机器上(需要访问 Docker Hub 和 npm):

```bash
deploy/build.sh                      # 产出 gpustack-custom:v2.3.0-<提交号>
deploy/build.sh my/tag:1 --build-arg NPM_REGISTRY=https://registry.npmmirror.com
```

构建会在以下情况直接失败(而不是等服务启动后才发现):前端没构建出来、代码里有
解析不了的 import(包括写在函数内部的)、目录图标或 Helm chart 缺失、版本戳丢失。

构建结束后 `deploy/build.sh` 还会逐个核对镜像自带的
`GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS`:worker 靠这些
`org.opencontainers.image.*` 标签找到自己的容器,再把运行时和设备透传给推理容器,
其中任何一个被改写都会让推理容器没有 GPU(表现为 "Failed to infer device type"),
所以这里不一致就直接失败,而不是等部署后排查。fork 的提交号单独记在自己的标签上,
官方标签保持基础镜像的值:

```bash
docker inspect -f '{{index .Config.Labels "io.github.shijunan126-creator.gpustack.commit"}}' gpustack-custom:<tag>
```

内网/离线的生产机:在联网机器上构建后搬运镜像。

```bash
docker save gpustack-custom:<tag> | gzip > gpustack-custom.tar.gz
# 拷到生产机后
gunzip -c gpustack-custom.tar.gz | docker load
```

### 基准测试执行端镜像

"基准测试"页面的每次测试由 worker 另起一个容器执行,默认镜像是
`aisbench-runner:<版本>`(版本见 `gpustack/__init__.py` 的 `__aisbench_runner_version__`)。
它基于华为的 aisbench 镜像,并内置 ShareGPT 数据集。基础镜像有 amd64 / arm64 两种架构(构建时用
`--platform` 选),以及两个系统版本:

- `swr.cn-south-1.myhuaweicloud.com/ascendhub/aisbench_benchmark:v3.1-20260630-master-ubuntu22.04-py310`(默认,已验证)
- `swr.cn-south-1.myhuaweicloud.com/ascendhub/aisbench_benchmark:v3.1-20260630-master-openeuler24.03-py311`(未验证,用 `--build-arg AISBENCH_IMAGE=...` 指定)


**默认由 worker 自动构建,不用手动处理。** GPUStack 镜像里带了这个镜像的构建上下文
(`/usr/local/share/gpustack/benchmark-runner`),worker 发现本机没有 runner 镜像时,
会通过挂进来的 `/var/run/docker.sock` 在本机构建:

- worker 启动时在后台构建一次,构建的是本机架构,鲲鹏机器上就是 arm64;
- 构建期间创建的测试会保持"排队中",并提示正在等待镜像构建,构建完成后自动开始;
- 构建失败时,测试会失败并显示构建报错;下一次测试会自动重试构建;
- 构建输出写在 worker 的 `<log-dir>/benchmarks/runner-image-build.log`
  (容器内默认是 `/var/lib/gpustack/log/benchmarks/runner-image-build.log`)。

构建需要从华为 SWR 拉基础镜像,并从 huggingface.co 下载 ShareGPT 数据集。访问不到时,
可以给 worker 设环境变量换成镜像源:

```bash
-e GPUSTACK_BENCHMARK_RUNNER_SHAREGPT_URL=https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
-e GPUSTACK_BENCHMARK_RUNNER_BASE_IMAGE=<内网 registry>/aisbench_benchmark:<tag>
```

以下两种情况 worker 不会自动构建,沿用原来的手动方式:

- 用 `--benchmark-image-repo` 指定了自己的镜像;
- 设了 `GPUSTACK_BENCHMARK_RUNNER_AUTO_BUILD=false`,适用于完全离线的 worker。

手动构建的命令:

```bash
deploy/build-benchmark-runner.sh                               # 产出 aisbench-runner:<版本>
deploy/build-benchmark-runner.sh --platform linux/arm64        # 鲲鹏机器
deploy/build-benchmark-runner.sh --build-arg SHAREGPT_URL=https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

手动构建后需要搬到每台 worker 上:`docker save` / `docker load`,或者推到内网 Registry,
再用 worker 的 `--benchmark-image-repo <registry>/aisbench-runner:<版本>` 指向它。
`gpustack save-images` / `copy-images` **不包含这个镜像**:它们只从 registry 拉取
(`skopeo copy docker://…`),把本地构建的镜像列进去会导致整次 save 失败。

还有两处容易踩坑的地方:

- 配了 `--system-default-container-registry` 时,worker 会把裸名改写成
  `<registry>/aisbench-runner:<版本>`。自动构建会直接用改写后的名字打 tag;
  手动搬运时,只 `docker load` 裸名是对不上的,要么按改写后的名字 `docker tag`,
  要么显式设 `--benchmark-image-repo` 指向实际名字。
- amd64 和 arm64 用的是同一个 tag,两种架构推到同一个 registry tag 会互相覆盖。
  混合架构部署时用 `--platform` 分别构建,各自 `docker load` 到对应架构的机器上。

## 部署(昇腾 910B 单机,server 与 worker 同机)

以下步骤已在 192.168.1.110(openEuler 24.03 aarch64、8×910B2、Docker 18.09)上
从零执行验证过。多机部署时,其他机器只需执行第 4 步(worker),`GPUSTACK_SERVER_URL`
指向 server 所在机器即可。

### 0. 前提

- 昇腾驱动和固件已装好,`npu-smi info` 能看到全部卡。
- 已安装 Docker 和 git(openEuler 上用 `dnf install -y git` 安装 git)。
- 机器能访问 Docker Hub(官方基础镜像)、quay.io(vLLM-Ascend 推理镜像)、
  华为 SWR(benchmark 基础镜像)和 huggingface.co。访问不到时可以换成镜像源,
  见下文"基准测试执行端镜像"和第 1 步的 `NPM_REGISTRY`。

下面用到的路径:

| 路径 | 内容 |
| --- | --- |
| `/data/gpustack/src` | 源码 |
| `/data/gpustack/server` | server 数据:数据库、admin 初始密码、worker 注册 token |
| `/data/gpustack/worker` | worker 数据:模型缓存、日志 |

### 1. 获取代码并构建镜像

```bash
mkdir -p /data/gpustack && cd /data/gpustack
git clone -b custom/v2.3 https://github.com/shijunan126-creator/gpustack.git src
cd src
deploy/build.sh "" --build-arg NPM_REGISTRY=https://registry.npmmirror.com
```

镜像名是 `gpustack-custom:v2.3.0-<提交号>`,后面的命令用 `TAG` 变量代替:

```bash
TAG=gpustack-custom:v2.3.0-$(git rev-parse --short HEAD)
docker image inspect "$TAG" > /dev/null && echo "$TAG"
```

### 2. 启动 server

`GPUSTACK_TOKEN` 是 worker 注册用的 token,这里随机生成一个。只跑 server 的容器
必须带上它:server 用它创建默认集群并写出 token 文件(第 3 步);不带的话不会创建
默认集群,只能到页面的"添加 Worker"里取 token。

```bash
docker run -d --name gpustack --restart unless-stopped \
  -p 80:80 \
  -v /data/gpustack/server:/var/lib/gpustack \
  -e GPUSTACK_TOKEN="$(openssl rand -hex 24)" \
  "$TAG"
```

等它就绪(首次启动要初始化数据库,大约一两分钟):

```bash
until curl -fs http://127.0.0.1/healthz > /dev/null; do sleep 5; done; echo ready
```

### 3. 取 admin 初始密码和 worker 注册 token

```bash
cat /data/gpustack/server/initial_admin_password   # 用户名 admin,首次登录后要求改密码
cat /data/gpustack/server/token                    # worker 注册用
```

### 4. 启动 worker

worker 和 server 用**同一个镜像**。`--network host` 是必须的:推理容器直接监听宿主机端口。
`/var/run/docker.sock` 用来创建推理容器和 benchmark 容器;`/data:/data` 让页面上可以选择
宿主机 `/data` 下已有的模型目录。

```bash
SERVER_URL=http://<本机IP>      # 例如 http://192.168.1.110,不要写 127.0.0.1
docker run -d --name gpustack-worker --restart unless-stopped \
  --network host --privileged \
  $(for d in /dev/davinci[0-9]* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do printf -- '--device %s ' "$d"; done) \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /data/gpustack/worker:/var/lib/gpustack \
  -v /data:/data \
  -e GPUSTACK_SERVER_URL="$SERVER_URL" \
  -e GPUSTACK_TOKEN="$(cat /data/gpustack/server/token)" \
  "$TAG"
```

worker 启动后会在后台自动构建 benchmark 执行端镜像 `aisbench-runner:<版本>`,
首次在 110 上用了约 12 分钟,大部分时间在从华为 SWR 拉基础镜像。构建期间创建的 benchmark
会保持"排队中",构建完成后自动开始。进度见:

```bash
tail -f /data/gpustack/worker/log/benchmarks/runner-image-build.log
```

### 5. 检查

```bash
docker logs gpustack-worker 2>&1 | grep "registered with worker_id"   # worker 已注册
docker images | grep aisbench-runner       # runner 镜像构建完成后出现
```

在浏览器打开 `http://<本机IP>`,用 admin 登录,Worker 列表里应能看到本机(状态 Ready)和全部 NPU。
部署模型时,"查看日志"默认只显示 vLLM serve 自身的输出;勾选"完整日志"可以看到模型下载
和拉镜像的过程,部署在推理容器启动前就失败时需要看这个。

### 卸载

会删除模型缓存和数据库,不可恢复:

```bash
docker rm -f gpustack gpustack-worker
# 由 GPUStack 创建的推理容器和 benchmark 容器
docker ps -aq --filter label=runtime.gpustack.ai/workload | xargs -r docker rm -f
rm -rf /data/gpustack
```

## 升级与回滚

升级:

```bash
cd /data/gpustack/src && git pull && deploy/build.sh "" --build-arg NPM_REGISTRY=https://registry.npmmirror.com
TAG=gpustack-custom:v2.3.0-$(git rev-parse --short HEAD)
docker stop gpustack-worker gpustack
docker rename gpustack gpustack-old && docker rename gpustack-worker gpustack-worker-old
```

然后用新的 `TAG` 重新执行第 2 步和第 4 步。数据目录不变,所以模型、用户和配置都会保留。
第 2 步里新随机生成的 `GPUSTACK_TOKEN` 不会替换已有的 token:集群的 token 存在数据库里,
server 启动时会把它写回 `/data/gpustack/server/token`,所以第 4 步照常读这个文件即可。
确认新版本正常后,再 `docker rm gpustack-old gpustack-worker-old`。

回滚:`docker rm -f gpustack gpustack-worker`,再把 `gpustack-old`、`gpustack-worker-old`
改回原名并 `docker start`。

自己改启动参数时,**换容器之前先用 `docker inspect` 对比新旧容器的完整 HostConfig 和 Env**,
只允许镜像不同。漏掉设备映射、挂载这类参数,模型会表现为"没有可用的后端版本"
或"Failed to infer device type"之类的问题。

## 限制

- 改了 `pyproject.toml` 里的**依赖**不会生效:镜像沿用官方镜像的 Python 环境。需要时
  改走官方 `pack/Dockerfile` 做完整构建。
- 改了 `charts/` 不会生效:Helm chart 取自官方镜像。
- 数据库没有新增迁移;移除"模型提供商"后 `model_providers` 表和各表的 `provider_id`
  列仍在,只是不再使用。

## 开发

- 后端测试:`pytest tests`(`tests/k8s/test_bootstrap_manifest.py` 需要先打包 Helm chart,
  源码仓库里会失败,和本 fork 无关)。
- 前端:`cd ui && pnpm install && pnpm dev`,开发代理指向 `PROXY_HOST`
  (默认 `http://localhost`);`pnpm test`、`pnpm check:locales`。
- 改后端后跑 `python hack/check_imports.py`,可以抓到 linter 看不到的、写在函数里的失效 import。
