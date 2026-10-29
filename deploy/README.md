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


```bash
deploy/build-benchmark-runner.sh                               # 产出 aisbench-runner:<版本>
deploy/build-benchmark-runner.sh --platform linux/arm64        # 鲲鹏机器
deploy/build-benchmark-runner.sh --build-arg SHAREGPT_URL=https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

这个镜像不在任何公共仓库里,**每台 worker 都要有**:`docker save` / `docker load` 搬过去,
或推到内网 Registry 后用 worker 的 `--benchmark-image-repo <registry>/aisbench-runner:<版本>` 指过去。
worker 找不到镜像时,测试会停在拉取镜像失败。

`gpustack save-images` / `copy-images` **不包含这个镜像**:它们只从 registry 拉取
(`skopeo copy docker://…`),本地构建的镜像列进去会让整次 save 失败。所以离线部署时必须按
上面两种方式单独搬运。

还有两处容易踩的地方:

- 配了 `--system-default-container-registry` 时,worker 会把裸名改写成
  `<registry>/aisbench-runner:<版本>`。此时只 `docker load` 裸名是对不上的:要么按这个名字
  `docker tag` 后推上去,要么显式设 `--benchmark-image-repo` 指向实际名字。
- amd64 与 arm64 用的是同一个 tag;把两种架构推到同一个 registry tag 会互相覆盖。
  混布架构时用 `--platform` 分别构建,各自 `docker load` 到对应架构的机器上。

## 运行

Server(数据在命名卷 `gpustack-data`,升级/回滚不丢数据):

```bash
docker run -d --name gpustack --restart unless-stopped \
  -p 80:80 -v gpustack-data:/var/lib/gpustack \
  gpustack-custom:<tag>
```

Worker(与 server 用**同一个镜像**;`<SERVER_URL>`、`<TOKEN>` 在 server 页面的"添加 Worker"里取):

```bash
docker run -d --name gpustack-worker --restart unless-stopped \
  --runtime nvidia --network host --privileged \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v gpustack-worker-data:/var/lib/gpustack \
  gpustack-custom:<tag> \
  --server-url <SERVER_URL> --token <TOKEN> --advertise-address <本机IP>
```

`--runtime nvidia` 只用于 NVIDIA 机器。**昇腾机器的 worker 参数(设备映射等)尚未用本镜像
验证过**,按官方文档的昇腾安装说明来,并先在一台机器上试通再铺开。

## 升级与回滚

升级:构建新 tag → 停掉并改名旧容器 → 用相同参数起新容器。**换容器之前先
`docker inspect` 对比旧容器的完整 HostConfig 和 Env**,只允许镜像和主机名不同;
漏掉 `--runtime`、挂载这类参数,模型会表现为"没有可用的后端版本"之类的问题。

回滚:停掉新容器,用旧 tag 按同样参数起回来(数据卷是同一个)。

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
