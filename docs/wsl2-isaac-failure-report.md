# Isaac Lab on WSL2 —— 完整问题记录（给接手的人 / 其他 AI）

> 目的：把一次"在 Windows 11/10 + WSL2 上从零安装 Isaac Sim 5.1 + Isaac Lab 并尝试跑四足 RL 训练"的全过程记录下来。
> 结果：**安装成功、空场景跑通，但训练跑不了**，根因是 WSL2 的 GPU 能力不满足 Isaac Sim 5.1 的要求。
> 本文所有命令均为实际执行过的，报错为原文粘贴（部分过长处截断）。请勿重复已排除的假设。

---

## 0. 一句话结论

> **WSL2 里 Isaac Sim 5.1 + Isaac Lab 可以安装成功、`create_empty.py --headless` 可以跑通，但 `train.py` 训练跑不了。**
> 两个相互独立的组件在同一处撞墙：
> 1. `gpu.foundation.plugin` 建不了 GPU 设备（错误信息明确写缺少 **Vulkan ray_tracing** 能力）
> 2. **Warp**（Isaac Sim 5.x 的硬依赖）建不了 CUDA stream（`Failed to create stream on device cuda:0.0`）
>
> 而 WSL 里**不能安装原生 NVIDIA Linux 驱动**（会破坏 GPU 直通），所以这不是配置问题，是平台限制。

---

## 1. 环境

| 项 | 值 |
|---|---|
| 宿主 OS | Windows 10 Pro 22H2（Build 19045） |
| 宿主驱动 | 610.88（KMD 610.88，CUDA UMD 13.3） |
| WSL | Ubuntu 22.04.5 LTS，WSL2 |
| WSL 内 nvidia-smi | 610.57.01（WSL 侧驱动桩，正常） |
| GPU | NVIDIA GeForce RTX 4060，**8188 MiB** |
| CPU / 内存 | i5-11400（6 核 12 线程）/ 32 GB（WSL 分配 24 GB） |
| glibc | 2.35 |
| 已有环境 | miniconda（base 处于激活状态）+ `legged_env`（Isaac Gym 老栈） |
| 目标 | Isaac Sim 5.1 + Isaac Lab，跑 `Isaac-Velocity-Flat-Anymal-C-v0`（rsl_rl PPO） |

### 安装方式
- **pip 路线**（不是二进制 zip）：`isaacsim[all,extscache]==5.1.0`，装在 **venv**（不是 conda）
- Isaac Lab 源码：从同一台机器的 Windows 分区 `E:\IsaacLab` 拷贝到 WSL（**这一步引入了坑 #1**）
- WSL 内核版本：未记录（可补 `uname -r`）

---

## 2. 已完成的部分（干净、可复用）

```bash
# 1) 系统依赖 + Python 3.11（Ubuntu 22.04 自带 3.10，Isaac Sim 5.x 要求 3.11）
sudo apt install -y build-essential cmake git curl wget software-properties-common \
  libgl1 libglu1-mesa libxi6 libxrandr2 libxinerama1 libxcursor1 libsm6 libice6 libxt6 libfontconfig1
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# 2) venv + Isaac Sim + torch
mkdir -p ~/venvs && python3.11 -m venv ~/venvs/isaaclab
source ~/venvs/isaaclab/bin/activate
python -m pip install -U pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --retries 5 --timeout 60 "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 3) Isaac Lab（从 Windows 拷贝后）
(cd /mnt/e/IsaacLab && tar cf - --exclude=.git --exclude=logs --exclude=_isaac_sim --exclude=__pycache__ .) \
  | (mkdir -p ~/isaac/IsaacLab && cd ~/isaac/IsaacLab && tar xf -)
```

**最终版本状态**

| 组件 | 版本 |
|---|---|
| isaacsim | 5.1.0.0 |
| torch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| Python | 3.11.16 |
| Isaac Lab 源码包 | isaaclab / isaaclab_assets / isaaclab_contrib / isaaclab_mimic / isaaclab_rl / isaaclab_tasks（6/6 全部装上） |
| rsl-rl-lib | 3.1.2（与 Isaac Lab 的 `source/isaaclab_rl/setup.py` 钉的版本一致） |
| omni.physx.fabric | 107.3.26（随 extscache，**没有**从 registry 拉取的第二份） |

**验证过的成功项**

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
# 日志结尾：
#   [INFO] Using python from: /home/df/venvs/isaaclab/bin/python
#   [INFO][AppLauncher]: Using device: cuda:0
#   [INFO][AppLauncher]: Loading experience file: .../isaaclab.python.headless.kit
#   00:31:54 [simulation_context.py] WARNING: ... PhysxCfg ...
# 进程持续运行 8 分钟被 timeout 杀掉（该脚本本身是 while is_running() 死循环，不退出=正常）
```

---

## 3. 踩到的坑（时间序，症状 → 根因 → 处置）

### 坑 1：CRLF 换行

**症状**
```
/usr/bin/env: 'bash\r': No such file or directory
```
**根因**：从 Windows 分区拷贝的仓库，所有 `.sh` 是 CRLF，shebang 变成 `#!/usr/bin/env bash\r`。
**处置**
```bash
find ~/isaac/IsaacLab -type f -name "*.sh" -exec sed -i 's/\r$//' {} +
find ~/isaac/IsaacLab -type f -name "*.sh" -exec chmod +x {} +
```
**结果**：✅ 解决

---

### 坑 2：`isaaclab.sh` 用了 conda base 的 Python（最隐蔽的一个）

**症状**
```bash
./isaaclab.sh --install rsl_rl
# [INFO] Installing torch==2.7.0 and torchvision==0.22.0 (cu128) from https://download.pytorch.org/whl/cu128...
# Looking in indexes: https://download.pytorch.org/whl/cu128
# ERROR: Could not find a version that satisfies the requirement torch==2.7.0
#        (from versions: 2.9.0+cu128, 2.9.1+cu128, 2.10.0+cu128, 2.11.0+cu128)
# ERROR: No matching distribution found for torch==2.7.0
```
（手工 `pip install torch==2.7.0+cu128` 却能成功 —— 这是关键线索）

**根因**：`isaaclab.sh` 选解释器的优先级是

```
CONDA_PREFIX  →  VIRTUAL_ENV  →  _isaac_sim
  第180行          第183行        第186行
```

conda base 一直处于激活状态 → `CONDA_PREFIX=/home/df/miniconda3` 被优先命中 → 脚本用 **miniconda base 的 Python（3.14）** 去解析依赖。
而 **torch 2.7.0 没有 cp314 轮子**，所以 pip 只列出 2.9.0+（2.9.0 是第一个带 cp314 的版本）。

**排查过程（留档，避免重复）**
- 一度怀疑官方把 2.7.0 从 cu128 源下架 → **实测：文件仍在**（`torch-2.7.0+cu128-cp311-...manylinux_2_28_x86_64.whl` 存在，未被 yank）
- 一度怀疑索引页被截断 → 不是
- 决定性证据：pip 的候选列表**恰好从 2.9.0 起跳**，而 2.9.0 正是第一个有 `cp314` 轮子的版本 → 说明解释器是 3.14

**处置**
```bash
conda deactivate                       # 让 CONDA_PREFIX 变空
conda config --set auto_activate_base false   # 永久生效（重开终端）
echo "CONDA_PREFIX=[${CONDA_PREFIX:-空}]"     # 必须为空
```
**结果**：✅ 全部包装进了 `/home/df/venvs/isaaclab/lib/python3.11/site-packages`

---

### 坑 3：`flatdict==4.0.1` 构建失败 → 核心包 `isaaclab` 没装上

**症状**：5 个源码包装上了，唯独核心的 `isaaclab` 没有；`import isaaclab` 报 `ModuleNotFoundError`。
```bash
./isaaclab.sh --install rsl_rl
#   × Getting requirements to build wheel did not run successfully.
#     File "<string>", line 1, in <module>
#   ModuleNotFoundError: No module named 'pkg_resources'
# ERROR: Failed to build 'flatdict' when getting requirements to build wheel
```
**根因**：`source/isaaclab/setup.py:45` 钉了 `"flatdict==4.0.1"`；而 **flatdict 4.0.1 在 PyPI 上只有 sdist（没有 wheel）**，必须现场构建。它的 `setup.py` 第一行是 `import pkg_resources`，而 pip 的**隔离构建环境**里装的是新版 setuptools，**已不再提供 `pkg_resources`**。（venv 自己的 setuptools 是 79.0.1，有 `pkg_resources`）

**处置（非隔离构建，用 venv 的 setuptools）**
```bash
pip install "flatdict==4.0.1" --no-build-isolation
# 然后重跑
./isaaclab.sh --install rsl_rl
```
**备选方案**（如果还有别的老 sdist 报同样错）
```bash
echo "setuptools==79.0.1" > ~/pip-constraints.txt
export PIP_CONSTRAINT=$HOME/pip-constraints.txt
```
**结果**：✅ 6/6 源码包全部装上

---

### 坑 4：`train.py` 卡住（CPU 99% / GPU 1%）—— `libcuda.so` 不在链接器搜索路径

**症状**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Anymal-C-v0 --headless --num_envs 256 --max_iterations 3
# 日志停在：
#   [INFO]: Base environment: ... Physics step-size : 0.005 ...
#   00:39:15 [terrain_importer.py] WARNING: Visual material ... ground plane ...
# 之后 20+ 分钟无输出
```
**排查数据**
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
# 1 %, 1025 MiB                     ← GPU 几乎闲着
top -b -n1 | head -15
# python 120% CPU × 2，累计 CPU 时间 19min / 27min
```
```bash
# 用 py-spy 抓 Python 栈（需 sudo）
sudo "$(which py-spy)" dump --pid <PID>
#   initialize_physics (simulation_manager/impl/simulation_manager.py:237)
#   _warm_start        (simulation_manager/impl/simulation_manager.py:166)
#   play / reset       (simulation_context.py:914 → 644 → isaaclab/sim/simulation_context.py:519)
#   __init__ (manager_based_env.py:173)
```
**根因**
```
$ python -c "import ctypes; ctypes.CDLL('libcuda.so')"
OSError: libcuda.so: cannot open shared object file: No such file or directory

$ ldconfig -p | grep -i libcuda
        libcuda.so.1 (libc6,x86-64) => /usr/lib/wsl/lib/libcuda.so.1
        # 注意：没有 libcuda.so 这一条
```
- 文件 `/usr/lib/wsl/lib/libcuda.so` **存在**（187,984 字节的 WSL 桩库）
- 但 **`ldconfig` 只按 SONAME 注册了 `libcuda.so.1`**；`dlopen("libcuda.so")` 是按**精确文件名**查，只会去**默认搜索目录**找
- 而 WSL 的库目录 `/usr/lib/wsl/lib` 不在默认搜索目录里（也没有进 ld.so.conf）
- 结果：PhysX 无法初始化 GPU 管线 → **退化为 CPU 物理** → 256 环境在 CPU 上爬

参考资料：[IsaacLab #3497](https://github.com/isaac-sim/IsaacLab/issues/3497)（官方维护者结论：问题在宿主侧 CUDA 驱动注入，核心检查就是 `ctypes.CDLL("libcuda.so")`）

**处置**
```bash
sudo ln -sf /usr/lib/wsl/lib/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so
sudo ldconfig
python3 -c "import ctypes; ctypes.CDLL('libcuda.so'); print('OK')"   # ✅
```
运行训练时再加一层保险：
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```
**结果**：✅ GPU 利用率从 1% 升到 **77%**，PhysX 真的用上 GPU 了
**副作用**：暴露了下面的坑 5

---

### 坑 5：`omni.physx.fabric` 的 CUDA illegal memory access

**症状**
```
[Error] [omni.physx.fabric.plugin] CUDA error: an illegal memory access was encountered:
        ../../../extensions/runtime/source/omni.physx.fabric/plugins/DirectGpuHelper.cpp: 563
        （566 / 569 / 572 / … 重复报）
[Warning] [omni.physx.plugin] USD stage detach not called, holding a loose ptr to a stage!
[Warning] [omni.physx.plugin] PhysX warning: .../PxgCudaMemoryAllocator.cpp, LINE 68
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```
**排查**：确认不存在"registry 拉下来的第二份 fabric 导致版本冲突"（论坛常见病因）——
```bash
ls -d ~/.local/share/ov/data/exts/v2/omni.physx*     # 空
ls -d ~/venvs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.physx.fabric*
# …/extscache/omni.physx.fabric-107.3.26+107.3.3.lx64.r.cp311.u353   ← 唯一来源，无冲突
```
**处置尝试**：关掉 fabric
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Anymal-C-v0 --headless --num_envs 16 --max_iterations 2 \
  env.sim.use_fabric=False
```
**结果**：⚠️ 非法访存没了，但变成 **`PxgCudaMemoryAllocator` 告警刷屏 135 次**，仍未进入训练循环

---

### 坑 6（真正的根因）：`gpu.foundation.plugin` 建不了设备

**症状**（在日志更前面，容易被刷屏内容淹没）
```
2026-09-30T16:42:04Z [5,338ms] [Error] [gpu.foundation.plugin] No device could be created. Some known system issues:
- The driver is not installed properly and requires a clean re-install.
- Your GPUs do not support RayTracing: DXR or Vulkan ray_tracing, or hardware is excluded due to performance.
- The driver cannot enumerate any GPU: driver, display, TCC mode or a docker issue.
  For Vulkan, test it with Vulkaninfo tool from Vulkan SDK, instead of nvidia-smi.
- For Ubuntu, it requires server-xorg-core 1.20.7+ and a display to work without --no-window.
- For Linux dockers, the setup is not complete. Install the latest driver, xServer and NVIDIA container runtime.
2026-09-30T16:42:04Z [5,343ms] [Error] [gpu.foundation.plugin] Invalid getDeviceInfo parameters.
```
**含义**：`gpu.foundation` 是 Isaac Sim 的 GPU 基础层（PhysX / fabric 都用它），它需要**原生的 Vulkan 光追设备枚举能力**。它建不了设备 → 后面所有 GPU 分配都失败（就是那 135 条刷屏）。

**关键澄清**：`vulkaninfo --summary` 显示 **Vulkan Instance 存在（1.3.204）**，instance 扩展齐全 —— 但那是 **Mesa/Dozen 之类的软件/转译层**，**没有 NVIDIA 原生光追**。gpu.foundation 检查的正是 ray tracing 能力。

参考：[IsaacLab #2062](https://github.com/isaac-sim/IsaacLab/issues/2062)（原文贴出了上面这段完整列表）、[讨论 #3769](https://github.com/isaac-sim/IsaacLab/discussions/3769)

---

### 坑 7：`--device cpu` 也失败 —— Warp 建不了 CUDA stream

**症状**（试图把物理放到 CPU 来绕开 GPU）
```
RuntimeError: Failed to create stream on device cuda:0.0
  File ".../omni.warp.core-1.8.2+lx64/warp/context.py", line 2961, in _init_streams
    self.set_stream(Stream(self))
  File ".../warp/context.py", line 2720, in __init__
    raise RuntimeError(f"Failed to create stream on device {device}")
```
**含义**：Warp 是 Isaac Sim 5.x 的**硬依赖**，它需要 CUDA stream。即使物理放 CPU，Warp 仍然去 `cuda:0` 建 stream 并失败 → 训练无法继续。
（exit=0，0 个 iteration）

---

## 4. 最终判定

**两个相互独立的组件，在同一处撞墙 → 平台级限制**

| 组件 | 结果 |
|---|---|
| Kit 渲染器 / `create_empty.py --headless` | ✅ 能跑 |
| torch 的 CUDA | ✅ `torch.cuda.is_available() == True` |
| `gpu.foundation`（PhysX/fabric 的 GPU 基础层） | ❌ 建不了设备（缺 Vulkan ray_tracing） |
| Warp（Isaac Sim 5.x 核心） | ❌ 建不了 CUDA stream |
| PhysX GPU 物理 | ❌ 因此不可用 |

**为什么 WSL 上无解**：
- WSL2 的 GPU 支持本质是 **CUDA 直通**（+ 有限的 D3D12→GL / Mesa Vulkan 转译），**不提供 Isaac Sim 需要的 Vulkan 光追设备**
- WSL 里**不能安装原生 NVIDIA Linux 驱动**（会破坏直通，`nvidia-driver-*` 是禁区）
- 结论：**这是平台限制，不是配置错误；继续调参/换版本/改环境变量都解决不了**

**注意一个容易误判的点**：宿主 Windows 上的 GPU 栈（D3D12 + RTX 4060 + 驱动 610.88）**本来是正常的**。WSL 里失败不代表 Windows 里失败。

---

## 5. 已排除的假设（请勿重复排查）

| 假设 | 排除依据 |
|---|---|
| 官方把 `torch==2.7.0` 从 cu128 源下架了 | 实测文件仍在（未被 yank）：`torch-2.7.0+cu128-cp311-...manylinux_2_28_x86_64.whl` |
| PyTorch 索引页被截断 | 直接抓取索引页，2.7.0 在列 |
| 卡住是 Warp 首次 JIT 编译 | `grep -inE "warp\|nvcc\|compil"` 在两个日志里**零命中**；`~/.nv/ComputeCache` 无新增 |
| 是显存不足（8 GB） | 崩在 `--num_envs 16` 时也一样；分配器失败的上游是 gpu.foundation 建不了设备 |
| 是 fabric 插件版本冲突（论坛常见） | 只有一份 fabric（extscache 107.3.26），registry 目录为空 |
| 是 conda/OpenSSL/DLL 老问题 | 换成 venv 后 `import ssl`、`import torch`、`import isaacsim` 全部正常 |
| 关闭 fabric 就能跑 | 关掉后非法访存消失，但分配器刷屏，仍未进入训练循环 |
| `--device cpu` 能绕开 | Warp 仍要 CUDA stream，失败 |

---

## 6. 当前状态与下一步

**当前状态**
- WSL 里 `~/venvs/isaaclab`（Isaac Sim 5.1 + torch + Isaac Lab 6 包）**安装完整**
- `create_empty.py --headless` **跑得通**
- `train.py` **跑不通**（平台限制）
- 已落地的修复：CRLF 转换、`auto_activate_base=false`、`flatdict --no-build-isolation`、
  `libcuda.so` 软链 + `ldconfig`、运行前 `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`

**下一步计划（按优先级）**
1. **Windows 原生 + Isaac Sim 二进制 zip 安装**（可以全程远程桌面完成，不需重启机器）
   - 理由：Windows 的 GPU 栈正常；二进制版用 Isaac Sim 自带 Python，`isaaclab.bat` 优先取 `_isaac_sim\python.bat`（脚本第 186 行逻辑），可绕开"conda python + pip torch 混装"这条曾导致 Windows 侧崩溃的路径
   - 步骤：下载 5.1 zip → 解压 `C:\isaacsim` → 接受 EULA → `mklink /D E:\IsaacLab\_isaac_sim C:\isaacsim` → `isaaclab.bat -i rsl_rl` → 空场景 → 训练
2. 租 Linux GPU 云主机（AutoDL 之类，4090 约 ¥1.5–2/小时）：流程与本文 1–2 节相同
3. 物理接触那台机器时再装**原生 Ubuntu 双系统**（注意：**远程桌面环境下无法安装双系统** —— 重启即失联，消费级主板没有 IPMI/KVM-over-IP，有把机器搞成"失联+无法引导"的风险）

**注**：`~/isaac/IsaacLab` 与 `~/venvs/isaaclab` 建议先保留，别删——换到原生 Linux 时，Isaac Lab 源码可以直接复用（CRLF、flatdict 这两个坑的处置也照抄）。

---

## 7. 给其他 AI 的开放问题

1. 有没有办法在 **WSL2** 上让 `gpu.foundation` 成功创建设备？（例如：让 Kit 走某种 minimal/软件渲染模式、`--kit_args` 里禁用 RTX 相关扩展、或配置 Mesa 的某种 ICD？）
   - 已知：`vulkaninfo --summary` 能列出 Vulkan **Instance**（1.3.204）与完整 instance 扩展；**未确认 Devices 段是否有 RTX 4060、以及是否有 `VK_KHR_ray_tracing_pipeline` / `VK_KHR_acceleration_structure`**
2. **Warp** 的 `Failed to create stream on device cuda:0.0` 有没有 workaround？（例如强制 Warp 用 CPU 设备 `wp.set_device("cpu")` / 环境变量，而让 Isaac Lab 继续跑）
   - 如果这条能成，`--device cpu` 或许真能跑通训练（慢但可用）
3. 有没有人在 **WSL2 上成功跑过 Isaac Sim 5.x 的 PhysX GPU 训练**？如果有，配置是什么（WSL 版本 / 驱动 / Isaac Sim 版本 / 特殊参数）？
4. 若以上都不行：**Windows 二进制 zip 路线**是否能绕开 Linux 侧的这些坑（WSL 的问题是 Vulkan 光追 + CUDA stream；Windows 走 D3D12，理论上不存在这两个问题）——需要实际验证。

---

## 附：常用命令速查

```bash
# 环境状态三连
echo "CONDA_PREFIX=[${CONDA_PREFIX:-空}]  VIRTUAL_ENV=[${VIRTUAL_ENV:-空}]"
python -c "import torch, isaaclab; print(torch.__version__, torch.cuda.is_available()); print(isaaclab.__file__)"
python -c "import ctypes; ctypes.CDLL('libcuda.so'); print('libcuda OK')"

# 空场景（不退出=成功）
cd ~/isaac/IsaacLab && source ~/venvs/isaaclab/bin/activate
timeout 480 ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
echo "exit=$?   # 124 = 活满超时 = 成功"

# 训练（本文中失败）
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
timeout 600 env PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Anymal-C-v0 --headless --num_envs 16 --max_iterations 2

# 日志位置
ls /tmp/isaaclab/logs/                                   # Isaac Lab 侧日志
find ~/venvs/isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs -name "kit_*.log" | tail

# py-spy 抓卡住的进程（需 sudo）
sudo "$(which py-spy)" dump --native --pid <PID>
```
