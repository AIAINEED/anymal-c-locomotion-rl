# 部署栈：ROS2 + 硬件约束验证（第 ③ 阶段）

> sim2real 的四步：① 仿真训练 ✅ → ② 跨引擎 sim2sim ✅ → ③ **部署就绪栈（本目录）** → ④ 真机验证（无硬件）
>
> 本目录完成第 ③ 阶段：把策略做成**可与真机对接的实时控制器**，并在仿真里量化它的**部署边界**。

---

## 1. 本地闭环结果（已实测）

不依赖 ROS2 的纯 Python 闭环（`closed_loop.py`）：策略 ↔ MuJoCo 被控对象，50 Hz 控制周期。

| 计算开销 | 数值 |
|---|---|
| 单次策略推理（ONNX, CPU）p99 | **0.26 ms** |
| 整环耗时（推理 + 4 步物理）p99 | **1.2 ms** |
| 占 20 ms 控制周期的比例 | **≈ 6%** ✅ 实时余量充足 |

**观测延迟容限**（每个配置 15 个 20 秒 episode，随机速度指令）：

| 观测延迟 | 摔倒率 | 平均存活 | 跟踪误差 |
|---|---|---|---|
| 0 ms | **0%** | 20.0 s | 0.075 m/s |
| 20 ms | **0%** | 20.0 s | 0.073 m/s |
| 40 ms | **0%** | 20.0 s | 0.074 m/s |
| 60 ms | **0%** | 20.0 s | 0.092 m/s |
| 80 ms | 26.7% | 16.3 s | 0.164 m/s |
| 120 ms | 73.3% | 11.1 s | 0.222 m/s |
| 160 ms | 100% | 6.1 s | 0.275 m/s |

**结论：延迟容限 ≈ 80 ms**（60 ms 以内完全无退化，80 ms 起开始摔倒，160 ms 完全失效）。
→ 真机部署时，控制周期 20 ms + 感知/估计链路 **总延迟应控制在 60 ms 以内**。

**状态估计误差敏感性**（20 ms 延迟下，给机体系线速度加高斯噪声 —— 真机上 `base_lin_vel` 来自状态估计器，是主要误差源）：

| 估计噪声 σ | 摔倒率 | 跟踪误差 |
|---|---|---|
| 0（基线） | 0% | 0.073 m/s |
| 0.1 m/s | 6.7% | 0.115 m/s |
| 0.2 m/s | 13.3% | 0.157 m/s |
| 0.3 m/s | 13.3% | 0.192 m/s |

**结论**：速度估计误差 σ ≥ 0.1 m/s 就会带来可观测的性能损失；真机部署应优先保证状态估计精度
（这正是 sim2real 中最常见的失败原因之一）。

![deployment limits](../assets/deployment_limits.png)

### 跨平台复现（Windows 原生 vs WSL2/Ubuntu）

同一份代码、同一随机种子，在两套环境各跑 15 个 episode：

| 观测延迟 | Windows 摔倒率 / 跟踪误差 | WSL2 摔倒率 / 跟踪误差 | 一致性 |
|---|---|---|---|
| 0 ms | 0% / 0.0749 | 0% / **0.0749** | 逐位一致 |
| 20 ms | 0% / 0.0727 | 0% / **0.0727** | 逐位一致 |
| 40 ms | 0% / 0.0741 | 0% / **0.0741** | 逐位一致 |
| 60 ms | 0% / 0.0919 | 0% / **0.0919** | 逐位一致 |
| 80 ms | 26.7% / 0.164 | 13.3% / 0.127 | 混沌区差异 |
| 120 ms | 73.3% / 0.222 | 86.7% / 0.325 | 混沌区差异 |
| 160 ms | 100% / 0.275 | 100% / 0.288 | 一致（都失效） |

* 延迟 ≤ 60 ms 时两平台结果**完全一致** → 核心逻辑（观测拼装 / 动作转换 / 时序）跨平台等价 ✓
* 超过延迟容限后进入**混沌区**：摔倒与否对浮点/编译器差异敏感，单点数值有波动属正常 ——
  因此结论应按**阈值**表述（"容限 ≈ 80 ms"），而非纠结单点数值。
* 算力：WSL2 上推理 p99 **0.055 ms**、整环 p99 **0.38 ms**（占 20 ms 周期的 **1.9%**）。

### ROS2 闭环实测（第②层：WSL2 + ROS2 Humble，两节点通过话题通信）

两个节点分别运行（`ros2_sim_node.py` 被控对象 + `ros2_policy_node.py` 控制器），实测：

| 指标 | 实测值 | 说明 |
|---|---|---|
| 控制周期（策略节点自测） | 均值 **20.00 ms**，p99 **20.50 ms** | 抖动 2.5%，守住 50 Hz |
| 单次策略推理 | 均值 **0.09 ms**，p99 **0.16 ms** | 占 20 ms 周期的 **0.45%** |
| `/joint_command` 发布频率（`ros2 topic hz`） | **49.3–49.4 Hz** | 略低于 50 Hz，来自 ROS2 timer + DDS 开销 |
| 话题 | `/joint_states` `/imu` `/odom` `/cmd_vel` `/joint_command` | 均为预期类型 |

**一个真实部署教训**：`ros2 topic hz` 观察到一次 **0.596 s** 的发布间隔离群（WSL 调度抖动；std dev 12 ms 说明整体稳定）。
→ 真机部署必须加入**看门狗/心跳监控**：控制周期超时立即进入安全状态，不能盲目继续下发指令。

**复现命令**：

```bash
# 终端 1
source /opt/ros/humble/setup.bash && cd ~/anymal_deploy
python3 ros2_sim_node.py --model ~/mujoco_menagerie/scene.xml

# 终端 2
source /opt/ros/humble/setup.bash && cd ~/anymal_deploy
python3 ros2_policy_node.py --policy policy.onnx --cmd 0.8,0,0

# 终端 3
ros2 topic hz /joint_command
ros2 topic echo /joint_states --once
```

---

## 2. 文件

| 文件 | 作用 | 依赖 |
|---|---|---|
| `anymal_core.py` | **核心**：关节顺序映射、48 维观测拼装、动作→关节目标、IMU→重力方向 | numpy, onnxruntime |
| `anymal_plant.py` | MuJoCo 被控对象（位置指令进、关节/机体状态出，200 Hz 物理） | numpy, mujoco |
| `closed_loop.py` | **本地闭环评估**：控制周期、推理延迟、观测延迟/估计噪声鲁棒性 | 上面两个 |
| `ros2_policy_node.py` | **ROS2 策略节点**（50 Hz 控制器） | rclpy + 核心 |
| `ros2_sim_node.py` | **ROS2 被控对象节点**（MuJoCo 扮演真机，模拟传感器） | rclpy + 上面两个 |

> `anymal_core.py` 是唯一需要被真机复用的文件 —— 换真机时只需替换"状态来源"（IMU/状态估计器）
> 和"指令出口"（力矩接口），本文件里的约定保持不变。

---

## 3. ROS2 接口

| 话题 | 类型 | 方向 | 内容 |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | 被控对象 → 策略 | 12 关节角/角速度（顺序 = 按腿分组，`LF_HAA…RH_KFE`） |
| `/imu` | `sensor_msgs/Imu` | 被控对象 → 策略 | 角速度 + 姿态（策略从中算重力方向） |
| `/odom` | `nav_msgs/Odometry` | 被控对象 → 策略 | **机体系线速度**（真机：状态估计器输出） |
| `/cmd_vel` | `geometry_msgs/Twist` | 操作者 → 策略 | `linear.x/linear.y` = vx/vy，`angular.z` = wz |
| `/joint_command` | `sensor_msgs/JointState` | 策略 → 被控对象 | 12 关节目标角（位置控制） |

**设计说明**：真机上 `base_lin_vel` 无法直接测量（腿式机器人），必须由**状态估计器**提供 —— 所以接口设计成
`/odom` 而不是把速度塞进 `/joint_states`；`projected_gravity` 则来自 IMU 姿态。这两点是与真机对接的关键约定。

---

## 4. 运行（WSL2 + ROS2 Humble，不需要真机）

> 实验室电脑的 WSL 是 Ubuntu 22.04，ROS2 Humble 可用 apt 直装。

```bash
# ① 装 ROS2（约 500 MB，一次性）
sudo apt update && sudo apt install -y ros-humble-ros-base python3-pip
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source /opt/ros/humble/setup.bash

# ② 装 Python 依赖（纯 CPU）
pip install numpy mujoco onnxruntime

# ③ 准备文件：把 deploy/ 目录、policy.onnx、MuJoCo 的 anymal_c 模型放进 WSL
#    Windows 盘在 WSL 下是 /mnt/c、/mnt/e；<WIN_PROJECT> = 仓库在 Windows 上的路径，
#    例如 /mnt/c/Users/<用户名>/Desktop/anymal-c-locomotion-rl
mkdir -p ~/anymal_deploy && cp -r "<WIN_PROJECT>/deploy/"* ~/anymal_deploy/
cp "<WIN_PROJECT>/policy.onnx" ~/anymal_deploy/
cp -r /mnt/e/mujoco_menagerie ~/mujoco_menagerie

# ④ 起被控对象节点（终端 1）
cd ~/anymal_deploy
python3 ros2_sim_node.py --model ~/mujoco_menagerie/scene.xml

# ⑤ 起策略节点，固定前进指令（终端 2）
source /opt/ros/humble/setup.bash && cd ~/anymal_deploy
python3 ros2_policy_node.py --policy policy.onnx --cmd 0.8,0,0

# ⑥ 观察：控制频率、关节状态、指令下发（终端 3）
ros2 topic hz /joint_command          # 应为 ~50 Hz
ros2 topic echo /joint_states --once
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5}, angular: {z: 0.4}}" -r 10
```

**延迟注入测试**（验证第 1 节的延迟容限在 ROS2 链路上同样成立）：

```bash
python3 ros2_policy_node.py --policy policy.onnx --cmd 0.8,0,0 --delay_steps 4   # 80 ms 延迟 → 应开始摔
```

---

## 5. 与真机的差距（诚实边界）

| 项 | 本目录状态 | 真机所需 |
|---|---|---|
| 策略推理 | ✅ ONNX / CPU，0.26 ms | 同（或 TensorRT） |
| 控制频率 | ✅ 50 Hz，p99 整环 1.2 ms | 同 |
| 消息接口 | ✅ ROS2 话题已定义 | 对接厂商 SDK（ANYdrive → 力矩接口） |
| 状态估计 | ⚠️ 用 MuJoCo 真值 + 噪声仿真 | 需实际状态估计器（腿部里程计/IMU 融合） |
| 执行器 | ⚠️ PD 近似 LSTM 执行器网络 | ANYdrive 力矩接口 + 安全限幅 |
| 硬件 | ❌ 无 | ANYmal-C（或同执行器平台） |

**结论**：本目录把策略做成了"只差硬件"的形态 —— 接口、频率、算力、延迟容限都已量化；
第 ④ 步（真机）需要硬件接入，属于明确的后续工作。
